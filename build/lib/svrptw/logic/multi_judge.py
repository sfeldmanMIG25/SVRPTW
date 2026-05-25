"""Multi-judge VLM panel for solution-image pair scoring.

Combines three judge families into a single concurrent panel:
  1. OpenRouter free-tier vision committee (Tier-A models reused from
     `svrptw.logic.committee.TIER_A`).
  2. Gemini direct (single call via `google-genai` SDK; same auth path
     as `svrptw.council.proposer` / `svrptw.vivrp.assessor._GeminiBackend`).
  3. Anthropic Haiku fallback (lazy-imported `anthropic` SDK; degrades
     gracefully to a single error verdict when the SDK or
     `ANTHROPIC_API_KEY` is unavailable).

Each judge call is wrapped in try/except so a single failure never tanks
the panel. Aggregation is tier-weighted median across non-failed
verdicts; dissent flips True when max-min span > 0.3.
"""
from __future__ import annotations

import base64
import concurrent.futures as cf
import json
import logging
import os
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

_LOG = logging.getLogger("svrptw.logic.multi_judge")


# --- Tier weights (mirror committee.py for consistency) ----------------

_TIER_WEIGHTS: dict[str, float] = {
    "S": 0.5,
    "A": 1.0,
    "B": 0.6,
    "C": 0.4,
    "gemini": 1.0,   # Tier-A equivalent
    "haiku":  1.0,   # Tier-A equivalent
}


# --- Result dataclasses -------------------------------------------------

@dataclass
class JudgeVerdict:
    """One judge's response on a (image_a, image_b) pair."""

    judge_id: str               # e.g. "openrouter:google/gemma-4-31b-it:free"
    judge_kind: str             # "openrouter" | "gemini" | "anthropic"
    tier: str                   # "S"/"A"/"B"/"C" | "gemini" | "haiku"
    score: Optional[float]      # in [0,1]; None on failure
    rationale: str
    confidence: Optional[float]
    latency_s: float
    error: Optional[str] = None


@dataclass
class MultiJudgeConsensus:
    """Aggregate verdict across the panel."""

    score: Optional[float]              # tier-weighted median; None if 0 responders
    dissent: bool                       # span > 0.3 across non-failed verdicts
    n_responded: int
    n_failed: int
    span: Optional[float]               # max - min across responded
    contributing_tiers: list[str] = field(default_factory=list)


# --- Schema (single-image / pair score) --------------------------------

# Judges emit a SINGLE score for the (image_a vs image_b) pair: 1.0 means
# A is unambiguously better, 0.0 means B is unambiguously better, 0.5 is
# a toss-up. This matches the JUDGMENT_SCHEMA shape in openrouter_client
# and lets aggregation use a single weighted-median axis.
JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {
            "type": "number", "minimum": 0.0, "maximum": 1.0,
            "description": "P(Solution A is better than Solution B), 0 to 1",
        },
        "rationale": {"type": "string",
                      "description": "Two to four sentences justifying the score"},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0,
                       "description": "Self-reported certainty"},
    },
    "required": ["score", "rationale", "confidence"],
}


# --- Aggregation primitives --------------------------------------------

def _weighted_median(values: list[float], weights: list[float]) -> float:
    """Tier-weighted median; mirrors `committee._weighted_median`."""
    paired = sorted(zip(values, weights), key=lambda x: x[0])
    total = sum(w for _, w in paired)
    if total <= 0:
        # Fall back to raw median if all weights are zero.
        return statistics.median(values)
    cum = 0.0
    for v, w in paired:
        cum += w
        if cum >= total / 2:
            return v
    return paired[-1][0]


def aggregate_verdicts(
    verdicts: list[JudgeVerdict],
    *,
    dissent_threshold: float = 0.30,
) -> MultiJudgeConsensus:
    """Tier-weighted-median consensus + dissent flag across non-failed verdicts."""
    responded = [v for v in verdicts if v.score is not None and v.error is None]
    n_failed = len(verdicts) - len(responded)
    if not responded:
        return MultiJudgeConsensus(
            score=None, dissent=False, n_responded=0, n_failed=n_failed,
            span=None, contributing_tiers=[],
        )
    scores = [float(v.score) for v in responded]
    weights = [_TIER_WEIGHTS.get(v.tier, 0.5) for v in responded]
    median = _weighted_median(scores, weights)
    span = max(scores) - min(scores)
    return MultiJudgeConsensus(
        score=median,
        dissent=bool(span > dissent_threshold),
        n_responded=len(responded),
        n_failed=n_failed,
        span=span,
        contributing_tiers=sorted({v.tier for v in responded}),
    )


# --- OpenRouter judge ---------------------------------------------------

def _call_openrouter(
    model: str,
    prompt: str,
    image_a: Path,
    image_b: Path,
    *,
    timeout_s: float = 30.0,
) -> JudgeVerdict:
    """Call one OpenRouter Tier-A model on the (image_a, image_b) pair."""
    from svrptw.logic.openrouter_client import call_model

    judge_id = f"openrouter:{model}"
    t0 = time.perf_counter()
    try:
        resp = call_model(
            model, prompt, image_a,
            extra_image_paths=[image_b],
            schema=JUDGE_SCHEMA,
            timeout_s=timeout_s,
        )
    except Exception as e:  # noqa: BLE001 — defensive; one failure must not tank the panel
        return JudgeVerdict(
            judge_id=judge_id, judge_kind="openrouter", tier="A",
            score=None, rationale="", confidence=None,
            latency_s=time.perf_counter() - t0,
            error=f"{type(e).__name__}: {e}",
        )
    return JudgeVerdict(
        judge_id=judge_id, judge_kind="openrouter", tier="A",
        score=float(resp.score),
        rationale=str(resp.rationale or ""),
        confidence=float(resp.confidence) if resp.confidence is not None else None,
        latency_s=resp.latency_s or (time.perf_counter() - t0),
    )


# --- Gemini judge -------------------------------------------------------

def _call_gemini(
    prompt: str,
    image_a: Path,
    image_b: Path,
    *,
    timeout_s: float = 30.0,  # noqa: ARG001 — google-genai sets its own
) -> JudgeVerdict:
    """Single call to Gemini via the same SDK path as proposer.py."""
    judge_id = "gemini:unknown"
    t0 = time.perf_counter()
    try:
        # Reuse the assessor's _GeminiBackend because it handles .env loading,
        # the API key fallback chain, and the Gemini schema-pruning quirks.
        from PIL import Image as _Image
        from svrptw.vivrp.assessor import _GeminiBackend, _parse_json_loose

        backend = _GeminiBackend()
        judge_id = f"gemini:{backend._model_id}"
        img_a = _Image.open(str(image_a))
        img_b = _Image.open(str(image_b))
        cfg = {
            "response_mime_type": "application/json",
            "response_schema": _GeminiBackend._strip_unsupported(JUDGE_SCHEMA),
            "temperature": 0.0,
        }
        out = backend._client.models.generate_content(
            model=backend._model_id,
            contents=[prompt, img_a, img_b],
            config=cfg,
        )
        parsed = _parse_json_loose(out.text)
    except Exception as e:  # noqa: BLE001
        return JudgeVerdict(
            judge_id=judge_id, judge_kind="gemini", tier="gemini",
            score=None, rationale="", confidence=None,
            latency_s=time.perf_counter() - t0,
            error=f"{type(e).__name__}: {e}",
        )
    score = float(parsed.get("score", 0.5))
    score = max(0.0, min(1.0, score))
    confidence = parsed.get("confidence")
    if confidence is not None:
        confidence = max(0.0, min(1.0, float(confidence)))
    return JudgeVerdict(
        judge_id=judge_id, judge_kind="gemini", tier="gemini",
        score=score,
        rationale=str(parsed.get("rationale", "")),
        confidence=confidence,
        latency_s=time.perf_counter() - t0,
    )


# --- Anthropic Haiku judge ---------------------------------------------

_ANTHROPIC_MODEL = "claude-3-5-haiku-latest"


def _encode_png_b64(image_path: Path) -> str:
    return base64.b64encode(image_path.read_bytes()).decode("ascii")


def _call_anthropic_haiku(
    prompt: str,
    image_a: Path,
    image_b: Path,
    *,
    timeout_s: float = 60.0,
) -> JudgeVerdict:
    """Anthropic Haiku via lazy-imported SDK; degrades on missing dep/key."""
    judge_id = f"anthropic:{_ANTHROPIC_MODEL}"
    t0 = time.perf_counter()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return JudgeVerdict(
            judge_id=judge_id, judge_kind="anthropic", tier="haiku",
            score=None, rationale="", confidence=None,
            latency_s=time.perf_counter() - t0,
            error="anthropic-not-available: ANTHROPIC_API_KEY unset",
        )
    try:
        import anthropic  # noqa: PLC0415 — lazy import on purpose
    except Exception as e:  # noqa: BLE001
        return JudgeVerdict(
            judge_id=judge_id, judge_kind="anthropic", tier="haiku",
            score=None, rationale="", confidence=None,
            latency_s=time.perf_counter() - t0,
            error=f"anthropic-not-available: {type(e).__name__}: {e}",
        )


    schema_block = json.dumps(JUDGE_SCHEMA, indent=2)
    sys_prompt = (
        "You are an expert dispatcher comparing two VRP solutions side by "
        "side. Image 1 is Solution A; Image 2 is Solution B. Return strict "
        "JSON matching this schema (no surrounding prose, no markdown):\n"
        f"{schema_block}\n"
        "Score 0.0 = B is unambiguously better; 1.0 = A is unambiguously "
        "better; 0.5 = toss-up."
    )
    try:
        client = anthropic.Anthropic(timeout=timeout_s)
        msg = client.messages.create(
            model=_ANTHROPIC_MODEL,
            max_tokens=512,
            system=sys_prompt,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image", "source": {
                        "type": "base64", "media_type": "image/png",
                        "data": _encode_png_b64(image_a),
                    }},
                    {"type": "image", "source": {
                        "type": "base64", "media_type": "image/png",
                        "data": _encode_png_b64(image_b),
                    }},
                ],
            }],
        )
    except Exception as e:  # noqa: BLE001
        return JudgeVerdict(
            judge_id=judge_id, judge_kind="anthropic", tier="haiku",
            score=None, rationale="", confidence=None,
            latency_s=time.perf_counter() - t0,
            error=f"{type(e).__name__}: {e}",
        )


    # Concatenate any text blocks the SDK returns, then parse loose JSON.
    raw = ""
    try:
        for block in (msg.content or []):
            text = getattr(block, "text", None)
            if isinstance(text, str):
                raw += text
        from svrptw.vivrp.assessor import _parse_json_loose
        parsed = _parse_json_loose(raw) if raw else {}
        score = float(parsed.get("score", 0.5))
        score = max(0.0, min(1.0, score))
        confidence = parsed.get("confidence")
        if confidence is not None:
            confidence = max(0.0, min(1.0, float(confidence)))
    except Exception as e:  # noqa: BLE001
        return JudgeVerdict(
            judge_id=judge_id, judge_kind="anthropic", tier="haiku",
            score=None, rationale=raw[:300],
            confidence=None,
            latency_s=time.perf_counter() - t0,
            error=f"parse: {type(e).__name__}: {e}",
        )
    return JudgeVerdict(
        judge_id=judge_id, judge_kind="anthropic", tier="haiku",
        score=score,
        rationale=str(parsed.get("rationale", "")),
        confidence=confidence,
        latency_s=time.perf_counter() - t0,
    )


# --- Public surface -----------------------------------------------------

def judge_pair(
    image_a_path: Path,
    image_b_path: Path,
    *,
    prompt: str,
    enable_openrouter: bool = True,
    enable_gemini: bool = True,
    enable_anthropic_haiku: bool = True,
    max_workers: int = 8,
    pair_id: Optional[str] = None,
    push_to_webui: bool = True,
    openrouter_models: Optional[tuple[str, ...]] = None,
    timeout_s: float = 30.0,
) -> tuple[list[JudgeVerdict], MultiJudgeConsensus]:
    """Run all enabled judges concurrently and aggregate to a consensus.

    Each judge call is wrapped in try/except inside its own helper, so
    one failure never tanks the panel. If `push_to_webui=True` and
    `webui.client` is importable, the verdict list and consensus are
    POSTed via `push_judges` for the live UI.
    """
    image_a_path = Path(image_a_path)
    image_b_path = Path(image_b_path)

    if openrouter_models is None:
        try:
            from svrptw.logic.committee import TIER_A
            openrouter_models = tuple(TIER_A.models)
        except Exception:
            openrouter_models = ()

    # Build the call list. Each entry is (callable, args-tuple).
    calls: list = []
    if enable_openrouter:
        for m in openrouter_models:
            calls.append((_call_openrouter,
                          (m, prompt, image_a_path, image_b_path),
                          {"timeout_s": timeout_s}))
    if enable_gemini:
        calls.append((_call_gemini,
                      (prompt, image_a_path, image_b_path),
                      {"timeout_s": timeout_s}))
    if enable_anthropic_haiku:
        calls.append((_call_anthropic_haiku,
                      (prompt, image_a_path, image_b_path),
                      {"timeout_s": max(timeout_s, 60.0)}))


    verdicts: list[JudgeVerdict] = []
    if not calls:
        # All judges disabled — no panel, no consensus.
        consensus = aggregate_verdicts([])
        return verdicts, consensus

    workers = max(1, min(int(max_workers), len(calls)))
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(fn, *args, **kw): (fn.__name__, args)
                   for fn, args, kw in calls}
        try:
            for fut in cf.as_completed(futures, timeout=timeout_s * 4):
                try:
                    v = fut.result()
                except Exception as e:  # noqa: BLE001 — defensive belt-and-braces
                    name, args = futures[fut]
                    v = JudgeVerdict(
                        judge_id=f"{name}:{args[0] if args else '?'}",
                        judge_kind="unknown", tier="A",
                        score=None, rationale="", confidence=None,
                        latency_s=0.0,
                        error=f"executor: {type(e).__name__}: {e}",
                    )
                verdicts.append(v)
        except cf.TimeoutError:
            for fut in futures:
                if not fut.done():
                    fut.cancel()

    consensus = aggregate_verdicts(verdicts)

    if push_to_webui:
        try:
            from webui import client as ui
            ui.push_judges(
                pair_id or "<unknown>",
                judges=[asdict(v) for v in verdicts],
                consensus=asdict(consensus),
            )
        except Exception as e:  # noqa: BLE001
            _LOG.debug("push_judges failed: %s", e)

    return verdicts, consensus
