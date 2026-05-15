"""Multi-model OpenRouter committee teacher (SPEC-6-LOGIC-02).

Free-tier vision committee across 6 model families. Tier-weighted
median consensus + variance-based authoritative flag → distilled into
the SPEC-6-LOGIC-01 student.

The committee is the *teacher*: called offline to label preference
pairs, ~30 s/pair. The *student* (50 ms) is what the solver inner
loop calls; this module never sees the solver's hot path.
"""
from __future__ import annotations

import concurrent.futures as cf
import hashlib
import logging
import os
import statistics
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from svrptw.logic.openrouter_client import (
    ModelDeprecated,
    RateLimited,
    VLMResponse,
    call_model,
)

if TYPE_CHECKING:
    from svrptw.io.instance import Instance
    from svrptw.solvers.common.solution import Solution


_LOG = logging.getLogger("svrptw.logic.committee")


# --- Tier definitions ---------------------------------------------------

@dataclass(frozen=True)
class _TierConfig:
    name: str
    models: tuple[str, ...]
    vote_weight: float
    sanitize: bool  # True → strip identifying info before sending


# Catalog verified live against OpenRouter /api/v1/models on 2026-05-13.
# The user's research-style spec list named ~12 models, but only these 4
# are currently free-vision-capable on OpenRouter. Re-query the live
# catalog when adding tiers — the lifecycle is short.
TIER_S = _TierConfig(
    name="S",
    models=(),   # no cloaked-alpha vision models live on free tier right now
    vote_weight=0.5,
    sanitize=True,
)
TIER_A = _TierConfig(
    name="A",
    models=(
        "google/gemma-4-31b-it:free",
        "google/gemma-4-26b-a4b-it:free",
        "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
        "nvidia/nemotron-nano-12b-v2-vl:free",
    ),
    vote_weight=1.0,
    sanitize=False,
)
TIER_B = _TierConfig(
    name="B",
    models=(),   # currently empty; populate when more free-vision slugs appear
    vote_weight=0.6,
    sanitize=False,
)
TIER_C = _TierConfig(
    name="C",
    models=(),   # currently empty; tiebreak tier
    vote_weight=0.4,
    sanitize=False,
)


# --- Result dataclasses -------------------------------------------------

@dataclass
class ModelVote:
    model: str
    tier: str
    score: float
    rationale: str
    confidence: float
    latency_s: float


@dataclass
class CommitteeLabel:
    """The teacher's verdict on one (instance, solution) pair."""

    score: Optional[float]            # tier-weighted median; None on total failure
    score_std: float
    confidence: float                  # 1 - clamp(score_std, 0, 1)
    authoritative: bool
    n_responders: int
    rationale: str
    per_model: dict[str, ModelVote] = field(default_factory=dict)
    rationale_review: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "score": self.score,
            "score_std": self.score_std,
            "confidence": self.confidence,
            "authoritative": self.authoritative,
            "n_responders": self.n_responders,
            "rationale": self.rationale,
            "rationale_review": self.rationale_review,
            "per_model": {
                m: {"score": v.score, "tier": v.tier, "latency_s": v.latency_s}
                for m, v in self.per_model.items()
            },
        }


@dataclass
class PairLabel:
    """Joint verdict from a single combined judging call (SPEC-6-LOGIC-02
    split-channel design): VLM sees both rendered plans + text channel."""

    score_a: Optional[float]
    score_b: Optional[float]
    score_a_std: float
    score_b_std: float
    rationale: str
    authoritative: bool
    n_responders: int
    per_model: dict[str, dict] = field(default_factory=dict)


# JSON-schema for the combined-pair judging call.
PAIR_SCHEMA = {
    "type": "object",
    "properties": {
        "score_a": {"type": "number", "minimum": 0.0, "maximum": 1.0,
                    "description": "P(dispatcher ships solution A), 0 to 1"},
        "score_b": {"type": "number", "minimum": 0.0, "maximum": 1.0,
                    "description": "P(dispatcher ships solution B), 0 to 1"},
        "rationale": {"type": "string",
                      "description": "Two to four sentences justifying both scores"},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    },
    "required": ["score_a", "score_b", "rationale", "confidence"],
}


# --- Sanitization for Tier S -------------------------------------------

def _sanitize_prompt(prompt: str, instance_id: str) -> str:
    """Replace identifying tokens before sending to logging tiers."""
    hashed = hashlib.sha1(instance_id.encode("utf-8")).hexdigest()[:12]
    return prompt.replace(instance_id, f"inst-{hashed}")


# --- Tier S liveness tracking (process-global) -------------------------

_dead_models_lock = threading.Lock()
_dead_models: set[str] = set()


def _is_dead(model: str) -> bool:
    with _dead_models_lock:
        return model in _dead_models


def _mark_dead(model: str, reason: str) -> None:
    with _dead_models_lock:
        if model in _dead_models:
            return
        _dead_models.add(model)
    _LOG.warning("dropping %s from committee: %s", model, reason)


# --- Aggregation --------------------------------------------------------

def _weighted_median(values: list[float], weights: list[float]) -> float:
    paired = sorted(zip(values, weights), key=lambda x: x[0])
    total = sum(w for _, w in paired)
    cum = 0.0
    for v, w in paired:
        cum += w
        if cum >= total / 2:
            return v
    return paired[-1][0]


def _aggregate(votes: list[tuple[ModelVote, float]]) -> tuple[float, float, str]:
    """Returns (weighted_median_score, std_across_responders, best_rationale)."""
    scores = [v.score for v, _ in votes]
    weights = [w for _, w in votes]
    median = _weighted_median(scores, weights)
    std = statistics.pstdev(scores) if len(scores) >= 2 else 0.0
    # Pick rationale: highest weight, tiebreak on closeness to median, then brevity.
    best = max(
        votes,
        key=lambda vw: (vw[1], -abs(vw[0].score - median), -len(vw[0].rationale)),
    )
    return median, std, best[0].rationale


# --- The committee ------------------------------------------------------

class OpenRouterCommittee:
    """Tier-orchestrated free-tier OpenRouter committee."""

    def __init__(
        self,
        *,
        tiers_primary: tuple[_TierConfig, ...] = (TIER_A, TIER_B),
        tier_stealth: Optional[_TierConfig] = TIER_S,
        tier_tiebreaker: _TierConfig = TIER_C,
        max_workers: int = 8,
        per_model_timeout_s: float = 30.0,
        std_escalate_threshold: float = 0.20,
        # 3-of-4-current-free-tier vs 6-of-12-original-spec. OpenRouter's
        # free vision catalog has only 4 viable slugs as of 2026-05-13;
        # the original 6 threshold was permanently unreachable. Further
        # relaxed to 2 responders + std≤0.30 after free-tier rate-limiting
        # capped achievable nresp at 1-2 per call. See
        # bench/figures/threeaxis_smoke_capacity_constrained.md.
        authoritative_min_responders: int = 2,
        authoritative_max_std: float = 0.30,
        include_gemini: bool = True,   # 5th voter via direct google-genai SDK
        gemini_weight: float = 1.0,    # Tier-A equivalent weight
    ):
        self._tiers_primary = tiers_primary
        self._tier_stealth = tier_stealth
        self._tier_tiebreaker = tier_tiebreaker
        self._max_workers = max_workers
        self._timeout = per_model_timeout_s
        self._std_escalate = std_escalate_threshold
        self._auth_min_n = authoritative_min_responders
        self._auth_max_std = authoritative_max_std
        self._include_gemini = include_gemini
        self._gemini_weight = gemini_weight

    def _call_one(
        self,
        model: str,
        tier: _TierConfig,
        prompt: str,
        image_path: Optional[Path],
        instance_id: str,
    ) -> Optional[ModelVote]:
        if _is_dead(model):
            return None
        send_prompt = _sanitize_prompt(prompt, instance_id) if tier.sanitize else prompt
        try:
            resp: VLMResponse = call_model(
                model, send_prompt, image_path, timeout_s=self._timeout
            )
        except ModelDeprecated as e:
            _mark_dead(model, str(e))
            return None
        except RateLimited as e:
            _LOG.info("%s rate-limited: %s", model, e)
            return None
        except Exception as e:  # pragma: no cover — defensive
            _LOG.info("%s errored: %s", model, e)
            return None
        return ModelVote(
            model=model,
            tier=tier.name,
            score=resp.score,
            rationale=resp.rationale,
            confidence=resp.confidence,
            latency_s=resp.latency_s,
        )

    def _call_one_pair(
        self,
        model: str,
        tier: _TierConfig,
        prompt: str,
        image_a: Path,
        image_b: Path,
        instance_id: str,
    ) -> Optional[tuple[float, float, str]]:
        """Combined dual-image call. Returns (score_a, score_b, rationale) or None."""
        if _is_dead(model):
            return None
        send_prompt = _sanitize_prompt(prompt, instance_id) if tier.sanitize else prompt
        try:
            resp: VLMResponse = call_model(
                model, send_prompt, image_a,
                extra_image_paths=[image_b],
                schema=__import__("svrptw.logic.committee", fromlist=["PAIR_SCHEMA"]).PAIR_SCHEMA,
                timeout_s=self._timeout,
            )
        except ModelDeprecated as e:
            _mark_dead(model, str(e))
            return None
        except (RateLimited, Exception) as e:
            _LOG.info("%s pair-call failed: %s", model, e)
            return None
        parsed = resp.raw_json.get("parsed", {})
        if "score_a" not in parsed or "score_b" not in parsed:
            return None
        sa = max(0.0, min(1.0, float(parsed["score_a"])))
        sb = max(0.0, min(1.0, float(parsed["score_b"])))
        return (sa, sb, resp.rationale)

    def _call_gemini_pair(
        self, prompt: str, image_a: Path, image_b: Path,
    ) -> Optional[tuple[float, float, str]]:
        """Gemini-direct dual-image pair-judging call."""
        if not self._include_gemini:
            return None
        try:
            from svrptw.vivrp.assessor import _GeminiBackend
            from PIL import Image
        except Exception:
            return None
        try:
            backend = _GeminiBackend()
            img_a = Image.open(str(image_a))
            img_b = Image.open(str(image_b))
            schema = __import__("svrptw.logic.committee",
                                fromlist=["PAIR_SCHEMA"]).PAIR_SCHEMA
            # Gemini's query() takes a single image + prompt. For dual-image
            # we use the underlying client directly.
            cfg = {
                "response_mime_type": "application/json",
                "response_schema": backend._strip_unsupported(schema),
                "temperature": 0.0,
            }
            out = backend._client.models.generate_content(
                model=backend._model_id,
                contents=[prompt, img_a, img_b],
                config=cfg,
            )
            from svrptw.vivrp.assessor import _parse_json_loose
            parsed = _parse_json_loose(out.text)
        except Exception as e:
            _LOG.info("gemini pair-call failed: %s", e)
            return None
        if not isinstance(parsed, dict) or "score_a" not in parsed:
            return None
        sa = max(0.0, min(1.0, float(parsed.get("score_a", 0.5))))
        sb = max(0.0, min(1.0, float(parsed.get("score_b", 0.5))))
        rat = str(parsed.get("rationale", ""))
        return (sa, sb, rat)

    def label_pair(
        self, instance, sol_a, sol_b, prompt: str,
        image_a: Path, image_b: Path,
    ) -> "PairLabel":
        """Joint judging of two solutions in a single API call per model.

        Halves cost vs separate-judge and gives the VLM side-by-side
        context. Each model returns (score_a, score_b, rationale) and we
        aggregate the per-model scores into the consensus PairLabel.
        """
        import concurrent.futures as cf
        votes: list[tuple[float, float, str, str, str, float]] = []
        # tuples: (score_a, score_b, rationale, model, tier_name, weight)
        primary_tiers = list(self._tiers_primary)
        if self._tier_stealth is not None and self._tier_stealth.models:
            primary_tiers.append(self._tier_stealth)

        for tier in primary_tiers:
            models = [m for m in tier.models if not _is_dead(m)]
            if not models:
                continue
            with cf.ThreadPoolExecutor(
                max_workers=min(self._max_workers, len(models))
            ) as ex:
                futs = {
                    ex.submit(self._call_one_pair, m, tier, prompt,
                              image_a, image_b, instance.instance_id): m
                    for m in models
                }
                try:
                    for fut in cf.as_completed(futs, timeout=self._timeout * 2):
                        try:
                            r = fut.result()
                        except Exception:
                            continue
                        if r is None:
                            continue
                        sa, sb, rat = r
                        votes.append((sa, sb, rat, futs[fut], tier.name, tier.vote_weight))
                except cf.TimeoutError:
                    for fut in futs:
                        if not fut.done():
                            fut.cancel()

        # Gemini direct.
        g = self._call_gemini_pair(prompt, image_a, image_b)
        if g is not None:
            sa, sb, rat = g
            votes.append((sa, sb, rat, "gemini-direct", "A", self._gemini_weight))

        if not votes:
            return PairLabel(
                score_a=None, score_b=None, score_a_std=0.0, score_b_std=0.0,
                rationale="no responders", authoritative=False, n_responders=0,
            )

        import statistics as _stat
        sas = [v[0] for v in votes]
        sbs = [v[1] for v in votes]
        weights = [v[5] for v in votes]
        med_a = _weighted_median(sas, weights)
        med_b = _weighted_median(sbs, weights)
        std_a = _stat.pstdev(sas) if len(sas) >= 2 else 0.0
        std_b = _stat.pstdev(sbs) if len(sbs) >= 2 else 0.0
        # Pick the rationale closest to the joint median.
        best = min(votes, key=lambda v: abs(v[0] - med_a) + abs(v[1] - med_b))
        n = len(votes)
        auth = (n >= self._auth_min_n
                and max(std_a, std_b) <= self._auth_max_std)
        return PairLabel(
            score_a=med_a, score_b=med_b,
            score_a_std=std_a, score_b_std=std_b,
            rationale=best[2], authoritative=auth, n_responders=n,
            per_model={
                v[3]: {"score_a": v[0], "score_b": v[1], "tier": v[4]}
                for v in votes
            },
        )

    def _call_gemini(
        self,
        prompt: str,
        image_path: Optional[Path],
    ) -> Optional[ModelVote]:
        """5th voter via google-genai SDK (not on OpenRouter free tier).

        Returns None on any error so the committee degrades gracefully.
        """
        if not self._include_gemini:
            return None
        try:
            from svrptw.vivrp.assessor import _GeminiBackend
            from PIL import Image
        except Exception as e:
            _LOG.info("gemini-direct unavailable: %s", e)
            return None
        if image_path is None:
            return None
        try:
            backend = _GeminiBackend()
            img = Image.open(str(image_path))
            schema = {
                "type": "object",
                "properties": {
                    "score": {"type": "number"},
                    "rationale": {"type": "string"},
                    "confidence": {"type": "number"},
                },
                "required": ["score", "rationale", "confidence"],
            }
            t0 = time.time()
            parsed = backend.query(img, prompt, schema=schema)
            dt = time.time() - t0
        except Exception as e:  # pragma: no cover — defensive
            _LOG.info("gemini-direct errored: %s", e)
            return None
        try:
            score = max(0.0, min(1.0, float(parsed.get("score", 0.5))))
            rationale = str(parsed.get("rationale", ""))
            confidence = max(0.0, min(1.0, float(parsed.get("confidence", 0.5))))
        except Exception as e:  # pragma: no cover
            _LOG.info("gemini parse failed: %s", e)
            return None
        return ModelVote(
            model=f"gemini-direct:{backend._model_id}",
            tier="A",
            score=score,
            rationale=rationale,
            confidence=confidence,
            latency_s=dt,
        )

    def _call_tier(
        self,
        tier: _TierConfig,
        prompt: str,
        image_path: Optional[Path],
        instance_id: str,
    ) -> list[tuple[ModelVote, float]]:
        results: list[tuple[ModelVote, float]] = []
        models = [m for m in tier.models if not _is_dead(m)]
        if not models:
            return results
        with cf.ThreadPoolExecutor(max_workers=min(self._max_workers, len(models))) as ex:
            futures = {
                ex.submit(self._call_one, m, tier, prompt, image_path, instance_id): m
                for m in models
            }
            # Harvest whatever's ready before the outer deadline; don't crash
            # if a slow voter never finishes — its per-model timeout already
            # bounds it, and we degrade gracefully to "fewer responders".
            try:
                for fut in cf.as_completed(futures, timeout=self._timeout * 2):
                    try:
                        vote = fut.result()
                    except Exception:  # pragma: no cover
                        continue
                    if vote is not None:
                        results.append((vote, tier.vote_weight))
            except cf.TimeoutError:
                # Cancel any still-pending futures so the executor shuts down.
                for fut in futures:
                    if not fut.done():
                        fut.cancel()
        return results

    def label(
        self,
        instance: "Instance",
        solution: "Solution",
        prompt: str,
        image_path: Optional[Path],
    ) -> CommitteeLabel:
        """Run the committee on one (instance, solution) pair.

        `prompt` is the rubric-instantiated prompt text;
        `image_path` is the rendered solution PNG (None → text-only).
        """
        # Primary tiers (A + B), and optional stealth tier, run in parallel.
        all_votes: list[tuple[ModelVote, float]] = []

        primary_tiers = list(self._tiers_primary)
        if self._tier_stealth is not None and self._tier_stealth.models:
            primary_tiers.append(self._tier_stealth)

        for tier in primary_tiers:
            all_votes.extend(self._call_tier(tier, prompt, image_path, instance.instance_id))

        # 5th voter: direct Gemini Flash Lite call (not on OpenRouter free tier).
        gemini_vote = self._call_gemini(prompt, image_path)
        if gemini_vote is not None:
            all_votes.append((gemini_vote, self._gemini_weight))

        # Escalate to tiebreaker tier on disagreement / insufficient responders.
        if all_votes:
            scores = [v.score for v, _ in all_votes]
            std = statistics.pstdev(scores) if len(scores) >= 2 else 0.0
            if std > self._std_escalate or len(scores) < self._auth_min_n:
                all_votes.extend(
                    self._call_tier(
                        self._tier_tiebreaker, prompt, image_path, instance.instance_id
                    )
                )

        if not all_votes:
            return CommitteeLabel(
                score=None,
                score_std=0.0,
                confidence=0.0,
                authoritative=False,
                n_responders=0,
                rationale="committee: no responders",
            )

        median, std, rationale = _aggregate(all_votes)
        confidence = max(0.0, 1.0 - min(1.0, std))
        n = len(all_votes)
        authoritative = (n >= self._auth_min_n) and (std <= self._auth_max_std)

        return CommitteeLabel(
            score=median,
            score_std=std,
            confidence=confidence,
            authoritative=authoritative,
            n_responders=n,
            rationale=rationale,
            per_model={v.model: v for v, _ in all_votes},
        )
