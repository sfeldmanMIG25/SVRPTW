"""OpenRouter HTTP client (SPEC-6-LOGIC-02).

Free-tier focused: every call sets `max_price=0` as a belt-and-braces
guard; the `:free` suffix in model strings is the primary safety. We
retry once on 429, drop on persistent 404 (cloaked-model deprecation).
"""
from __future__ import annotations

import base64
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import urllib.error
import urllib.request

_LOG = logging.getLogger("svrptw.logic.openrouter")

_DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
_TIMEOUT_SECS = 30.0
_RETRY_BACKOFF_SECS = 2.0


class ModelDeprecated(RuntimeError):
    """Raised when OpenRouter returns 404 for a model — drop from pool."""


class RateLimited(RuntimeError):
    """Raised when 429 persists past one retry."""


@dataclass
class VLMResponse:
    """One model's structured response on a (instance, solution) pair."""

    model: str
    score: float          # in [0, 1]
    rationale: str
    confidence: float     # in [0, 1] — model's self-reported certainty
    raw_json: dict[str, Any] = field(repr=False)
    latency_s: float = 0.0


def _read_api_key() -> str:
    key = os.environ.get("SVRPTW_OPENROUTER_API_KEY", "")
    if key:
        return key
    # Lazy .env load — keep the dependency surface tiny.
    env_path = Path(".env")
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("SVRPTW_OPENROUTER_API_KEY="):
                return line.split("=", 1)[1].strip()
    raise RuntimeError("SVRPTW_OPENROUTER_API_KEY not set in env or .env")


def _read_base_url() -> str:
    return os.environ.get("SVRPTW_OPENROUTER_BASE_URL", _DEFAULT_BASE_URL)


def _encode_image(image_path: Path) -> str:
    data = image_path.read_bytes()
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:image/png;base64,{b64}"


# The judgment schema is shared across every model — defined once.
JUDGMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {"type": "number", "minimum": 0.0, "maximum": 1.0,
                  "description": "P(dispatcher ships this), 0 to 1"},
        "rationale": {"type": "string",
                      "description": "Two to four sentences justifying the score"},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0,
                       "description": "Model's certainty in its own score"},
    },
    "required": ["score", "rationale", "confidence"],
}


def call_model(
    model: str,
    prompt_text: str,
    image_path: Optional[Path] = None,
    *,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    timeout_s: float = _TIMEOUT_SECS,
    extra_image_paths: Optional[list[Path]] = None,
    schema: Optional[dict] = None,
) -> VLMResponse:
    """Single synchronous call to one OpenRouter model.

    Pass multiple images via `extra_image_paths` for dual/multi-solution
    judging (SPEC-6-LOGIC-02 split-channel design: VLM sees both
    rendered plans + text channel with exact numerics).

    Raises ModelDeprecated on 404 (caller drops model from pool).
    Raises RateLimited on persistent 429.
    Raises ValueError on malformed structured output.
    """
    if api_key is None:
        api_key = _read_api_key()
    if base_url is None:
        base_url = _read_base_url()

    content: list[dict[str, Any]] = [{"type": "text", "text": prompt_text}]
    if image_path is not None:
        content.append({
            "type": "image_url",
            "image_url": {"url": _encode_image(image_path)},
        })
    if extra_image_paths:
        for p in extra_image_paths:
            content.append({
                "type": "image_url",
                "image_url": {"url": _encode_image(p)},
            })

    body = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        # The `:free` suffix in the model id is the actual paywall guard.
        # An earlier `max_price=0` here returned 400 from OpenRouter because
        # the field expects an object (per-token caps), not a scalar.
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "dispatcher_judgment",
                "strict": True,
                "schema": schema if schema is not None else JUDGMENT_SCHEMA,
            },
        },
    }
    payload = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=payload,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/sfeldmanMIG25/SVRPTW",
            "X-Title": "SVRPTW Logic Teacher",
        },
    )

    t0 = time.perf_counter()
    for attempt in (0, 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                raw = json.loads(resp.read().decode("utf-8"))
            break
        except urllib.error.HTTPError as e:
            if e.code == 404:
                raise ModelDeprecated(f"{model} returned 404 — cloaked model retired") from e
            if e.code == 429 and attempt == 0:
                time.sleep(_RETRY_BACKOFF_SECS)
                continue
            if e.code == 429:
                raise RateLimited(f"{model}: 429 after retry") from e
            raise
    else:  # pragma: no cover — for-else covers exhausted retries
        raise RateLimited(f"{model}: retries exhausted")

    latency = time.perf_counter() - t0
    choice = raw.get("choices", [{}])[0]
    msg = choice.get("message", {})
    content_str = msg.get("content")
    if isinstance(content_str, list):
        # Some providers split structured output into parts; concat text.
        content_str = "".join(p.get("text", "") for p in content_str if isinstance(p, dict))
    if not content_str:
        raise ValueError(f"{model}: empty content in response")
    # SPEC-WEBUI-11 -- robust JSON extraction for free-tier models that
    # don't strictly honor JSON-schema constraints (e.g. gemma-4-26b
    # consistently wraps responses in ```json ... ``` markdown fences).
    # Strategy: try plain json.loads first; on failure, strip markdown
    # code-fences; on second failure, regex-extract the first {...} block.
    def _try_parse(s: str):
        return json.loads(s)
    parsed = None
    try:
        parsed = _try_parse(content_str)
    except json.JSONDecodeError:
        import re as _re
        # strip ```json...``` or ```...``` fences
        m = _re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", content_str)
        if m:
            try:
                parsed = _try_parse(m.group(1))
            except json.JSONDecodeError:
                parsed = None
        if parsed is None:
            # last resort: greedy regex for the first balanced {...}
            m2 = _re.search(r"(\{[\s\S]*\})", content_str)
            if m2:
                try:
                    parsed = _try_parse(m2.group(1))
                except json.JSONDecodeError:
                    parsed = None
        if parsed is None:
            raise ValueError(
                f"{model}: non-JSON content: {content_str[:200]}"
            )

    # Pair-judging schema (score_a, score_b) returns the raw dict in
    # `raw_json["parsed"]`; the legacy single-score path keeps the score
    # field populated for back-compat callers.
    rationale = str(parsed.get("rationale", ""))
    confidence = float(parsed.get("confidence", 0.5))
    confidence = max(0.0, min(1.0, confidence))
    if "score_a" in parsed and "score_b" in parsed:
        score = float(parsed["score_a"])    # convention: A is primary
    else:
        score = float(parsed.get("score", 0.5))
    score = max(0.0, min(1.0, score))
    raw["parsed"] = parsed   # store full parsed JSON for callers that want both scores

    return VLMResponse(
        model=model,
        score=score,
        rationale=rationale,
        confidence=confidence,
        raw_json=raw,
        latency_s=latency,
    )
