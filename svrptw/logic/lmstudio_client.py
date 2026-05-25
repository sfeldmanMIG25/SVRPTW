"""LM Studio (OpenAI-compatible local) vision client.

The user runs LM Studio locally with a vision model loaded. Local
inference is faster + free + has no rate limits compared to the
OpenRouter free tier, so it makes consistency studies tractable.

Defaults:
  - base URL: http://127.0.0.1:1234/v1
  - timeout: generous (local first calls warm the model)
"""
from __future__ import annotations

import base64
import json
import logging
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

_LOG = logging.getLogger("svrptw.logic.lmstudio")
_DEFAULT_BASE_URL = "http://127.0.0.1:1234/v1"
_TIMEOUT_S = 180.0


def _base_url() -> str:
    return os.environ.get("SVRPTW_LMSTUDIO_BASE_URL", _DEFAULT_BASE_URL).rstrip("/")


@dataclass
class LMStudioResponse:
    model: str
    score: Optional[float]
    rationale: str
    confidence: Optional[float]
    raw_text: str
    latency_s: float
    error: Optional[str] = None


def list_models() -> list[str]:
    """Return list of LM Studio model IDs currently loaded (or [] on failure)."""
    try:
        url = f"{_base_url()}/models"
        with urllib.request.urlopen(url, timeout=4) as r:
            data = json.load(r)
        return [m["id"] for m in data.get("data", []) if "id" in m]
    except Exception as e:
        _LOG.debug("list_models failed: %s", e)
        return []


def _parse_score_json(text: str) -> tuple[Optional[float], str, Optional[float]]:
    """Robust score extraction.

    Tries (in order):
      1. strict json.loads on the whole text
      2. extract content of ```json ... ``` fence
      3. greedy first {...}
      4. regex find the first 'score' number
    Returns (score, rationale, confidence).
    """
    def _from_obj(obj: dict) -> tuple[Optional[float], str, Optional[float]]:
        sc = obj.get("score")
        if isinstance(sc, str):
            try: sc = float(sc)
            except ValueError: sc = None
        return (
            float(sc) if isinstance(sc, (int, float)) else None,
            str(obj.get("rationale", "")),
            (float(obj["confidence"]) if isinstance(obj.get("confidence"), (int, float, str))
             and str(obj.get("confidence")).replace(".", "").replace("-", "").isdigit() else None),
        )

    try:
        return _from_obj(json.loads(text))
    except Exception:
        pass
    m = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text)
    if m:
        try:
            return _from_obj(json.loads(m.group(1)))
        except Exception:
            pass
    m = re.search(r"(\{[\s\S]*\})", text)
    if m:
        try:
            return _from_obj(json.loads(m.group(1)))
        except Exception:
            pass
    # last resort: find a number after "score"
    m = re.search(r"score['\"]?\s*[:=]\s*([0-9]*\.?[0-9]+)", text, re.IGNORECASE)
    if m:
        try:
            return (float(m.group(1)), text[:240], None)
        except ValueError:
            return (None, text[:240], None)
    return (None, text[:240], None)


def call_model(
    model: str,
    prompt_text: str,
    image_path: Optional[Path] = None,
    *,
    extra_image_paths: Optional[list[Path]] = None,
    base_url: Optional[str] = None,
    timeout_s: float = _TIMEOUT_S,
    temperature: float = 0.0,
    max_tokens: int = 600,
) -> LMStudioResponse:
    """One synchronous call to LM Studio with optional image(s).

    Sends prompt + base64 PNG(s) via OpenAI chat-completions schema.
    Returns LMStudioResponse with score / rationale / confidence parsed
    from the model's text output by `_parse_score_json`.
    """
    base = (base_url or _base_url()).rstrip("/")
    url = f"{base}/chat/completions"
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt_text}]
    img_paths = []
    if image_path is not None:
        img_paths.append(Path(image_path))
    if extra_image_paths:
        img_paths.extend(Path(p) for p in extra_image_paths)
    for ip in img_paths:
        b64 = base64.b64encode(ip.read_bytes()).decode("ascii")
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}"},
        })

    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }).encode("utf-8")
    req = urllib.request.Request(
        url, data=body, method="POST",
        headers={"Content-Type": "application/json"},
    )

    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            data = json.load(resp)
    except urllib.error.HTTPError as e:
        return LMStudioResponse(model=model, score=None, rationale="",
                                 confidence=None, raw_text=str(e),
                                 latency_s=time.perf_counter() - t0,
                                 error=f"HTTPError {e.code}: {e.reason}")
    except Exception as e:
        return LMStudioResponse(model=model, score=None, rationale="",
                                 confidence=None, raw_text=str(e),
                                 latency_s=time.perf_counter() - t0,
                                 error=f"{type(e).__name__}: {e}")
    elapsed = time.perf_counter() - t0
    text = ""
    try:
        text = data["choices"][0]["message"]["content"] or ""
    except Exception:
        return LMStudioResponse(model=model, score=None, rationale="",
                                 confidence=None, raw_text=json.dumps(data)[:500],
                                 latency_s=elapsed, error="no choices in response")
    sc, rat, conf = _parse_score_json(text)
    return LMStudioResponse(model=model, score=sc, rationale=rat,
                             confidence=conf, raw_text=text,
                             latency_s=elapsed,
                             error=None if sc is not None else "no score parsed")
