"""ViVRP visual quality assessor.  SPEC-5-VIVRP-01.

Backends, in preferred order:
  - lmstudio: hit a local LM Studio OpenAI-compatible endpoint (default
    http://localhost:1234) running a vision model like qwen/qwen3-vl-4b.
    Best speed + quality trade-off when LM Studio is up.
  - local: Qwen2-VL-2B-Instruct via `transformers` (in-process).
  - gemini: Gemini 2.0 Flash via `google-genai` (requires
    SVRPTW_GEMINI_API_KEY).
  - stub: returns neutral 5/5/5/5 — for tests / no-VLM environments.
"""
from __future__ import annotations

import base64
import io
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal, Protocol

from PIL import Image

_RUBRIC = (Path(__file__).parent / "rubric.md").read_text(encoding="utf-8")


@dataclass
class ZoomFinding:
    bbox: tuple[float, float, float, float]
    finding: str
    severity: Literal["info", "minor", "major"]


@dataclass
class ViVRPReport:
    overall_score: int
    clustering_score: int
    geometry_score: int
    interpretability_score: int
    notes: str
    zoom_findings: list[ZoomFinding] = field(default_factory=list)
    rubric_version: str = "v1"
    model: str = "unknown"
    latency_seconds: float = 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["zoom_findings"] = [asdict(z) for z in self.zoom_findings]
        return d


class _Backend(Protocol):
    name: str
    def query(self, image: Image.Image, prompt: str) -> dict: ...


# ---------- Backend: LM Studio (OpenAI-compatible HTTP) ----------


def _image_to_data_url(image: Image.Image) -> str:
    buf = io.BytesIO()
    # Keep PNG so the model sees clean lines.  Cap size to avoid token blow-up.
    img = image.copy()
    img.thumbnail((1024, 1024))
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


class _LMStudioBackend:
    name = "lmstudio"

    def __init__(self, base_url: str | None = None, model: str | None = None):
        import requests  # type: ignore
        self._requests = requests
        self.base_url = (base_url or os.environ.get("SVRPTW_LMSTUDIO_URL")
                         or "http://localhost:1234").rstrip("/")
        # Auto-pick the first vision-capable model if none specified.
        try:
            r = requests.get(f"{self.base_url}/v1/models", timeout=3)
            r.raise_for_status()
            ids = [m["id"] for m in r.json().get("data", [])]
        except Exception as e:
            raise RuntimeError(f"LM Studio not reachable at {self.base_url}: {e}") from e
        if model:
            self.model = model
        else:
            vision_ids = [m for m in ids if "vl" in m.lower() or "vision" in m.lower()]
            if not vision_ids:
                raise RuntimeError(f"No vision model loaded in LM Studio.  Found: {ids}")
            self.model = vision_ids[0]
        self.name = f"lmstudio:{self.model}"

    # JSON-schema constrained decoding eliminates score clumping and frees
    # the regex fallback for legacy-format responses only.  Per the
    # Track-3 May-2026 brief.
    _SCHEMA = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "overall_score", "clustering_score",
            "geometry_score", "interpretability_score",
            "notes", "worst_region_bbox",
        ],
        "properties": {
            "overall_score":          {"type": "integer", "minimum": 1, "maximum": 10},
            "clustering_score":       {"type": "integer", "minimum": 1, "maximum": 10},
            "geometry_score":         {"type": "integer", "minimum": 1, "maximum": 10},
            "interpretability_score": {"type": "integer", "minimum": 1, "maximum": 10},
            "notes":                  {"type": "string",  "maxLength": 400},
            "worst_region_bbox": {
                "type": "array", "items": {"type": "number", "minimum": 0, "maximum": 100},
                "minItems": 4, "maxItems": 4,
            },
            "zoom_requests": {
                "type": "array", "maxItems": 3,
                "items": {
                    "type": "object",
                    "required": ["bbox_pct", "why"],
                    "properties": {
                        "bbox_pct": {
                            "type": "array",
                            "items": {"type": "number", "minimum": 0, "maximum": 100},
                            "minItems": 4, "maxItems": 4,
                        },
                        "why": {"type": "string", "maxLength": 200},
                    },
                },
            },
        },
    }

    _ZOOM_SCHEMA = {
        "type": "object",
        "additionalProperties": False,
        "required": ["finding", "severity"],
        "properties": {
            "finding":  {"type": "string", "maxLength": 300},
            "severity": {"type": "string", "enum": ["info", "minor", "major"]},
        },
    }

    def query(self, image: Image.Image, prompt: str, schema: dict | None = None) -> dict:
        body = {
            "model": self.model,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": _image_to_data_url(image)}},
                ],
            }],
            "temperature": 0.0,
            "max_tokens": 512,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "vivrp", "strict": True,
                    "schema": schema or self._SCHEMA,
                },
            },
        }
        r = self._requests.post(f"{self.base_url}/v1/chat/completions",
                                json=body, timeout=180)
        r.raise_for_status()
        text = r.json()["choices"][0]["message"]["content"]
        return _parse_json_loose(text)


# ---------- Backend: Qwen2-VL local (in-process via transformers) ----------

class _Qwen2VLBackend:
    name = "qwen2-vl-2b"

    def __init__(self):
        import torch
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration  # type: ignore
        self._torch = torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_id = "Qwen/Qwen2-VL-2B-Instruct"
        self._proc = AutoProcessor.from_pretrained(model_id)
        self._model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.float16, device_map=device
        )

    def query(self, image: Image.Image, prompt: str) -> dict:
        msgs = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": prompt},
        ]}]
        text = self._proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = self._proc(text=[text], images=[image], padding=True, return_tensors="pt")
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
        out = self._model.generate(**inputs, max_new_tokens=512, do_sample=False)
        trimmed = out[:, inputs["input_ids"].shape[-1]:]
        decoded = self._proc.batch_decode(trimmed, skip_special_tokens=True)[0]
        return _parse_json_loose(decoded)


# ---------- Backend: Gemini fallback ----------

class _GeminiBackend:
    name = "gemini"

    def __init__(self, model_id: str | None = None):
        # Load .env if present (local secrets).
        env_path = Path(__file__).resolve().parents[2] / ".env"
        if env_path.exists():
            for line in env_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())
        from google import genai  # type: ignore
        key = os.environ.get("SVRPTW_GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("SVRPTW_GEMINI_API_KEY not set")
        self._client = genai.Client(api_key=key)
        self._model_id = model_id or os.environ.get("SVRPTW_GEMINI_MODEL", "gemini-flash-latest")
        self.name = f"gemini:{self._model_id}"

    @staticmethod
    def _strip_unsupported(schema: dict) -> dict:
        """Gemini's response_schema is JSON-Schema subset — drops
        `additionalProperties`, `$ref`, etc.  Recursively prune."""
        if not isinstance(schema, dict):
            return schema
        out = {k: v for k, v in schema.items()
               if k not in ("additionalProperties", "$schema", "$id", "title")}
        if "properties" in out:
            out["properties"] = {k: _GeminiBackend._strip_unsupported(v)
                                 for k, v in out["properties"].items()}
        if "items" in out:
            out["items"] = _GeminiBackend._strip_unsupported(out["items"])
        return out

    # Reuse the LM Studio schemas, pruned to Gemini's subset.
    _SCHEMA = _LMStudioBackend._SCHEMA      # raw; pruned at query time
    _ZOOM_SCHEMA = _LMStudioBackend._ZOOM_SCHEMA

    def query(self, image: Image.Image, prompt: str,
              schema: dict | None = None) -> dict:
        cfg = {
            "response_mime_type": "application/json",
            "response_schema": _GeminiBackend._strip_unsupported(schema or self._SCHEMA),
            "temperature": 0.0,
        }
        try:
            out = self._client.models.generate_content(
                model=self._model_id, contents=[prompt, image], config=cfg,
            )
        except TypeError:
            out = self._client.models.generate_content(
                model=self._model_id, contents=[prompt, image],
            )
        return _parse_json_loose(out.text)


# ---------- Backend: stub (no-VLM environments) ----------

class _StubBackend:
    name = "stub"
    def query(self, image: Image.Image, prompt: str) -> dict:
        return {
            "overall_score": 5, "clustering_score": 5,
            "geometry_score": 5, "interpretability_score": 5,
            "notes": "Stub backend — no VLM available; install transformers + qwen2-vl-2b or set SVRPTW_GEMINI_API_KEY.",
            "zoom_requests": [],
        }


def _parse_json_loose(text: str) -> dict:
    """Extract the first {...} object from a model response.  Falls back to
    regex extraction of the four score fields if JSON parsing fails (small
    VLMs frequently emit unescaped quotes inside `notes`)."""
    import re
    text = text.strip()
    if "```" in text:
        start = text.find("```")
        text = text[start + 3:]
        if text.startswith("json"):
            text = text[4:]
        end = text.rfind("```")
        if end != -1:
            text = text[:end]
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1:
        blob = text[start: end + 1]
        try:
            return json.loads(blob)
        except json.JSONDecodeError:
            repaired = blob.replace(",}", "}").replace(",]", "]")
            try:
                return json.loads(repaired)
            except json.JSONDecodeError:
                pass

    # Regex fallback: pull just the four integer scores.
    out: dict = {"zoom_requests": []}
    for key in ("overall_score", "clustering_score", "geometry_score", "interpretability_score"):
        m = re.search(rf'"{key}"\s*:\s*(\d+)', text)
        if m:
            out[key] = int(m.group(1))
    m = re.search(r'"notes"\s*:\s*"([^"]{0,300})', text)
    if m:
        out["notes"] = m.group(1)
    if any(k in out for k in ("overall_score", "clustering_score", "geometry_score", "interpretability_score")):
        return out
    raise ValueError(f"Could not parse any score fields from response: {text[:200]}")


# ---------- Public API ----------

_BACKEND: _Backend | None = None


def get_assessor(prefer: Literal["auto", "lmstudio", "local", "gemini", "stub"] = "auto") -> _Backend:
    global _BACKEND
    if _BACKEND is not None and prefer == "auto":
        return _BACKEND
    if prefer == "stub":
        _BACKEND = _StubBackend()
        return _BACKEND
    if prefer == "lmstudio":
        _BACKEND = _LMStudioBackend()
        return _BACKEND
    if prefer == "gemini":
        _BACKEND = _GeminiBackend()
        return _BACKEND
    if prefer == "local":
        _BACKEND = _Qwen2VLBackend()
        return _BACKEND
    # auto: prefer LM Studio (best quality on this machine), then transformers
    # in-process, then Gemini, then stub.
    for ctor in (_LMStudioBackend, _Qwen2VLBackend, _GeminiBackend):
        try:
            _BACKEND = ctor()
            return _BACKEND
        except Exception:
            continue
    _BACKEND = _StubBackend()
    return _BACKEND


def assess(image_path: str | Path, summary: dict | None = None,
           prefer: Literal["auto", "lmstudio", "local", "gemini", "stub"] = "auto",
           do_zoom: bool = True) -> ViVRPReport:
    backend = get_assessor(prefer)
    img = Image.open(image_path).convert("RGB")
    prompt_parts = [_RUBRIC]
    if summary:
        prompt_parts.append(
            "\n## Numeric context (do NOT use for scoring, geometry only):\n"
            + json.dumps(summary, indent=2)
        )
    prompt = "\n".join(prompt_parts)

    t0 = time.perf_counter()
    raw = backend.query(img, prompt)

    zoom_findings: list[ZoomFinding] = []
    # Track-3 May-2026 multi-pass policy: zoom on `worst_region_bbox` when the
    # overall score is below 7 OR clustering_score is below 5.  Also process
    # any explicit zoom_requests from the model (legacy path).
    W, H = img.size

    def _crop_and_query(bbox_pct, why_text=""):
        x0 = max(0, int(bbox_pct[0] * W / 100))
        y0 = max(0, int(bbox_pct[1] * H / 100))
        x1 = min(W, int(bbox_pct[2] * W / 100))
        y1 = min(H, int(bbox_pct[3] * H / 100))
        if x1 - x0 < 20 or y1 - y0 < 20:
            return None
        crop = img.crop((x0, y0, x1, y1)).resize(((x1 - x0) * 2, (y1 - y0) * 2))
        prompt_z = (
            "Zoomed-in crop of a VRP solution. " + (why_text or "") +
            " Describe one specific quality issue or strength in 1-2 sentences."
        )
        zoom_schema = getattr(backend, "_ZOOM_SCHEMA", None)
        if zoom_schema is not None:
            z = backend.query(crop, prompt_z, schema=zoom_schema)  # type: ignore[arg-type]
        else:
            z = backend.query(crop, prompt_z)
        return ZoomFinding(
            bbox=(float(bbox_pct[0]), float(bbox_pct[1]),
                  float(bbox_pct[2]), float(bbox_pct[3])),
            finding=str(z.get("finding", ""))[:300],
            severity=str(z.get("severity", "info")),  # type: ignore
        )

    if do_zoom:
        overall = int(raw.get("overall_score", 5))
        clustering = int(raw.get("clustering_score", 5))
        wrb = raw.get("worst_region_bbox")
        if (overall < 7 or clustering < 5) and isinstance(wrb, list) and len(wrb) == 4:
            try:
                zf = _crop_and_query(wrb, why_text="This region was flagged as the worst by the full-map pass.")
                if zf is not None:
                    zoom_findings.append(zf)
            except Exception as e:
                zoom_findings.append(ZoomFinding(bbox=(0, 0, 0, 0),
                    finding=f"worst-region zoom failed: {e!r}", severity="info"))
        if isinstance(raw.get("zoom_requests"), list):
            for zr in raw["zoom_requests"][:3]:
                try:
                    zf = _crop_and_query(zr["bbox_pct"], why_text=zr.get("why", ""))
                    if zf is not None:
                        zoom_findings.append(zf)
                except Exception as e:
                    zoom_findings.append(ZoomFinding(bbox=(0, 0, 0, 0),
                        finding=f"zoom_request failed: {e!r}", severity="info"))

    elapsed = time.perf_counter() - t0
    return ViVRPReport(
        overall_score=int(raw.get("overall_score", 5)),
        clustering_score=int(raw.get("clustering_score", 5)),
        geometry_score=int(raw.get("geometry_score", 5)),
        interpretability_score=int(raw.get("interpretability_score", 5)),
        notes=str(raw.get("notes", ""))[:600],
        zoom_findings=zoom_findings,
        model=backend.name,
        latency_seconds=elapsed,
    )


def _cli() -> int:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--summary", default=None, help="JSON file with N, K, missed, etc.")
    p.add_argument("--backend", choices=["auto", "lmstudio", "local", "gemini", "stub"], default="auto")
    p.add_argument("--no-zoom", action="store_true")
    args = p.parse_args()
    summary = json.loads(Path(args.summary).read_text()) if args.summary else None
    report = assess(args.image, summary, args.backend, do_zoom=not args.no_zoom)
    print(json.dumps(report.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
