"""Unified scoring "pseudo-API" sub-agent.

No HTTP — just a clean Python module that other code (council loop,
bench scripts, ad-hoc analysis) calls to score one or compare two
solutions on a unified objective combining:

  - operational_cost   (raw $ from solver.metrics)
  - quality_index      (0..1 from svrptw.metrics.score_solution)
  - visual_score       (0..1 from a VLM judge, optional)

The final ``unified`` is in [0, 1], higher = better. Component
contributions are returned in ``UnifiedScore.breakdown`` so callers
can show why one beat the other (and detect "dissent" — e.g. cost
prefers A but quality + visual prefer B).

Default weights (renormalize when visual is None):
  operational_cost: 0.40
  quality_index   : 0.40
  visual_score    : 0.20

Pair-comparison (``compare``) optionally renders a side-by-side image
via ``svrptw.viz.renderer.render_llm_compare`` and asks the VLM for a
relative score in addition to the per-image visual scores.
"""
from __future__ import annotations

import logging
import math
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from svrptw.config import Settings
from svrptw.io import Instance
from svrptw.metrics import SolutionQualityScore, score_solution
from svrptw.solvers.common.solution import Solution, evaluate

_LOG = logging.getLogger("svrptw.scoring.sub_api")

# Default unified weights. Sum to 1.0 when visual is present;
# the visual weight is redistributed across the other two when
# ``visual_score`` is None (see ``_renorm_weights``).
DEFAULT_WEIGHTS: dict[str, float] = {
    "operational_cost": 0.40,
    "quality_index":    0.40,
    "visual_score":     0.20,
}


@dataclass
class UnifiedScore:
    """Per-solution unified scorecard.

    Attributes:
      operational_cost: raw $ from solver (lower=better, normalized internally)
      quality_index:    0..1 structural score (higher=better)
      visual_score:     0..1 VLM judge score (higher=better); None when
                        no VLM was called.
      unified:          weighted combination in [0,1] (higher=better)
      breakdown:        per-component normalized contribution (sums to ``unified``)
      raw:              dict of raw inputs (cost_max, weights used, etc.)
    """
    operational_cost: float
    quality_index: float
    visual_score: Optional[float]
    unified: float
    breakdown: dict[str, float] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _renorm_weights(weights: dict[str, float],
                    have_visual: bool) -> dict[str, float]:
    """Redistribute weight mass so the active components sum to 1.0.

    If ``have_visual`` is False, the visual_score weight is dropped and
    its mass is split proportionally between cost and quality.
    """
    w = dict(weights)
    if not have_visual:
        w.pop("visual_score", None)
    total = sum(v for v in w.values() if v > 0)
    if total <= 0:
        # Pathological — caller asked for an all-zero weighting.
        # Fall back to uniform across whatever components are present.
        return {k: 1.0 / max(1, len(w)) for k in w}
    return {k: (v / total) for k, v in w.items()}


def _instance_cost_max(inst: Instance, settings: Settings) -> float:
    """A coarse but deterministic upper bound on operational cost.

    Used as the normaliser when the caller doesn't supply ``cost_max``.
    Dropping every customer is a valid upper bound: cost = N * miss
    penalty. Adds a small headroom (1.5x) so realistic solutions
    register in the upper half of the [0, 1] cost-score band.
    """
    miss = float(settings.economics.hard_late_penalty)
    return max(1.0, float(inst.num_customers) * miss * 1.5)


def _operational_cost_score(cost: float, cost_max: float) -> float:
    """Map raw $ cost to [0, 1] (higher = better).

    Rule: 1 - clamp(cost / cost_max, 0, 1). A cost at zero scores 1.0;
    a cost at-or-above the upper bound scores 0.0. Non-finite or
    negative cost scores 0.0 (treated as worst).
    """
    if not math.isfinite(cost) or cost < 0:
        return 0.0
    if cost_max <= 0:
        return 0.0
    return max(0.0, 1.0 - min(1.0, float(cost) / float(cost_max)))


_VISUAL_PROMPT_SINGLE = (
    "You are evaluating the visual quality of a vehicle routing solution. "
    "Each colored polyline is one truck's route from a depot. Score the "
    "image on overall structural quality from 0.0 (chaotic, many "
    "crossings, overlapping territories) to 1.0 (clean, well-separated "
    "routes, low crossings). "
    "Return ONLY valid JSON of the form "
    '{\"score\": <float 0..1>, \"rationale\": \"<one-sentence reason>\"}. '
    "Do not include code fences or any other text."
)


_VISUAL_PROMPT_PAIR = (
    "You are comparing TWO vehicle routing solutions for the same "
    "instance, shown side-by-side (LEFT = A, RIGHT = B). Decide which "
    "has better structural visual quality (clean separation, low "
    "crossings, balanced territories). "
    "Return ONLY valid JSON of the form "
    '{\"score\": <float 0..1>, \"rationale\": \"<one-sentence reason>\"} '
    "where score=1.0 means B is much better than A, score=0.0 means A "
    "is much better than B, and 0.5 is a tie. Do not include code "
    "fences or any other text."
)


def _call_visual_judge(image_path: Path,
                       judge_model: str,
                       judge_kind: str,
                       prompt: str = _VISUAL_PROMPT_SINGLE,
                       extra_image_paths: Optional[list[Path]] = None,
                       ) -> Optional[float]:
    """Run a VLM judge and return a 0..1 score, or None on failure.

    ``judge_kind`` selects the transport. Currently only ``"lmstudio"``
    is wired (Qwen-4B local). Other values short-circuit to None so
    the pipeline degrades gracefully when the LM Studio server isn't
    running.
    """
    if judge_kind != "lmstudio":
        _LOG.debug("unknown judge_kind=%r — visual_score will be None",
                   judge_kind)
        return None
    try:
        from svrptw.logic.lmstudio_client import call_model
    except Exception as e:
        _LOG.debug("lmstudio_client import failed: %s", e)
        return None
    try:
        resp = call_model(
            model=judge_model,
            prompt_text=prompt,
            image_path=Path(image_path),
            extra_image_paths=extra_image_paths,
            max_tokens=240,
        )
    except Exception as e:
        _LOG.debug("VLM call raised: %s", e)
        return None
    if resp is None or resp.score is None:
        return None
    # Clamp to the documented range; some models drift a bit.
    return max(0.0, min(1.0, float(resp.score)))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def score(inst: Instance,
          sol: Solution,
          *,
          settings: Optional[Settings] = None,
          weights: Optional[dict[str, float]] = None,
          cost_max: Optional[float] = None,
          quality: Optional[SolutionQualityScore] = None,
          visual_image_path: Optional[Path] = None,
          visual_compare_with: Optional[Path] = None,
          judge_model: str = "qwen/qwen3-vl-4b",
          judge_kind: str = "lmstudio") -> UnifiedScore:
    """Score a single solution on the unified objective.

    The cost component is normalized against ``cost_max``; if not
    supplied, a deterministic per-instance bound is computed from the
    miss-penalty (``_instance_cost_max``). For batch scoring, the bench
    typically derives ``cost_max`` from the worst cost in the batch and
    passes it explicitly so all rows share a normaliser.

    ``visual_compare_with`` is accepted but ignored here (only meaningful
    in ``compare``); kept in the signature so callers can pass kw-args
    once and reuse them across single + paired scoring.
    """
    s = settings or Settings()
    if quality is None:
        quality = score_solution(inst, sol)
    # Prefer the cost the solver itself reported (it knows about extra
    # economic terms like fixed costs); fall back to evaluator on miss.
    cost = float(sol.metrics.get("operational_cost", float("nan")))
    if not math.isfinite(cost):
        cost = float(evaluate(inst, sol, s).get("operational_cost",
                                                 float("inf")))

    cmax = float(cost_max) if cost_max is not None else _instance_cost_max(inst, s)
    cost_score = _operational_cost_score(cost, cmax)
    q_score = max(0.0, min(1.0, float(quality.quality_index)))

    visual_score: Optional[float] = None
    if visual_image_path is not None:
        visual_score = _call_visual_judge(
            Path(visual_image_path), judge_model, judge_kind,
            prompt=_VISUAL_PROMPT_SINGLE,
        )

    use_w = _renorm_weights(weights or DEFAULT_WEIGHTS,
                            have_visual=visual_score is not None)
    parts: dict[str, float] = {
        "operational_cost": cost_score * use_w.get("operational_cost", 0.0),
        "quality_index":    q_score    * use_w.get("quality_index", 0.0),
    }
    if visual_score is not None:
        parts["visual_score"] = visual_score * use_w.get("visual_score", 0.0)
    unified = sum(parts.values())
    return UnifiedScore(
        operational_cost=cost,
        quality_index=q_score,
        visual_score=visual_score,
        unified=max(0.0, min(1.0, unified)),
        breakdown=parts,
        raw={
            "cost_max": cmax,
            "cost_score": cost_score,
            "weights_in": dict(weights or DEFAULT_WEIGHTS),
            "weights_used": use_w,
            "judge_model": judge_model,
            "judge_kind": judge_kind,
            "n_routes": quality.n_routes,
            "inter_route_crossings": quality.inter_route_crossings,
        },
    )


def _render_pair_image(inst: Instance,
                       sol_a: Solution,
                       sol_b: Solution,
                       out_dir: Optional[Path] = None) -> Optional[Path]:
    """Render a side-by-side LLM-comparison image, return its path.

    Uses ``svrptw.viz.renderer.render_llm_compare`` with stable color
    assignment across the two panels. Returns None if rendering fails
    (matplotlib missing on a headless box, basemap network blip, ...).
    """
    try:
        from svrptw.viz.renderer import render_llm_compare
    except Exception as e:
        _LOG.debug("renderer import failed: %s", e)
        return None
    out_dir = Path(out_dir) if out_dir else Path(tempfile.mkdtemp(
        prefix="svrptw_pair_"))
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        cmap: dict = {}
        a_path = out_dir / f"{inst.instance_id}__A.png"
        b_path = out_dir / f"{inst.instance_id}__B.png"
        render_llm_compare(inst, sol_a, a_path, route_color_map=cmap,
                           title="A")
        render_llm_compare(inst, sol_b, b_path, route_color_map=cmap,
                           title="B")
    except Exception as e:
        _LOG.debug("render_llm_compare failed: %s", e)
        return None
    # We return the A-path — the pair image is delivered as A + B via
    # ``extra_image_paths`` to the VLM call (it accepts a list).
    return a_path


def compare(inst: Instance,
            sol_a: Solution,
            sol_b: Solution,
            *,
            settings: Optional[Settings] = None,
            weights: Optional[dict[str, float]] = None,
            cost_max: Optional[float] = None,
            visual_image_paths: Optional[tuple[Path, Path]] = None,
            do_pair_visual: bool = False,
            judge_model: str = "qwen/qwen3-vl-4b",
            judge_kind: str = "lmstudio") -> dict[str, Any]:
    """Score two solutions and pick a winner on the unified objective.

    If ``visual_image_paths`` is supplied, each side is scored
    individually with the per-image VLM prompt. If ``do_pair_visual``
    is True, a side-by-side image is rendered and asked for a relative
    A-vs-B score; the relative score is folded into ``visual_score``
    of both sides as +/- delta from 0.5.

    Returns dict::

      {a: UnifiedScore, b: UnifiedScore,
       winner: 'a' | 'b' | 'tie',
       margin: float,            # |unified_a - unified_b|
       dissent_flags: [str, ...] # e.g. "cost_prefers_a but quality_prefers_b"
       pair_visual_score: float | None}
    """
    s = settings or Settings()
    img_a = visual_image_paths[0] if visual_image_paths else None
    img_b = visual_image_paths[1] if visual_image_paths else None
    # Share a cost_max so both sides are normalised against the same scale.
    if cost_max is None:
        ca = float(sol_a.metrics.get("operational_cost", float("nan")))
        cb = float(sol_b.metrics.get("operational_cost", float("nan")))
        finite = [c for c in (ca, cb) if math.isfinite(c)]
        cost_max = (max(finite) * 1.5) if finite else _instance_cost_max(inst, s)


    sa = score(inst, sol_a, settings=s, weights=weights, cost_max=cost_max,
               visual_image_path=img_a, judge_model=judge_model,
               judge_kind=judge_kind)
    sb = score(inst, sol_b, settings=s, weights=weights, cost_max=cost_max,
               visual_image_path=img_b, judge_model=judge_model,
               judge_kind=judge_kind)

    pair_score: Optional[float] = None
    if do_pair_visual:
        pair_dir = Path(tempfile.mkdtemp(prefix="svrptw_pair_"))
        a_path = _render_pair_image(inst, sol_a, sol_b, out_dir=pair_dir)
        if a_path is not None:
            b_path = a_path.with_name(a_path.name.replace("__A.", "__B."))
            pair_score = _call_visual_judge(
                a_path, judge_model, judge_kind,
                prompt=_VISUAL_PROMPT_PAIR,
                extra_image_paths=[b_path] if b_path.exists() else None,
            )
        if pair_score is not None:
            # Fold the relative-judge into both sides' visual_score
            # (delta from 0.5 ~ "B is better than A" by this much).
            delta = pair_score - 0.5
            cur_a = sa.visual_score if sa.visual_score is not None else 0.5
            cur_b = sb.visual_score if sb.visual_score is not None else 0.5
            sa = score(inst, sol_a, settings=s, weights=weights,
                       cost_max=cost_max,
                       visual_image_path=None, judge_model=judge_model,
                       judge_kind=judge_kind)
            sb = score(inst, sol_b, settings=s, weights=weights,
                       cost_max=cost_max,
                       visual_image_path=None, judge_model=judge_model,
                       judge_kind=judge_kind)
            sa = _replace_visual(sa, max(0.0, min(1.0, cur_a - delta)))
            sb = _replace_visual(sb, max(0.0, min(1.0, cur_b + delta)))


    margin = abs(sa.unified - sb.unified)
    if margin < 1e-6:
        winner = "tie"
    elif sa.unified > sb.unified:
        winner = "a"
    else:
        winner = "b"

    # Dissent: which component prefers which side? Useful for the
    # cost-vs-quality tension that motivated this whole API.
    dissents: list[str] = []
    cost_winner = "a" if sa.operational_cost < sb.operational_cost else (
        "b" if sb.operational_cost < sa.operational_cost else "tie")
    qual_winner = "a" if sa.quality_index > sb.quality_index else (
        "b" if sb.quality_index > sa.quality_index else "tie")
    vis_winner = "tie"
    if sa.visual_score is not None and sb.visual_score is not None:
        vis_winner = "a" if sa.visual_score > sb.visual_score else (
            "b" if sb.visual_score > sa.visual_score else "tie")
    if cost_winner != "tie" and qual_winner != "tie" and cost_winner != qual_winner:
        dissents.append(f"cost_prefers_{cost_winner}_but_quality_prefers_{qual_winner}")
    if vis_winner != "tie" and cost_winner != "tie" and vis_winner != cost_winner:
        dissents.append(f"cost_prefers_{cost_winner}_but_visual_prefers_{vis_winner}")
    if vis_winner != "tie" and qual_winner != "tie" and vis_winner != qual_winner:
        dissents.append(f"quality_prefers_{qual_winner}_but_visual_prefers_{vis_winner}")
    return {
        "a": sa, "b": sb,
        "winner": winner, "margin": margin,
        "dissent_flags": dissents,
        "pair_visual_score": pair_score,
        "cost_winner": cost_winner,
        "quality_winner": qual_winner,
        "visual_winner": vis_winner,
    }


def _replace_visual(s: UnifiedScore, new_visual: float) -> UnifiedScore:
    """Return a copy of ``s`` with ``visual_score`` updated and the
    unified + breakdown re-derived using ``s.raw['weights_used']`` if
    visual was already in play, else the default split.
    """
    have_visual = True
    base_w = s.raw.get("weights_in") or DEFAULT_WEIGHTS
    use_w = _renorm_weights(base_w, have_visual=True)
    cost_score = s.raw.get("cost_score", 0.0)
    parts = {
        "operational_cost": cost_score * use_w.get("operational_cost", 0.0),
        "quality_index":    s.quality_index * use_w.get("quality_index", 0.0),
        "visual_score":     new_visual * use_w.get("visual_score", 0.0),
    }
    unified = sum(parts.values())
    return UnifiedScore(
        operational_cost=s.operational_cost,
        quality_index=s.quality_index,
        visual_score=new_visual,
        unified=max(0.0, min(1.0, unified)),
        breakdown=parts,
        raw={**s.raw, "weights_used": use_w, "visual_overridden": True},
    )


def score_batch(inst_paths: Iterable[Path],
                solver_fn: Callable[[Instance], Solution],
                *,
                settings: Optional[Settings] = None,
                weights: Optional[dict[str, float]] = None,
                cost_max: Optional[float] = None) -> list[UnifiedScore]:
    """Score a batch of (instance, solution-from-solver_fn) pairs."""
    from svrptw.io.instance import load_instance
    s = settings or Settings()
    out: list[UnifiedScore] = []
    for p in inst_paths:
        inst = load_instance(Path(p))
        sol = solver_fn(inst)
        out.append(score(inst, sol, settings=s, weights=weights,
                         cost_max=cost_max))
    return out
