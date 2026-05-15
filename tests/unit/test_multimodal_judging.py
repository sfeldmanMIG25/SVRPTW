"""SPEC-6-LOGIC-02 multimodal-pipeline tests (renderer + text + dual-judge)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from svrptw.config import Settings
from svrptw.instances_gen.synthetic import generate
from svrptw.logic.solution_text import to_text
from svrptw.solvers.classical import greedy as greedy_mod
from svrptw.viz.renderer import render_solution


def _make_sol(N=20, seed=0):
    inst = generate(N=N, seed=seed)
    sol = greedy_mod.solve(inst, Settings())
    return inst, sol


def test_renderer_info_bearing_color(tmp_path: Path):
    """fair_mode renders different load utilisations as different brightness
    (viridis 0→1). A nearly-full route should have noticeably brighter pixels
    than a nearly-empty one."""
    from svrptw.solvers.common import Route, Solution, evaluate

    inst = generate(N=20, seed=42)
    # Build a hand-crafted "full" solution: pack high-demand customers.
    by_demand = sorted(inst.customers, key=lambda c: -c.demand)
    heavy = [c.id for c in by_demand[:5]]   # high util
    light = [by_demand[-1].id]              # low util
    sol_full = Solution(
        instance_id=inst.instance_id, routes=[Route(customers=heavy)],
        solver="hand", wall_clock_seconds=0.0, budget_seconds=0.0,
        feasible=True, metrics={},
    )
    sol_full.metrics = evaluate(inst, sol_full, Settings())
    sol_light = Solution(
        instance_id=inst.instance_id, routes=[Route(customers=light)],
        solver="hand", wall_clock_seconds=0.0, budget_seconds=0.0,
        feasible=True, metrics={},
    )
    sol_light.metrics = evaluate(inst, sol_light, Settings())

    p_full = tmp_path / "full.png"
    p_light = tmp_path / "light.png"
    render_solution(inst, sol_full, p_full, fair_mode=True)
    render_solution(inst, sol_light, p_light, fair_mode=True)

    full_img = np.asarray(Image.open(p_full).convert("RGB"))
    light_img = np.asarray(Image.open(p_light).convert("RGB"))

    # The two solutions have different load utilisations so their info-bearing
    # rendering must produce visibly different pixels — strictly different
    # images (the route color encodes util, not identity).
    assert full_img.shape == light_img.shape
    pix_diff = (full_img.astype(int) - light_img.astype(int))
    nonzero_frac = float(np.mean(np.abs(pix_diff).sum(axis=-1) > 10))
    assert nonzero_frac > 0.005, (
        f"only {nonzero_frac:.4f} of pixels differ — images look identical "
        "but loads were different (renderer is not info-bearing)"
    )

    # Viridis @ util=1.0 is bright yellow with high R+G, low B.
    # Viridis @ util=0.05 is dark purple with low R+G, moderate B.
    # Mean R-channel value (excluding white BG ≥ 250) should be higher
    # for the full-load image's route pixels.
    full_route = full_img[(full_img.sum(axis=-1) < 700)]
    light_route = light_img[(light_img.sum(axis=-1) < 700)]
    if len(full_route) > 0 and len(light_route) > 0:
        # Yellow has R≈253, dark purple has R≈72 — full > light by 2-3×.
        assert full_route[:, 0].mean() > light_route[:, 0].mean(), (
            f"full route mean R {full_route[:, 0].mean():.1f} not greater than "
            f"light route mean R {light_route[:, 0].mean():.1f}"
        )


def test_solution_text_roundtrip():
    """to_text returns deterministic markdown for the same input."""
    inst, sol = _make_sol(N=15, seed=1)
    t1 = to_text(inst, sol)
    t2 = to_text(inst, sol)
    assert t1 == t2
    # Must contain the required section headers (Solver section is now
    # the anonymous "## <label>" or "## <solver-name>" first heading).
    for sec in ("## Aggregate", "## Routes", "## TW tightness"):
        assert sec in t1
    # First non-empty line should be a heading naming the plan.
    first_h = next(ln for ln in t1.splitlines() if ln.startswith("## "))
    assert "Aggregate" not in first_h  # not the second section


def test_solution_text_contains_per_route_numerics():
    """Each non-empty route should appear in the table with load + util%."""
    inst, sol = _make_sol(N=25, seed=2)
    txt = to_text(inst, sol)
    # Header columns present
    for col in ("n_cust", "load", "util%", "dist_mi", "time_min", "tw_slack"):
        assert col in txt
    # At least one route row (the greedy solution always has ≥ 1 route).
    rows = [line for line in txt.splitlines() if line.startswith("| ") and "%" in line]
    assert len(rows) >= 1


def test_combined_judge_prompt_contains_both():
    """The combined-judge prompt must reference Solution A AND Solution B."""
    from svrptw.logic.dataset import _PAIR_PROMPT_HEADER
    inst, sol_a = _make_sol(N=12, seed=3)
    inst_b, sol_b = _make_sol(N=12, seed=4)
    prompt = (
        _PAIR_PROMPT_HEADER
        + "\n\n## Solution A details:\n" + to_text(inst, sol_a)
        + "\n\n## Solution B details:\n" + to_text(inst, sol_b)
    )
    assert "## Solution A details" in prompt
    assert "## Solution B details" in prompt
    # Header tells the VLM both plans use the same color encoding.
    assert "viridis" in prompt.lower()


def test_pair_schema_shape():
    from svrptw.logic.committee import PAIR_SCHEMA
    assert set(PAIR_SCHEMA["required"]) == {"score_a", "score_b",
                                              "rationale", "confidence"}


def test_solution_text_is_blind_when_blind_label_given():
    """blind_label suppresses solver identity in the markdown output."""
    inst, sol = _make_sol(N=12, seed=7)
    txt_blind = to_text(inst, sol, blind_label="Plan A")
    txt_named = to_text(inst, sol)
    # Solver name must not appear in blinded output.
    assert sol.solver not in txt_blind
    assert "Plan A" in txt_blind
    # Without blind_label, solver name DOES appear.
    assert sol.solver in txt_named


def test_judge_pair_combined_uses_blind_labels():
    """_judge_pair_combined must call the committee with a prompt that
    contains 'Plan A' and 'Plan B' but neither raw solver name."""
    from unittest.mock import patch
    from svrptw.logic import dataset as ds
    from svrptw.logic.committee import PairLabel

    inst, sol_a = _make_sol(N=10, seed=8)
    inst_b, sol_b = _make_sol(N=10, seed=9)
    sol_a.solver = "portfolio@10"
    sol_b.solver = "pyvrp@30"

    captured = {"prompt": ""}

    def fake_label_pair(self, instance, s_a, s_b, prompt, image_a, image_b):
        captured["prompt"] = prompt
        return PairLabel(
            score_a=0.7, score_b=0.4, score_a_std=0.0, score_b_std=0.0,
            rationale="mock", authoritative=True, n_responders=1,
        )

    # tmp dir for rendered images + a real (empty) committee instance.
    from pathlib import Path as _P
    import tempfile
    from svrptw.logic.committee import OpenRouterCommittee, _TierConfig
    empty = _TierConfig("X", (), 0.0, False)
    committee = OpenRouterCommittee(
        tiers_primary=(empty,), tier_stealth=None, tier_tiebreaker=empty,
        include_gemini=False,
    )
    with tempfile.TemporaryDirectory() as td:
        cache = _P(td)
        with patch.object(OpenRouterCommittee, "label_pair", fake_label_pair):
            sa, sb, auth, rationale = ds._judge_pair_combined(
                committee, inst, sol_a, sol_b, cache, rng=None,
            )

    prompt = captured["prompt"]
    assert "Plan A" in prompt
    assert "Plan B" in prompt
    # Crucial: solver names must NOT leak through.
    assert "portfolio@10" not in prompt
    assert "pyvrp@30" not in prompt


def test_call_model_accepts_extra_images():
    """call_model signature accepts extra_image_paths kwarg (no network call)."""
    import inspect
    from svrptw.logic.openrouter_client import call_model
    sig = inspect.signature(call_model)
    assert "extra_image_paths" in sig.parameters
    assert "schema" in sig.parameters
