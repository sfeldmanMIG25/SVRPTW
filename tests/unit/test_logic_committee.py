"""Unit tests for SPEC-6-LOGIC-02 OpenRouter committee.

No network calls — `call_model` is monkeypatched.
"""
from __future__ import annotations

import statistics
from pathlib import Path
from unittest.mock import patch

import pytest

from svrptw.logic import committee as cm
from svrptw.logic.openrouter_client import ModelDeprecated, VLMResponse


def _vote(model: str, tier: str, score: float) -> cm.ModelVote:
    return cm.ModelVote(
        model=model, tier=tier, score=score,
        rationale="rationale for " + model, confidence=0.8, latency_s=0.1,
    )


def test_weighted_median_uniform_weights():
    out = cm._weighted_median([0.2, 0.5, 0.9], [1.0, 1.0, 1.0])
    assert out == 0.5


def test_weighted_median_skews_toward_heavy_weight():
    # Tier-A vote (weight 1.0) at 0.9 should pull median up vs Tier-C (0.4) at 0.2.
    out = cm._weighted_median([0.2, 0.9], [0.4, 1.0])
    assert out == 0.9


def test_aggregate_chooses_highest_weight_rationale():
    votes = [
        (_vote("model-a", "A", 0.7), 1.0),
        (_vote("model-b", "B", 0.7), 0.6),
        (_vote("model-c", "C", 0.7), 0.4),
    ]
    median, std, rationale = cm._aggregate(votes)
    assert median == 0.7
    assert std == 0.0
    assert rationale == "rationale for model-a"


def test_sanitize_replaces_instance_id():
    prompt = "Please judge instance OSM-Cambridge-N050-I000 with care."
    out = cm._sanitize_prompt(prompt, "OSM-Cambridge-N050-I000")
    assert "OSM-Cambridge-N050-I000" not in out
    assert "inst-" in out


def test_sanitize_idempotent_when_id_absent():
    prompt = "Please judge this plan."
    out = cm._sanitize_prompt(prompt, "OSM-Cambridge-N050-I000")
    assert out == prompt


def test_mark_dead_drops_from_pool(tmp_path: Path):
    cm._mark_dead("model-x", "404")
    assert cm._is_dead("model-x")
    with cm._dead_models_lock:
        cm._dead_models.discard("model-x")  # cleanup for test isolation


class _FakeInstance:
    instance_id = "OSM-Test-N050-I000"


class _FakeSolution:
    pass


def test_label_handles_total_failure(tmp_path: Path):
    """Every call errors → committee returns score=None, authoritative=False."""
    def boom(*a, **kw):
        raise RuntimeError("simulated network failure")

    with patch("svrptw.logic.committee.call_model", side_effect=boom):
        c = cm.OpenRouterCommittee(
            tiers_primary=(cm.TIER_A,),
            tier_stealth=None,
            tier_tiebreaker=cm._TierConfig("C", (), 0.4, False),
        )
        out = c.label(_FakeInstance(), _FakeSolution(), "judge it", None)
        assert out.score is None
        assert not out.authoritative
        assert out.n_responders == 0


def test_label_aggregates_partial_responders():
    """Three models respond, two fail → consensus + std computed on the three."""
    call_count = {"n": 0}

    def fake(model, prompt, image_path=None, **kw):
        call_count["n"] += 1
        if "fail" in model:
            raise RuntimeError("simulated")
        # First three models: 0.7, 0.75, 0.65 — std ≈ 0.04 (authoritative if n >= 6)
        score_map = {
            "google/gemma-4-31b-it:free": 0.70,
            "google/gemma-4-26b-a4b-it:free": 0.75,
            "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free": 0.65,
        }
        s = score_map.get(model, 0.5)
        return VLMResponse(
            model=model, score=s, rationale=f"r-{model}", confidence=0.8,
            raw_json={}, latency_s=0.05,
        )

    failing_tier = cm._TierConfig(
        "A",
        cm.TIER_A.models[:3] + ("provider/fail-a:free", "provider/fail-b:free"),
        1.0, False,
    )
    with patch("svrptw.logic.committee.call_model", side_effect=fake):
        c = cm.OpenRouterCommittee(
            tiers_primary=(failing_tier,),
            tier_stealth=None,
            tier_tiebreaker=cm._TierConfig("C", (), 0.4, False),
            authoritative_min_responders=3,
            authoritative_max_std=0.10,
        )
        out = c.label(_FakeInstance(), _FakeSolution(), "judge it", None)
    assert out.n_responders == 3
    assert out.score is not None and 0.65 <= out.score <= 0.75
    assert out.score_std < 0.10
    assert out.authoritative is True


def test_label_marks_deprecated_model_dead():
    """A 404 from a model removes it from the pool for the rest of the process."""
    def fake(model, *a, **kw):
        if model == "openrouter/owl-alpha":
            raise ModelDeprecated(f"{model} 404")
        return VLMResponse(
            model=model, score=0.5, rationale="r", confidence=0.8,
            raw_json={}, latency_s=0.05,
        )

    stealth_tier = cm._TierConfig("S", ("openrouter/owl-alpha",), 0.5, True)
    with patch("svrptw.logic.committee.call_model", side_effect=fake):
        c = cm.OpenRouterCommittee(
            tiers_primary=(cm._TierConfig("A", ("google/gemma-4-31b-it:free",), 1.0, False),),
            tier_stealth=stealth_tier,
            tier_tiebreaker=cm._TierConfig("C", (), 0.4, False),
            authoritative_min_responders=1,
            authoritative_max_std=1.0,
        )
        _ = c.label(_FakeInstance(), _FakeSolution(), "judge it", None)
        assert cm._is_dead("openrouter/owl-alpha")
    # cleanup
    with cm._dead_models_lock:
        cm._dead_models.discard("openrouter/owl-alpha")
