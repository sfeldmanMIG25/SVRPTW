"""Smoke tests for `svrptw.logic.multi_judge`.

No network calls. Uses the all-disabled path or constructs synthetic
verdicts directly so aggregation can be exercised without invoking
any real LLM SDK.
"""
from __future__ import annotations

from pathlib import Path

from svrptw.logic.multi_judge import (
    JudgeVerdict,
    MultiJudgeConsensus,
    aggregate_verdicts,
    judge_pair,
)


def _v(judge_id: str, tier: str, score: float | None,
        error: str | None = None) -> JudgeVerdict:
    return JudgeVerdict(
        judge_id=judge_id, judge_kind="test", tier=tier,
        score=score, rationale="r", confidence=0.9, latency_s=0.0,
        error=error,
    )


def test_judge_pair_all_disabled_returns_empty(tmp_path: Path) -> None:
    """No judges enabled → no calls, no crash, empty list + null consensus."""
    a = tmp_path / "a.png"
    b = tmp_path / "b.png"
    a.write_bytes(b"fake-png-bytes")
    b.write_bytes(b"fake-png-bytes")
    verdicts, consensus = judge_pair(
        a, b, prompt="ignored",
        enable_openrouter=False, enable_gemini=False,
        enable_anthropic_haiku=False, push_to_webui=False,
    )
    assert verdicts == []
    assert isinstance(consensus, MultiJudgeConsensus)
    assert consensus.score is None
    assert consensus.n_responded == 0
    assert consensus.n_failed == 0
    assert consensus.dissent is False
    assert consensus.contributing_tiers == []


def test_aggregate_all_failed_returns_none() -> None:
    """When every verdict is an error, consensus has no score."""
    verdicts = [
        _v("j1", "A", None, error="boom"),
        _v("j2", "gemini", None, error="boom"),
    ]
    cons = aggregate_verdicts(verdicts)
    assert cons.score is None
    assert cons.n_responded == 0
    assert cons.n_failed == 2
    assert cons.dissent is False
    assert cons.span is None


def test_aggregate_single_verdict_passes_through() -> None:
    """A single responder yields its own score; span is zero, no dissent."""
    cons = aggregate_verdicts([_v("solo", "A", 0.62)])
    assert cons.score == 0.62
    assert cons.n_responded == 1
    assert cons.n_failed == 0
    assert cons.span == 0.0
    assert cons.dissent is False
    assert cons.contributing_tiers == ["A"]


def test_aggregate_dissent_threshold_flips() -> None:
    """Two verdicts with span 0.5 (> 0.30 default) → dissent True."""
    cons = aggregate_verdicts([
        _v("a", "A", 0.2),
        _v("b", "gemini", 0.7),
    ])
    assert cons.score is not None
    assert cons.n_responded == 2
    assert cons.span is not None and abs(cons.span - 0.5) < 1e-9
    assert cons.dissent is True
    # Below threshold (span 0.2) → no dissent.
    cons2 = aggregate_verdicts([
        _v("a", "A", 0.5),
        _v("b", "gemini", 0.7),
    ])
    assert cons2.dissent is False


def test_aggregate_ignores_failed_verdicts_in_span() -> None:
    """A failed verdict mixed with successes is excluded from span/median."""
    cons = aggregate_verdicts([
        _v("a", "A", 0.4),
        _v("dead", "haiku", None, error="anthropic-not-available"),
        _v("b", "gemini", 0.6),
    ])
    assert cons.n_responded == 2
    assert cons.n_failed == 1
    assert cons.span is not None and abs(cons.span - 0.2) < 1e-9
    assert cons.dissent is False
