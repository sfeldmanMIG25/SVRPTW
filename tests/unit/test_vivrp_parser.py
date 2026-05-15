"""Smoke test for the ViVRP JSON parser fallback."""
import pytest

from svrptw.vivrp.assessor import _parse_json_loose


def test_clean_json():
    out = _parse_json_loose(
        '{"overall_score": 7, "clustering_score": 5, "geometry_score": 8, '
        '"interpretability_score": 6, "notes": "routes are tight"}'
    )
    assert out["overall_score"] == 7
    assert out["interpretability_score"] == 6


def test_regex_fallback_on_unescaped_quote_in_notes():
    bad = (
        '{"overall_score": 7, "clustering_score": 5, "geometry_score": 8, '
        '"interpretability_score": 6, "notes": "vehicle "3" detours"}'
    )
    out = _parse_json_loose(bad)
    assert out["overall_score"] == 7
    assert out["clustering_score"] == 5
    assert out["geometry_score"] == 8
    assert out["interpretability_score"] == 6


def test_no_scores_raises():
    with pytest.raises(ValueError):
        _parse_json_loose("nothing useful here")
