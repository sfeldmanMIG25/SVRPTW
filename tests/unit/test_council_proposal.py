"""SPEC-8-COUNCIL-01 — proposal schema + validator tests."""
from __future__ import annotations

from svrptw.config import Settings
from svrptw.council.proposal import (
    OperatorContext, OperatorProposal, validate_proposal,
)
from svrptw.instances_gen.synthetic import generate
from svrptw.solvers.classical import greedy as greedy_mod


_GOOD_CODE = '''
def operator(solution, context):
    # Trivial no-op operator — passes signature + smoke gates.
    return None
'''

_GOOD_TEST = '''
def test_no_op_returns_none():
    # The operator takes (solution, context) and may return None.
    class _CtxStub:
        pass
    assert operator(None, _CtxStub()) is None
'''

_BAD_SIG_CODE = '''
def operator(x):
    return None
'''

_BANNED_IMPORT = '''
import os
def operator(solution, context):
    return None
'''

_NON_PARSING = '''
def operator(solution, context:
'''


def test_proposal_id_is_deterministic():
    p1 = OperatorProposal(rationale="r", code=_GOOD_CODE, unit_test=_GOOD_TEST)
    p2 = OperatorProposal(rationale="r", code=_GOOD_CODE, unit_test=_GOOD_TEST)
    assert p1.proposal_id == p2.proposal_id


def test_good_proposal_passes_all_gates():
    p = OperatorProposal(
        rationale="trivial no-op", code=_GOOD_CODE, unit_test=_GOOD_TEST,
    )
    d = validate_proposal(p)
    assert d.parses
    assert d.signature_ok
    assert d.unit_test_passed
    assert d.smoke_passed
    assert d.can_run_bench
    assert d.operator_callable is not None
    assert d.reject_reason is None


def test_non_parsing_rejected():
    p = OperatorProposal(
        rationale="bad syntax", code=_NON_PARSING, unit_test=_GOOD_TEST,
    )
    d = validate_proposal(p)
    assert not d.signature_ok
    assert not d.can_run_bench
    assert d.reject_reason == "parse-or-signature-failed"


def test_wrong_signature_rejected():
    p = OperatorProposal(
        rationale="bad sig", code=_BAD_SIG_CODE, unit_test=_GOOD_TEST,
    )
    d = validate_proposal(p)
    assert not d.signature_ok
    assert d.reject_reason == "parse-or-signature-failed"


def test_banned_import_rejected():
    p = OperatorProposal(
        rationale="dangerous import", code=_BANNED_IMPORT, unit_test=_GOOD_TEST,
    )
    d = validate_proposal(p)
    assert not d.signature_ok


def test_unit_test_failure_rejected():
    bad_test = '''
def test_fail():
    assert False, "deliberately failing test"
'''
    p = OperatorProposal(rationale="x", code=_GOOD_CODE, unit_test=bad_test)
    d = validate_proposal(p)
    assert d.signature_ok
    assert not d.unit_test_passed
    assert d.reject_reason is not None
    assert "unit-test" in d.reject_reason


def test_raising_operator_rejected():
    """A raising operator fails at smoke even if its unit test tolerates the raise."""
    raising = '''
def operator(solution, context):
    raise RuntimeError("boom")
'''
    # Tolerant unit test so we exercise the smoke gate specifically.
    tolerant_test = '''
import pytest
def test_raises():
    with pytest.raises(RuntimeError):
        operator(None, object())
'''
    p = OperatorProposal(rationale="x", code=raising, unit_test=tolerant_test)
    d = validate_proposal(p)
    assert d.signature_ok
    assert d.unit_test_passed     # the tolerant test passes
    assert not d.smoke_passed
    assert "operator raised" in (d.reject_reason or "")
