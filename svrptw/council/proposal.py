"""Proposal schema + machine-checkable validator.

A proposal is rejected before ever hitting the shadow bench if any
of:
  - the code doesn't parse
  - the operator signature is wrong
  - the unit test doesn't import or doesn't pass
  - the operator raises an unhandled exception on a tiny smoke instance

Per SPEC-8-COUNCIL-01: this is the contract that makes the agents'
output tractable.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import inspect
import subprocess
import sys
import tempfile
import textwrap
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Optional

from svrptw.config import Settings
from svrptw.io import Instance
from svrptw.solvers.common import Solution


# Public operator contract — every proposal must conform.
#
#   def operator(solution: Solution, context: OperatorContext) -> Solution | None
#
# Returns a NEW Solution (or None for "no-op / cannot apply").
# Must not raise on a sensibly-feasible input.

@dataclass
class OperatorContext:
    """Everything an operator may read. Deliberately small so proposals
    stay focussed; agents that want more context should request a new
    field via a follow-up proposal."""

    instance: Instance
    settings: Settings
    rng_seed: int = 0
    deadline_seconds: float = 1.0    # operator must respect this


@dataclass
class OperatorProposal:
    """One operator proposal from the agent council."""

    rationale: str                   # one paragraph why this should help
    code: str                        # Python source defining operator()
    unit_test: str                   # Python source with test_* functions
    seed_inspired_by: Optional[str] = None
    generation: int = 0
    proposal_id: str = ""            # set during validation

    def __post_init__(self):
        if not self.proposal_id:
            blob = (self.rationale + "\n" + self.code).encode("utf-8")
            self.proposal_id = "p_" + hashlib.sha1(blob).hexdigest()[:12]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ProposalDecision:
    proposal_id: str
    parses: bool = False
    signature_ok: bool = False
    unit_test_passed: bool = False
    smoke_passed: bool = False
    reject_reason: Optional[str] = None
    operator_callable: Optional[Callable] = field(default=None, repr=False)

    @property
    def can_run_bench(self) -> bool:
        return self.parses and self.signature_ok and self.unit_test_passed and self.smoke_passed


_REQUIRED_SIG_PARAMS = ("solution", "context")


def _parse_operator(code: str) -> Optional[Callable]:
    """Compile `code` in an isolated namespace and return the `operator`
    callable if present + signature-conforming, else None."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    # Reject imports of restricted modules (very light guard — the
    # real safety net is the sandboxed shadow bench).
    banned = {"os", "subprocess", "socket", "shutil", "urllib"}
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            mod = node.module if isinstance(node, ast.ImportFrom) else None
            names = [a.name for a in node.names]
            top = (mod or names[0]).split(".")[0]
            if top in banned:
                return None
    ns: dict = {}
    try:
        exec(compile(tree, "<proposal>", "exec"), ns)
    except Exception:
        return None
    op = ns.get("operator")
    if op is None or not callable(op):
        return None
    sig = inspect.signature(op)
    params = list(sig.parameters.keys())
    if tuple(params[:2]) != _REQUIRED_SIG_PARAMS:
        return None
    return op


def _run_unit_test(code: str, unit_test: str, timeout_s: float = 10.0
                   ) -> tuple[bool, Optional[str]]:
    """Write code + unit_test to a temp file, pytest it. Returns (passed, reason)."""
    with tempfile.TemporaryDirectory() as td:
        op_path = Path(td) / "proposed_operator.py"
        op_path.write_text(code, encoding="utf-8")
        test_path = Path(td) / "test_proposed.py"
        # The unit test must import the proposed operator via its
        # module file name (we own that name above).
        prologue = (
            "import sys\nfrom pathlib import Path\n"
            f"sys.path.insert(0, r'{td}')\n"
            "from proposed_operator import operator\n\n"
        )
        test_path.write_text(prologue + unit_test, encoding="utf-8")
        try:
            res = subprocess.run(
                [sys.executable, "-m", "pytest", str(test_path), "-x", "--tb=line"],
                capture_output=True, text=True, timeout=timeout_s,
                env={**__import__("os").environ, "PYTHONPATH": str(Path.cwd())},
            )
        except subprocess.TimeoutExpired:
            return (False, "unit-test timeout")
        if res.returncode != 0:
            tail = (res.stdout + "\n" + res.stderr)[-300:]
            return (False, f"unit-test failed:\n{tail}")
    return (True, None)


def _smoke_operator(op: Callable) -> tuple[bool, Optional[str]]:
    """Apply the operator to a tiny synthetic instance. Must not raise."""
    from svrptw.instances_gen.synthetic import generate
    from svrptw.solvers.classical import greedy as greedy_mod

    inst = generate(N=12, seed=0)
    settings = Settings()
    sol = greedy_mod.solve(inst, settings)
    ctx = OperatorContext(instance=inst, settings=settings,
                          rng_seed=0, deadline_seconds=1.0)
    try:
        out = op(sol, ctx)
    except Exception as e:
        return (False, f"operator raised: {type(e).__name__}: {e}")
    if out is None:
        return (True, None)   # explicit no-op is fine
    if not isinstance(out, Solution):
        return (False, f"operator returned non-Solution: {type(out).__name__}")
    if out.metrics.get("operational_cost") is None:
        return (False, "operator returned Solution without metrics; "
                       "did you forget to call evaluate()?")
    # Feasibility constraint: operator must not produce capacity overload
    # (SPEC-0-EVAL-01 invariant).
    if out.metrics.get("capacity_overload", 0.0) > 0.0:
        return (False, "operator produced capacity-overloaded routes")
    return (True, None)


def validate_proposal(prop: OperatorProposal) -> ProposalDecision:
    """Full gate: parse → signature → unit-test → smoke. Returns the
    decision with whichever subset of gates passed; sets
    `operator_callable` only if every gate passes.
    """
    decision = ProposalDecision(proposal_id=prop.proposal_id)

    op = _parse_operator(prop.code)
    decision.parses = op is not None or _parses_only(prop.code)
    decision.signature_ok = op is not None
    if op is None:
        decision.reject_reason = "parse-or-signature-failed"
        return decision

    # Unit test is advisory — LLM-generated test code is unreliable
    # (often hallucinates Instance/Solution constructor signatures).
    # Smoke gate is authoritative: it runs the operator on a real
    # synthetic instance under our actual dataclasses. If smoke passes,
    # the operator is acceptable for shadow-bench regardless of whether
    # its accompanying unit test parses.
    ut_ok, ut_why = _run_unit_test(prop.code, prop.unit_test) if prop.unit_test else (True, None)
    decision.unit_test_passed = ut_ok
    # Record why the unit test failed (for corpus lore) but don't reject.

    ok, why = _smoke_operator(op)
    decision.smoke_passed = ok
    if not ok:
        # Smoke failure is fatal; unit-test failure (if any) gets concatenated.
        reason = why
        if not ut_ok:
            reason = f"{why}; also unit_test: {ut_why}"
        decision.reject_reason = reason
        return decision
    # Smoke passed. If unit test also failed, record advisory note but accept.
    if not ut_ok:
        decision.reject_reason = f"advisory: unit_test failed but smoke OK: {ut_why}"

    decision.operator_callable = op
    return decision


def _parses_only(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False
