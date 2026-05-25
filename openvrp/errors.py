"""SPEC-OPENVRP-08 — Exception hierarchy.

Contract: malformed problems raise structured exceptions at build time;
merely *hard* problems never raise — they return a Solution with
status="infeasible" and actionable diagnostics.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class Issue:
    """A single validation issue tagged with severity, code, and location.

    `location` is dotted/bracketed pydantic-style path
    (e.g. ``stops[id=ACME-12].demand``) so callers can navigate straight to
    the offending field.
    """

    severity: Literal["error", "warning"]
    code: str
    message: str
    location: str = ""


class OpenVRPError(Exception):
    """Base class for every exception OpenVRP raises.

    No bare ``Exception`` / ``ValueError`` / ``ImportError`` escapes the
    public API — they are wrapped into one of these subclasses.
    """


class ProblemValidationError(OpenVRPError):
    """Raised by ``Problem.from_*`` when input is malformed.

    Carries a list of ``Issue`` records so the caller can inspect every
    rejected field at once instead of fixing them one by one.
    """

    def __init__(self, issues: list[Issue]):
        self.issues: list[Issue] = list(issues)
        errors = [i for i in self.issues if i.severity == "error"]
        head = errors[0] if errors else (self.issues[0] if self.issues else None)
        msg = (f"{len(errors)} error(s) in input: "
               + (f"{head.code} at {head.location}: {head.message}"
                  if head else "(no detail)"))
        super().__init__(msg)


class MissingExtra(OpenVRPError):
    """Raised when a feature requires an uninstalled optional extra.

    Carries the exact ``pip install`` command in the message so the
    caller can fix it immediately.
    """

    def __init__(self, extra: str, *, install: str | None = None,
                 reason: str | None = None):
        self.extra = extra
        self.install_command = install or f"pip install openvrp[{extra}]"
        why = f": {reason}" if reason else ""
        super().__init__(
            f"openvrp[{extra}] extra is required{why}. Install with: {self.install_command}"
        )


class GeometryUnavailable(OpenVRPError):
    """Raised by ``Solution.to_geojson`` when called on an OD-only solution
    without supplying ``points=`` for an approximate-line fallback.

    Per SPEC-OPENVRP-00 D14, OpenVRP never silently substitutes Euclidean
    lines for real along-network geometry.
    """


class SolveAborted(OpenVRPError):
    """Raised when an ``on_progress`` callback raised something other than
    ``StopSolve`` (which is the clean-cancellation sentinel)."""


class StopSolve(Exception):
    """Sentinel a caller raises from inside ``on_progress`` to end the
    search cleanly with the best-so-far solution.

    Intentionally not a subclass of ``OpenVRPError`` — it is a control-flow
    signal, not an error.
    """


@dataclass
class ValidationReport:
    """Aggregate of issues from a single build call. Errors raise; warnings
    propagate into ``Solution.diagnostics.input_warnings``."""

    issues: list[Issue] = field(default_factory=list)

    def add_error(self, code: str, message: str, location: str = "") -> None:
        self.issues.append(Issue("error", code, message, location))

    def add_warning(self, code: str, message: str, location: str = "") -> None:
        self.issues.append(Issue("warning", code, message, location))

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == "warning"]

    def raise_if_errors(self) -> None:
        if self.errors:
            raise ProblemValidationError(self.issues)


__all__ = [
    "Issue",
    "OpenVRPError",
    "ProblemValidationError",
    "MissingExtra",
    "GeometryUnavailable",
    "SolveAborted",
    "StopSolve",
    "ValidationReport",
]
