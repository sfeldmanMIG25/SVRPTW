"""SPEC-8-COUNCIL-01 — agent-council infrastructure.

Three loops:
  Loop 1 — operator proposer: agents propose Python operators conforming
           to a fixed signature, validated by unit test + shadow bench.
  Loop 2 — component merger: config diffs combining existing pieces.
  Loop 3 — distillation watcher: extract features/rules from kept
           variants into the LogicStudent.

This package implements the shared infrastructure (proposal schema,
memory, shadow bench). The agent prompts + LLM client are
session-pluggable.
"""
from svrptw.council.proposal import (
    OperatorContext,
    OperatorProposal,
    ProposalDecision,
    validate_proposal,
)

__all__ = [
    "OperatorContext",
    "OperatorProposal",
    "ProposalDecision",
    "validate_proposal",
]
