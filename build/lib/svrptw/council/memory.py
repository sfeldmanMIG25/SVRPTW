"""SPEC-8-COUNCIL-01 — proposal corpus (sqlite).

Every proposal — accepted or rejected — is stored with its
rationale, code, unit_test, shadow result, decision, and post-hoc
reject_reason. Future proposers query the rejection corpus to
avoid repeating dead-end families (this is what ReEvo's reflective
evolution depends on).
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from svrptw.council.proposal import OperatorProposal, ProposalDecision


_DB_PATH = Path("cache/council/corpus.sqlite")


def _ensure_db() -> sqlite3.Connection:
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(_DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS proposals (
            proposal_id          TEXT PRIMARY KEY,
            generation           INTEGER NOT NULL,
            seed_inspired_by     TEXT,
            rationale            TEXT NOT NULL,
            code                 TEXT NOT NULL,
            unit_test            TEXT NOT NULL,
            decision             TEXT NOT NULL,
            reject_reason        TEXT,
            shadow_result_json   TEXT,
            created_at           REAL NOT NULL,
            accepted_at          REAL
        )
    """)
    conn.commit()
    return conn


_lock = threading.Lock()


def record(prop: OperatorProposal, decision: ProposalDecision,
           shadow_result: Optional[dict] = None) -> None:
    """Persist a (proposal, decision, optional shadow result) triple."""
    accepted = decision.can_run_bench and (
        shadow_result is None or shadow_result.get("accepted", False)
    )
    with _lock:
        conn = _ensure_db()
        conn.execute(
            """INSERT OR REPLACE INTO proposals
               (proposal_id, generation, seed_inspired_by, rationale, code,
                unit_test, decision, reject_reason, shadow_result_json,
                created_at, accepted_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                prop.proposal_id, prop.generation, prop.seed_inspired_by,
                prop.rationale, prop.code, prop.unit_test,
                "accepted" if accepted else "rejected",
                decision.reject_reason,
                json.dumps(shadow_result) if shadow_result else None,
                time.time(),
                time.time() if accepted else None,
            ),
        )
        conn.commit()


def fetch_rejection_corpus(limit: int = 50,
                            seed_family: Optional[str] = None) -> list[dict]:
    """Return the last `limit` rejected proposals (with rationale +
    reject_reason) for the agent council to read as reflection context."""
    conn = _ensure_db()
    if seed_family:
        cur = conn.execute(
            """SELECT proposal_id, rationale, reject_reason, seed_inspired_by
               FROM proposals
               WHERE decision = 'rejected' AND seed_inspired_by = ?
               ORDER BY created_at DESC LIMIT ?""",
            (seed_family, limit),
        )
    else:
        cur = conn.execute(
            """SELECT proposal_id, rationale, reject_reason, seed_inspired_by
               FROM proposals WHERE decision = 'rejected'
               ORDER BY created_at DESC LIMIT ?""",
            (limit,),
        )
    return [
        {"proposal_id": r[0], "rationale": r[1],
         "reject_reason": r[2], "seed_inspired_by": r[3]}
        for r in cur.fetchall()
    ]


def fetch_accepted(limit: int = 20) -> list[dict]:
    """Return the most recently accepted proposals — Loop 2 / Loop 3
    consumers read this to know what's in the operator pool."""
    conn = _ensure_db()
    cur = conn.execute(
        """SELECT proposal_id, rationale, code, shadow_result_json,
                  seed_inspired_by, generation
           FROM proposals WHERE decision = 'accepted'
           ORDER BY accepted_at DESC LIMIT ?""",
        (limit,),
    )
    return [
        {"proposal_id": r[0], "rationale": r[1], "code": r[2],
         "shadow_result": json.loads(r[3]) if r[3] else None,
         "seed_inspired_by": r[4], "generation": r[5]}
        for r in cur.fetchall()
    ]


def stats() -> dict:
    """Quick corpus summary for dashboards / loop heartbeat."""
    conn = _ensure_db()
    cur = conn.execute(
        "SELECT decision, COUNT(*) FROM proposals GROUP BY decision"
    )
    counts = dict(cur.fetchall())
    return {
        "accepted": int(counts.get("accepted", 0)),
        "rejected": int(counts.get("rejected", 0)),
        "total": int(counts.get("accepted", 0) + counts.get("rejected", 0)),
    }
