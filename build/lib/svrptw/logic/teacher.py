"""Gemini-3.1-Flash-Lite teacher wrapping the ViVRP backend (SPEC-6-LOGIC-01).

Returns a (score, rationale, confidence) triple cached on disk by
(instance_id, solution_hash). The teacher is a fixed oracle — we never
retrain it. Latency 5–10 s/call; cache hit ≤ 5 ms.

This module is the *entry point* for the logic axis. The student
(`svrptw.logic.student`) is what the inner loop will call; the teacher
is only used offline to label preference pairs.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from svrptw.io.instance import Instance
    from svrptw.solvers.common.solution import Solution


CACHE_DIR = Path("cache/logic")
CACHE_DB = CACHE_DIR / "teacher.sqlite"

# Gemini Flash Lite quota: 15 RPM / 500 RPD. We throttle to 12 RPM
# locally to leave headroom for parallel bench jobs sharing the key.
_MIN_CALL_INTERVAL_S = 5.0


@dataclass
class TeacherLabel:
    """One teacher judgment on a (instance, solution) pair.

    score in [0,1] is P(dispatcher ships it). `None` means the teacher
    refused or errored — search must treat this as 'no logic signal'.
    """

    score: Optional[float]
    rationale: str
    confidence: float
    basemap_path: Optional[Path]
    cached: bool


def solution_hash(solution: "Solution") -> str:
    """Stable hash of the route sequences.

    Independent of metrics/costs/ordering-of-vehicles. Used as the
    cache key alongside instance_id.
    """
    canon = sorted(
        tuple(int(c) for c in r.customers)
        for r in solution.routes
        if r.customers
    )
    blob = json.dumps(canon, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[:16]


def _ensure_cache() -> sqlite3.Connection:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(CACHE_DB)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS labels (
              instance_id   TEXT NOT NULL,
              solution_hash TEXT NOT NULL,
              score         REAL,
              rationale     TEXT NOT NULL,
              confidence    REAL NOT NULL,
              basemap_path  TEXT,
              created_at    REAL NOT NULL,
              PRIMARY KEY (instance_id, solution_hash)
           )"""
    )
    conn.commit()
    return conn


class GeminiTeacher:
    """Thin wrapper over `svrptw.vivrp.assessor._GeminiBackend` with caching.

    The actual `score()` implementation defers to the ViVRP backend's
    structured-output path; this class only adds (a) a dispatcher-focused
    rubric, (b) on-disk caching, (c) RPM throttling, (d) graceful
    failure (returns score=None on Gemini errors).

    Stubbed: `score()` raises NotImplementedError until the in-flight
    POMO/bench jobs free up Gemini quota. The cache layer is fully
    functional today so we can land tests around the key schema.
    """

    _call_lock = threading.Lock()
    _last_call_t = 0.0

    def __init__(self, rubric_path: Optional[Path] = None) -> None:
        self._rubric_path = rubric_path or Path("svrptw/logic/rubric_dispatcher.md")
        self._conn = _ensure_cache()

    def _cache_get(self, instance_id: str, sol_hash: str) -> Optional[TeacherLabel]:
        cur = self._conn.execute(
            "SELECT score, rationale, confidence, basemap_path FROM labels "
            "WHERE instance_id=? AND solution_hash=?",
            (instance_id, sol_hash),
        )
        row = cur.fetchone()
        if row is None:
            return None
        score, rationale, conf, bmap = row
        return TeacherLabel(
            score=score,
            rationale=rationale,
            confidence=conf,
            basemap_path=Path(bmap) if bmap else None,
            cached=True,
        )

    def _cache_put(self, instance_id: str, sol_hash: str, label: TeacherLabel) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO labels VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                instance_id,
                sol_hash,
                label.score,
                label.rationale,
                label.confidence,
                str(label.basemap_path) if label.basemap_path else None,
                time.time(),
            ),
        )
        self._conn.commit()

    def _throttle(self) -> None:
        # Serialise calls across threads; enforce min-interval to stay
        # under the 15 RPM Gemini Flash Lite quota.
        with GeminiTeacher._call_lock:
            dt = time.time() - GeminiTeacher._last_call_t
            if dt < _MIN_CALL_INTERVAL_S:
                time.sleep(_MIN_CALL_INTERVAL_S - dt)
            GeminiTeacher._last_call_t = time.time()

    def score(self, instance: "Instance", solution: "Solution") -> TeacherLabel:
        """Score one (instance, solution) pair. Cached.

        Cache hit (precomputed elsewhere) returns immediately. Online
        labeling goes through the OpenRouter committee (SPEC-6-LOGIC-02);
        gated on no in-flight bench job competing for bandwidth.
        """
        sh = solution_hash(solution)
        hit = self._cache_get(instance.instance_id, sh)
        if hit is not None:
            return hit
        # Online path delegates to the committee — but only when callers
        # explicitly opt in via OpenRouterCommitteeTeacher.
        raise NotImplementedError(
            "GeminiTeacher.score(): use OpenRouterCommitteeTeacher for "
            "online labeling (SPEC-6-LOGIC-02). This class only serves "
            "cached labels for backwards compatibility."
        )


class OpenRouterCommitteeTeacher:
    """Online labeler backed by the SPEC-6-LOGIC-02 multi-model committee.

    Caches per (instance_id, solution_hash) → CommitteeLabel.to_dict()
    in the same sqlite the legacy Gemini teacher uses; downstream
    consumers can choose whether to read consensus-median (`score`) or
    a per-model vote.
    """

    def __init__(self) -> None:
        from svrptw.logic.committee import OpenRouterCommittee  # local import: heavy deps
        self._committee = OpenRouterCommittee()
        self._conn = _ensure_cache()
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS committee_labels (
                  instance_id   TEXT NOT NULL,
                  solution_hash TEXT NOT NULL,
                  payload_json  TEXT NOT NULL,
                  created_at    REAL NOT NULL,
                  PRIMARY KEY (instance_id, solution_hash)
               )"""
        )
        self._conn.commit()

    def cached(self, instance_id: str, sol_hash: str) -> Optional[dict]:
        cur = self._conn.execute(
            "SELECT payload_json FROM committee_labels "
            "WHERE instance_id=? AND solution_hash=?",
            (instance_id, sol_hash),
        )
        row = cur.fetchone()
        return json.loads(row[0]) if row else None

    def label(
        self,
        instance: "Instance",
        solution: "Solution",
        prompt: str,
        image_path: Optional[Path],
    ) -> dict:
        sh = solution_hash(solution)
        hit = self.cached(instance.instance_id, sh)
        if hit is not None:
            return hit
        result = self._committee.label(instance, solution, prompt, image_path)
        payload = result.to_dict()
        self._conn.execute(
            "INSERT OR REPLACE INTO committee_labels VALUES (?, ?, ?, ?)",
            (instance.instance_id, sh, json.dumps(payload), time.time()),
        )
        self._conn.commit()
        return payload
