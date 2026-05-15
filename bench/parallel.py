"""Parallel-bench helper: ProcessPoolExecutor wrapper for solver tasks.

See bench/PARALLEL_GUIDANCE.md for usage rules.
"""
from __future__ import annotations

import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Callable, Iterable


def map_instances(
    fn: Callable[[Any], dict],
    tasks: Iterable[Any],
    *,
    max_workers: int = 4,
    log_progress: bool = True,
) -> list[dict]:
    """Run `fn(task)` for each task in parallel, return ordered list of results.

    `fn` MUST be top-level (pickle-able). Lambdas and closures will fail.
    Each `task` is whatever args `fn` needs — usually a tuple.

    Results are returned in completion order (faster tasks first). Each
    result dict gets an `_elapsed_s` field appended.
    """
    tasks = list(tasks)
    total = len(tasks)
    if total == 0:
        return []

    t0 = time.perf_counter()
    results: list[dict] = []
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(fn, t): (i, t) for i, t in enumerate(tasks)}
        for fut in as_completed(futures):
            i, t = futures[fut]
            try:
                r = fut.result()
                r["_elapsed_s"] = time.perf_counter() - t0
                r["_task_index"] = i
                results.append(r)
                if log_progress:
                    sys.stdout.write(f"  [{len(results):>3}/{total}] done "
                                     f"@ {r['_elapsed_s']:>6.1f}s  task={t}\n")
                    sys.stdout.flush()
            except Exception as e:
                sys.stdout.write(f"  [task {i}] FAILED: {e}  task={t}\n")
                sys.stdout.flush()
                results.append({"_task_index": i, "_failed": True,
                                "_error": str(e), "task": str(t)})

    # Reorder to task-submission order so downstream summaries are stable.
    results.sort(key=lambda r: r.get("_task_index", 0))
    return results
