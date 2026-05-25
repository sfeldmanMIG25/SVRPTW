"""Operator-annotated gap visualization (user-requested 2026-05-16).

Take a bandit_transitions.jsonl (from `collect_bandit_logs.py` or any solve
with `bandit_kind='logging'`) and render a single PNG per instance showing:

  Top panel:    cumulative cost decrease over wall time, with each accepted
                bandit step shown as an op-colored marker. Reviewer can see
                "which moves moved the needle, and where they landed."

  Bottom panel: per-operator contribution bar chart -- which ops actually
                provided the bulk of the cost decrease vs which were noise.

This is the "gap line plot with which operator got where" visualization
requested for figuring out what's working and what isn't.

Usage:
    python bench/scripts/build_operator_gap_plot.py \\
        --transitions bench/runs/bandit_transitions.jsonl \\
        --instance OSM-Manhattan-N100-I003.json \\
        --out bench/figures/operator_gap_manhattan100.png

Defaults: if --instance is omitted, renders one PNG per distinct instance
in the transitions file under bench/figures/op_gap_<instance>.png.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

# Headless backend for CI
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _load_transitions(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _per_instance(transitions: list[dict]) -> dict[str, list[dict]]:
    by_inst: dict[str, list[dict]] = defaultdict(list)
    for t in transitions:
        inst = t.get("instance", "<unknown>")
        by_inst[inst].append(t)
    for inst in by_inst:
        by_inst[inst].sort(key=lambda r: r["ts"])
    return dict(by_inst)


def _operator_palette(ops: list[str]) -> dict[str, tuple]:
    """Stable colour per operator (tab20 + secondary). Falls back to gray
    for unrecognized ops so the palette stays consistent across plots."""
    canon = [
        "merge_routes", "relocate", "two_opt_intra", "two_opt_star",
        "swap_star", "three_opt_intra", "sisr", "ejection_chain",
        "cyclic_3", "vehicle_kill", "soft_drop", "drop_route",
        "destroy_island", "drop_leg", "shift_start", "class_shift",
        "depot_shift",
    ]
    cmap = plt.get_cmap("tab20")
    palette = {op: cmap(i % 20) for i, op in enumerate(canon)}
    for op in ops:
        palette.setdefault(op, (0.6, 0.6, 0.6, 1.0))
    return palette


def _prep_series(rows: list[dict]) -> tuple:
    """Pre-compute wall_s, deltas, cum, ops, palette for a row list."""
    t0 = rows[0]["ts"]
    elapsed = [0.0]
    for i in range(1, len(rows)):
        elapsed.append(rows[i]["ts"] - rows[i - 1]["ts"])
    if len(elapsed) > 1:
        elapsed[0] = float(np.median([e for e in elapsed[1:] if e > 0]) or 0.1)
    deltas = np.array([r["reward"] * e for r, e in zip(rows, elapsed)])
    wall_s = np.array([r["ts"] - t0 for r in rows])
    cum = np.cumsum(deltas)
    ops = [r["op"] for r in rows]
    distinct_ops = sorted(set(ops))
    palette = _operator_palette(distinct_ops)
    return wall_s, deltas, cum, ops, distinct_ops, palette


def _render_frame(
    wall_s: np.ndarray, deltas: np.ndarray, cum: np.ndarray,
    ops: list[str], distinct_ops: list[str], palette: dict,
    *, n: int, instance: str, total_steps: int,
    figsize: tuple[float, float] = (12, 8),
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
) -> "plt.Figure":
    """Render the first `n` transitions as a single frame figure."""
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=figsize, gridspec_kw={"height_ratios": [2.0, 1.0]},
    )
    if n > 0:
        ax_top.plot(wall_s[:n], cum[:n], color="black", lw=1.1, alpha=0.5, zorder=1)
        max_abs = max(1.0, float(np.max(np.abs(deltas[:n]))) if n else 1.0)
        for op in distinct_ops:
            idx = [i for i in range(n) if ops[i] == op]
            if not idx:
                continue
            sizes = 10 + 90 * np.tanh(np.abs(deltas[idx]) / max_abs)
            ax_top.scatter(
                wall_s[idx], cum[idx], s=sizes, c=[palette[op]], label=op,
                edgecolors="white", linewidths=0.4, zorder=2,
            )
    ax_top.set_xlabel("wall time (s)")
    ax_top.set_ylabel("cumulative cost decrease ($)")
    if xlim: ax_top.set_xlim(*xlim)
    if ylim: ax_top.set_ylim(*ylim)
    cur_delta = float(cum[n - 1]) if n > 0 else 0.0
    ax_top.set_title(
        f"Operator-annotated cost trajectory  ({instance})\n"
        f"step {n}/{total_steps}  -  cumulative delta=${cur_delta:+.1f}",
        fontsize=10,
    )
    ax_top.grid(True, alpha=0.3)
    if n > 0:
        ax_top.legend(
            loc="best", fontsize=7, ncol=2,
            markerscale=0.7, framealpha=0.9, columnspacing=0.6,
        )

    # Per-operator contribution bar chart (up to step n)
    contrib = defaultdict(float)
    pulls = defaultdict(int)
    for i in range(n):
        contrib[ops[i]] += float(deltas[i])
        pulls[ops[i]] += 1
    op_order = sorted(distinct_ops, key=lambda o: contrib[o], reverse=True)
    ys = np.arange(len(op_order))
    vals = [contrib[op] for op in op_order]
    colors = [palette[op] for op in op_order]
    bars = ax_bot.barh(ys, vals, color=colors, edgecolor="white", linewidth=0.5)
    ax_bot.set_yticks(ys)
    ax_bot.set_yticklabels([f"{o}  (n={pulls[o]})" for o in op_order], fontsize=8)
    ax_bot.invert_yaxis()
    ax_bot.set_xlabel("total cost decrease attributed ($)")
    ax_bot.axvline(0, color="black", lw=0.6)
    ax_bot.set_title("per-operator contribution (descending)", fontsize=10)
    for op, v, bar in zip(op_order[:3], vals[:3], bars[:3]):
        if v == 0:
            continue
        ax_bot.text(
            v, bar.get_y() + bar.get_height() / 2,
            f"  ${v:+.0f}",
            va="center", ha="left" if v >= 0 else "right",
            fontsize=8, fontweight="bold",
        )
    ax_bot.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    return fig


def render_animation(
    rows: list[dict],
    *,
    instance: str,
    out_path: Path,
    fps: int = 6,
    max_frames: int = 80,
    figsize: tuple[float, float] = (12, 8),
) -> None:
    """Render a GIF that grows the operator-gap plot step by step."""
    if not rows:
        print(f"  ! no rows for {instance}", file=sys.stderr)
        return
    try:
        from PIL import Image
    except ImportError:
        print("  ! Pillow not installed; cannot build animation GIF",
              file=sys.stderr)
        return

    wall_s, deltas, cum, ops, distinct_ops, palette = _prep_series(rows)
    # Fix axis limits using final frame so the animation doesn't jitter
    xlim = (-0.02 * wall_s[-1], 1.02 * wall_s[-1])
    y_min = float(min(0.0, cum.min())) - 0.05 * max(1.0, abs(float(cum.max())))
    y_max = float(cum.max()) * 1.05
    ylim = (y_min, y_max)

    n_total = len(rows)
    n_frames = min(max_frames, n_total)
    step = max(1, n_total // n_frames)
    frame_steps = list(range(step, n_total + 1, step))
    if frame_steps[-1] != n_total:
        frame_steps.append(n_total)

    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        frame_paths: list[Path] = []
        for fi, n in enumerate(frame_steps):
            fig = _render_frame(
                wall_s, deltas, cum, ops, distinct_ops, palette,
                n=n, instance=instance, total_steps=n_total,
                figsize=figsize, xlim=xlim, ylim=ylim,
            )
            p = tmp / f"frame_{fi:04d}.png"
            fig.savefig(p, dpi=100)
            plt.close(fig)
            frame_paths.append(p)
        # Hold the last frame for a beat (4x duration)
        imgs = [Image.open(p).convert("P", palette=Image.ADAPTIVE)
                for p in frame_paths]
        for _ in range(3):
            imgs.append(imgs[-1])
        duration_ms = max(50, int(1000 / max(1, fps)))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        imgs[0].save(
            out_path, save_all=True, append_images=imgs[1:],
            duration=duration_ms, loop=0, optimize=True,
        )
    print(f"  wrote animation {out_path}  "
          f"({len(frame_paths)} frames, {n_total} bandit steps)")


def render_instance(
    rows: list[dict],
    *,
    instance: str,
    out_path: Path,
    figsize: tuple[float, float] = (12, 8),
) -> None:
    """Render one operator-gap figure for a single instance's transitions."""
    if not rows:
        print(f"  ! no rows for {instance}", file=sys.stderr)
        return

    # Reconstruct per-iter cost decrease from raw reward * elapsed.
    t0 = rows[0]["ts"]
    elapsed = [0.0]
    for i in range(1, len(rows)):
        elapsed.append(rows[i]["ts"] - rows[i - 1]["ts"])
    # First row's "elapsed" is unknown; approximate as median of remainder.
    if len(elapsed) > 1:
        elapsed[0] = float(np.median([e for e in elapsed[1:] if e > 0]) or 0.1)
    # cost_decrease per row = reward * elapsed (reward is per-second).
    deltas = np.array([r["reward"] * e for r, e in zip(rows, elapsed)])
    wall_s = np.array([r["ts"] - t0 for r in rows])
    cum = np.cumsum(deltas)
    ops = [r["op"] for r in rows]
    distinct_ops = sorted(set(ops))
    palette = _operator_palette(distinct_ops)

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=figsize, gridspec_kw={"height_ratios": [2.0, 1.0]},
    )

    # --- Top panel: cumulative cost decrease vs wall time ---
    ax_top.plot(wall_s, cum, color="black", lw=1.1, alpha=0.5, zorder=1)
    # Per-op markers (size by |reward|, color by op)
    for op in distinct_ops:
        idx = [i for i, o in enumerate(ops) if o == op]
        if not idx:
            continue
        sizes = 10 + 90 * np.tanh(np.abs(deltas[idx]) / max(1.0, np.max(np.abs(deltas))))
        ax_top.scatter(
            wall_s[idx], cum[idx],
            s=sizes, c=[palette[op]], label=op,
            edgecolors="white", linewidths=0.4, zorder=2,
        )
    ax_top.set_xlabel("wall time (s)")
    ax_top.set_ylabel("cumulative cost decrease ($)")
    ax_top.set_title(
        f"Operator-annotated cost trajectory  ({instance}, "
        f"{len(rows)} bandit steps, total delta=${cum[-1]:+.1f})",
        fontsize=11,
    )
    ax_top.grid(True, alpha=0.3)
    ax_top.legend(
        loc="best", fontsize=7, ncol=2,
        markerscale=0.7, framealpha=0.9, columnspacing=0.6,
    )

    # --- Bottom panel: total per-operator contribution ---
    contrib = defaultdict(float)
    pulls = defaultdict(int)
    for r, d in zip(rows, deltas):
        contrib[r["op"]] += float(d)
        pulls[r["op"]] += 1
    op_order = sorted(distinct_ops, key=lambda o: contrib[o], reverse=True)
    ys = np.arange(len(op_order))
    vals = [contrib[op] for op in op_order]
    colors = [palette[op] for op in op_order]
    bars = ax_bot.barh(ys, vals, color=colors, edgecolor="white", linewidth=0.5)
    ax_bot.set_yticks(ys)
    ax_bot.set_yticklabels([f"{o}  (n={pulls[o]})" for o in op_order], fontsize=8)
    ax_bot.invert_yaxis()
    ax_bot.set_xlabel("total cost decrease attributed ($)")
    ax_bot.axvline(0, color="black", lw=0.6)
    ax_bot.set_title("per-operator contribution (descending)", fontsize=10)
    # Annotate top 3 with value
    for op, v, bar in zip(op_order[:3], vals[:3], bars[:3]):
        ax_bot.text(
            v, bar.get_y() + bar.get_height() / 2,
            f"  ${v:+.0f}",
            va="center", ha="left" if v >= 0 else "right",
            fontsize=8, fontweight="bold",
        )
    ax_bot.grid(True, axis="x", alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"  wrote {out_path}  (total delta=${cum[-1]:+.1f}, {len(distinct_ops)} ops)")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--transitions", required=True,
                   help="Path to bandit_transitions.jsonl")
    p.add_argument("--instance", default=None,
                   help="Single instance filename to render. If omitted, "
                        "renders one PNG per distinct instance.")
    p.add_argument("--out", default=None,
                   help="Output PNG path (single-instance mode) OR output "
                        "directory (multi-instance mode). Defaults to "
                        "bench/figures/op_gap_<instance>.png")
    p.add_argument("--animate", action="store_true",
                   help="Produce an animated .gif growing the plot step by "
                        "step instead of a static PNG.")
    p.add_argument("--fps", type=int, default=6,
                   help="Frames per second for --animate (default 6).")
    p.add_argument("--max-frames", type=int, default=80,
                   help="Max frames per --animate GIF (subsamples large "
                        "trajectories).")
    args = p.parse_args()

    tx_path = Path(args.transitions)
    if not tx_path.exists():
        print(f"transitions file not found: {tx_path}", file=sys.stderr)
        return 2

    rows = _load_transitions(tx_path)
    print(f"loaded {len(rows)} transitions from {tx_path}")
    by_inst = _per_instance(rows)
    print(f"  -> {len(by_inst)} distinct instances")

    ext = ".gif" if args.animate else ".png"
    fn_prefix = "op_gap_anim_" if args.animate else "op_gap_"

    def _do_one(rows, inst_name, out_path):
        if args.animate:
            render_animation(rows, instance=inst_name, out_path=out_path,
                             fps=args.fps, max_frames=args.max_frames)
        else:
            render_instance(rows, instance=inst_name, out_path=out_path)

    if args.instance:
        if args.instance not in by_inst:
            avail = list(by_inst.keys())[:10]
            print(f"instance {args.instance!r} not in transitions. "
                  f"Available (first 10): {avail}", file=sys.stderr)
            return 3
        slug = args.instance.replace(".json", "")
        out_path = Path(args.out or f"bench/figures/{fn_prefix}{slug}{ext}")
        if out_path.is_dir():
            out_path = out_path / f"{fn_prefix}{slug}{ext}"
        _do_one(by_inst[args.instance], args.instance, out_path)
    else:
        out_dir = Path(args.out or "bench/figures") if args.out else Path("bench/figures")
        out_dir.mkdir(parents=True, exist_ok=True)
        for inst, irows in by_inst.items():
            slug = inst.replace(".json", "")
            _do_one(irows, inst, out_dir / f"{fn_prefix}{slug}{ext}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
