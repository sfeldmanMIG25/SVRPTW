"""Time-lapse generator: snapshot sequence -> animated GIF.

Bench scripts dump per-iteration snapshots into webui/static/snapshots/
(see ``webui.snapshot.push_solution`` and ``make_on_accept``). After a
solve completes, you can ask for a single GIF that walks through the
operator's accepted moves.

Implementation: matplotlib + PIL only (PIL is already a transitive dep
via matplotlib). No new dependencies.

Usage::

    from webui.timelapse import build_timelapse
    out = build_timelapse(stream="warm-N200-OSM-Manhattan-N200-I003",
                          out_path="webui/static/timelapses/warm_man200.gif",
                          fps=4)
    print("wrote", out)
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

_LOG = logging.getLogger("webui.timelapse")
_HERE = Path(__file__).resolve().parent
_SNAPSHOT_ROOT = _HERE / "static" / "snapshots"
_TIMELAPSE_ROOT = _HERE / "static" / "timelapses"


def _frames_for_stream(stream: str,
                        snapshots_dir: Path = _SNAPSHOT_ROOT) -> list[Path]:
    """Find PNGs whose filename starts with the slugified stream name.

    See ``webui.snapshot._slugify`` -- streams are slugified into the
    filename prefix ``<slug>__...png`` by ``push_solution``.
    """
    if not snapshots_dir.exists():
        return []
    keep = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_."
    slug = "".join(c if c in keep else "_" for c in stream)[:80]
    matches = sorted(snapshots_dir.glob(f"{slug}__*.png"))
    return matches


def list_streams(snapshots_dir: Path = _SNAPSHOT_ROOT) -> list[str]:
    """Return the unique stream slugs currently present on disk."""
    if not snapshots_dir.exists():
        return []
    streams: set[str] = set()
    for p in snapshots_dir.glob("*__*.png"):
        slug, _, _ = p.name.partition("__")
        streams.add(slug)
    return sorted(streams)


def build_timelapse(
    stream: str,
    *,
    out_path: str | Path | None = None,
    fps: int = 4,
    max_frames: int = 200,
    loop: int = 0,
) -> Path | None:
    """Build an animated GIF from snapshots in `stream`.

    Returns the output path, or None if no frames were found.

    Frames are loaded in lexicographic order (== timestamp order, since
    snapshot.py prefixes filenames with epoch_ms). PIL is used directly
    because matplotlib.animation forces an ImageMagick or ffmpeg dep.
    """
    frames = _frames_for_stream(stream)
    if not frames:
        _LOG.warning("no frames for stream=%s under %s", stream, _SNAPSHOT_ROOT)
        return None
    if max_frames and len(frames) > max_frames:
        # Subsample evenly so we keep first + last + spaced middle.
        step = len(frames) / max_frames
        frames = [frames[int(i * step)] for i in range(max_frames)]
    if out_path is None:
        _TIMELAPSE_ROOT.mkdir(parents=True, exist_ok=True)
        out_path = _TIMELAPSE_ROOT / f"{stream}.gif"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        from PIL import Image
    except ImportError:
        _LOG.error("Pillow not installed; cannot build GIF")
        return None

    imgs = [Image.open(p).convert("P", palette=Image.ADAPTIVE) for p in frames]
    duration_ms = max(50, int(1000 / max(1, fps)))
    imgs[0].save(out_path, save_all=True, append_images=imgs[1:],
                  duration=duration_ms, loop=loop, optimize=True)
    return out_path


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Build animated GIFs from snapshots")
    p.add_argument("--stream",
                    help="Stream slug to render. Pass --list to see available.")
    p.add_argument("--list", action="store_true",
                    help="List available stream slugs and exit")
    p.add_argument("--out", help="Output GIF path (default: static/timelapses/<stream>.gif)")
    p.add_argument("--fps", type=int, default=4)
    p.add_argument("--max-frames", type=int, default=200)
    args = p.parse_args()

    if args.list:
        streams = list_streams()
        if not streams:
            print(f"(no snapshots under {_SNAPSHOT_ROOT})")
        else:
            for s in streams:
                n_frames = len(_frames_for_stream(s))
                print(f"  {s}  ({n_frames} frames)")
        raise SystemExit(0)

    if not args.stream:
        p.error("--stream STREAM required (or pass --list)")

    out = build_timelapse(args.stream, out_path=args.out,
                          fps=args.fps, max_frames=args.max_frames)
    if out is None:
        print(f"no frames for stream {args.stream!r}")
        raise SystemExit(1)
    print(f"wrote {out}")
