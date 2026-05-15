"""Entry point: `python -m svrptw <subcommand>`."""
from __future__ import annotations

import sys


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: python -m svrptw {bench|gen|solve|viz|vivrp}", file=sys.stderr)
        return 1
    cmd = sys.argv.pop(1)
    if cmd == "bench":
        from svrptw.bench.harness import main as bench_main
        return bench_main()
    if cmd == "gen":
        from svrptw.instances_gen.osmnx_generator import main as gen_main
        return gen_main()
    if cmd == "vivrp":
        from svrptw.vivrp.assessor import _cli
        return _cli()
    if cmd == "viz":
        from svrptw.viz import renderer
        return renderer.__main__() if hasattr(renderer, "__main__") else 0  # type: ignore
    if cmd == "solve":
        sub = sys.argv.pop(1) if len(sys.argv) > 1 else "greedy"
        if sub == "greedy":
            from svrptw.solvers.classical.greedy import __main__ as _; return 0  # noqa
        # Defer to module CLI
        sys.argv = [sub] + sys.argv[1:]
    print(f"unknown subcommand: {cmd}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
