"""Probe Gemini API for available vision models + re-render anon pair."""
from __future__ import annotations
import os
import sys
from pathlib import Path

sys.path.insert(0, "D:/SVRPTW")
# load .env so SVRPTW_GEMINI_API_KEY is in env
_env = Path("D:/SVRPTW/.env")
if _env.exists():
    for line in _env.read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())
os.environ.setdefault("SVRPTW_WEBUI_URL", "http://127.0.0.1:8765")


def probe_gemini() -> None:
    print("=== Gemini API probe ===")
    key = (os.environ.get("SVRPTW_GEMINI_API_KEY")
           or os.environ.get("GEMINI_API_KEY"))
    if not key:
        print("  no GEMINI key in env")
        return
    print(f"  key set (len {len(key)})")
    try:
        import google.generativeai as genai
    except ImportError:
        print("  google-generativeai not installed")
        return
    try:
        genai.configure(api_key=key)
        models = list(genai.list_models())
        print(f"  found {len(models)} models")
        # show vision-capable ones (those that support generateContent + image input)
        vision_models = []
        for m in models:
            methods = list(getattr(m, "supported_generation_methods", []) or [])
            if "generateContent" in methods:
                vision_models.append(m)
        print(f"  generateContent-capable: {len(vision_models)}")
        for m in vision_models:
            tag = ""
            name = getattr(m, "name", "?")
            if "vision" in name.lower() or "flash" in name.lower() or "pro" in name.lower():
                tag = "  <-- likely vision-capable"
            print(f"    - {name}{tag}")
    except Exception as e:
        print(f"  probe failed: {type(e).__name__}: {e}")


def rerender() -> None:
    from svrptw.io import load_instance
    from svrptw.config import Settings
    from svrptw.solvers.classical.portfolio_pyvrp_warm import solve_auto
    from svrptw.solvers.classical import pyvrp_solver as pv
    from svrptw.viz.renderer import render_llm_compare

    print()
    print("=== rerender anon pair (vehicle-id colors) ===")
    inst = load_instance("instances/v1/OSM-Manhattan-N050-I003.json")
    warm = solve_auto(inst, Settings(), budget_seconds=4)
    pyvrp = pv.solve(inst, Settings(), budget_seconds=10)
    out_dir = Path("D:/SVRPTW/webui/static/snapshots/consistency/anon")
    out_dir.mkdir(parents=True, exist_ok=True)
    render_llm_compare(inst, warm, out_dir / "anon__warm.png", overlay_mode="minimal")
    render_llm_compare(inst, pyvrp, out_dir / "anon__pyvrp.png", overlay_mode="minimal")
    print(f"  warm K={int(warm.metrics['num_vehicles_used'])}, pyvrp K={int(pyvrp.metrics['num_vehicles_used'])}")
    print("  vehicle 0 in BOTH gets color slot 0; vehicle N in BOTH gets slot N.")


if __name__ == "__main__":
    probe_gemini()
    rerender()
