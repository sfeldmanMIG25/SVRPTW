"""
SessionStart hook: surfaces the current state of the WorldsFinestVRP vault as
additionalContext at session start, so Claude opens with the latest decisions
and project state, not a blank slate.

Receives JSON on stdin:
  {"session_id": "...", "transcript_path": "...", "source": "startup|resume|clear"}

Emits JSON on stdout:
  {"hookSpecificOutput": {"hookEventName": "SessionStart", "additionalContext": "..."}}
"""
import json
import os
import sys

REPO_ROOT = "D:/SVRPTW"
VAULT_ROOT = os.path.join(REPO_ROOT, "WorldsFinestVRP")
SESSIONS_DIR = os.path.join(VAULT_ROOT, "Sessions")

# Headline docs to inline (small ones only).
INLINE_DOCS = [
    "00 - Index.md",
    "05 - Decisions & Next Steps.md",
]
INLINE_CHAR_BUDGET = 3500  # per doc


def read_capped(path: str, cap: int) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        if len(text) > cap:
            return text[:cap] + f"\n\n_…[truncated; full file is {len(text)} chars]_"
        return text
    except Exception:
        return ""


def latest_session_note() -> str:
    if not os.path.isdir(SESSIONS_DIR):
        return ""
    files = sorted(f for f in os.listdir(SESSIONS_DIR) if f.endswith(".md"))
    if not files:
        return ""
    latest = files[-1]
    return f"### Latest session: `{latest}`\n\n" + read_capped(
        os.path.join(SESSIONS_DIR, latest), 800
    )


def main():
    try:
        data = json.load(sys.stdin)
    except Exception:
        data = {}

    source = data.get("source", "startup")

    parts = [
        f"## WorldsFinestVRP — session briefing ({source})",
        "",
        "Obsidian vault lives at `D:/SVRPTW/WorldsFinestVRP/`. Process docs:",
        "`06 - Hooks & Automation.md`, `07 - Memory.md`, `08 - Context Mode.md`.",
        "",
    ]

    for name in INLINE_DOCS:
        body = read_capped(os.path.join(VAULT_ROOT, name), INLINE_CHAR_BUDGET)
        if body:
            parts.append(f"---\n### {name}\n\n{body}")

    last = latest_session_note()
    if last:
        parts.append("---\n" + last)

    additional = "\n".join(parts)

    out = {
        "hookSpecificOutput": {
            "hookEventName": "SessionStart",
            "additionalContext": additional,
        }
    }
    sys.stdout.write(json.dumps(out))


if __name__ == "__main__":
    main()
