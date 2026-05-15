"""
Stop hook: writes a timestamped Obsidian session note into the
WorldsFinestVRP vault after every Claude session ends.

Receives JSON on stdin:
  {"session_id": "...", "transcript_path": "...", "stop_hook_active": false}
"""
import json
import os
import sys
from datetime import datetime

REPO_ROOT = "D:/SVRPTW"
VAULT_ROOT = os.path.join(REPO_ROOT, "WorldsFinestVRP")
SESSIONS_DIR = os.path.join(VAULT_ROOT, "Sessions")
MAX_SESSION_NOTES = 100  # rolling cap


def extract_summary(transcript_path: str, max_chars: int = 800) -> dict:
    """Pull the last user + assistant messages and a rough tool-use count."""
    out = {"last_user": "", "last_assistant": "", "tool_count": 0, "msg_count": 0}
    if not transcript_path or not os.path.exists(transcript_path):
        return out
    try:
        messages = []
        with open(transcript_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    messages.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        out["msg_count"] = len(messages)
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, list):
                for c in content:
                    if isinstance(c, dict) and c.get("type") == "tool_use":
                        out["tool_count"] += 1

        for msg in reversed(messages):
            role = msg.get("role", "")
            content = msg.get("content", "")
            if isinstance(content, list):
                content = " ".join(
                    c.get("text", "") for c in content if isinstance(c, dict)
                )
            content = str(content).strip()[:max_chars]
            if role == "user" and not out["last_user"]:
                out["last_user"] = content
            if role == "assistant" and not out["last_assistant"]:
                out["last_assistant"] = content
            if out["last_user"] and out["last_assistant"]:
                break
    except Exception as e:
        out["last_assistant"] = f"_Error reading transcript: {e}_"
    return out


def main():
    try:
        data = json.load(sys.stdin)
    except Exception:
        data = {}

    session_id = data.get("session_id", "unknown")[:8]
    transcript_path = data.get("transcript_path", "")

    os.makedirs(SESSIONS_DIR, exist_ok=True)

    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M")
    note_path = os.path.join(SESSIONS_DIR, f"{timestamp}-{session_id}.md")

    s = extract_summary(transcript_path)
    last_user = s["last_user"] or "_(no user message captured)_"
    last_assistant = s["last_assistant"] or "_(no assistant message captured)_"

    note = f"""---
date: {now.strftime("%Y-%m-%d")}
time: {now.strftime("%H:%M")}
session_id: {session_id}
project: SVRPTW
messages: {s["msg_count"]}
tool_calls: {s["tool_count"]}
tags: [session, svrptw, vrp, gart]
---

# Session {timestamp}

> **Stats** — {s["msg_count"]} messages, {s["tool_count"]} tool calls.

## Last user message

> {last_user[:400]}

## Last assistant message

> {last_assistant[:600]}

## Notes

<!-- Add manual notes here -->

## Links

- [[00 - Index]]
- [[05 - Decisions & Next Steps]]
- [[06 - Hooks & Automation]]
"""

    with open(note_path, "w", encoding="utf-8") as f:
        f.write(note)

    # Rolling cap — keep newest N notes, prune oldest.
    session_files = sorted(
        [f for f in os.listdir(SESSIONS_DIR) if f.endswith(".md")]
    )
    while len(session_files) > MAX_SESSION_NOTES:
        os.remove(os.path.join(SESSIONS_DIR, session_files.pop(0)))


if __name__ == "__main__":
    main()
