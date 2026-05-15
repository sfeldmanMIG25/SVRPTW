"""
Stop hook: writes a timestamped Obsidian-compatible session note to
.claude/wiki/sessions/ after every Claude session ends.

Receives JSON on stdin:
  {"session_id": "...", "transcript_path": "...", "stop_hook_active": false}
"""
import json
import os
import sys
from datetime import datetime

REPO_ROOT = "D:/SVRPTW"
SESSIONS_DIR = os.path.join(REPO_ROOT, ".claude", "wiki", "sessions")


def extract_summary(transcript_path: str, max_chars: int = 800) -> str:
    if not transcript_path or not os.path.exists(transcript_path):
        return "_No transcript available._"
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

        last_user = last_assistant = ""
        for msg in reversed(messages):
            role = msg.get("role", "")
            content = msg.get("content", "")
            if isinstance(content, list):
                content = " ".join(
                    c.get("text", "") for c in content if isinstance(c, dict)
                )
            content = str(content).strip()[:max_chars]
            if role == "user" and not last_user:
                last_user = content
            if role == "assistant" and not last_assistant:
                last_assistant = content
            if last_user and last_assistant:
                break

        parts = []
        if last_user:
            parts.append(f"**Last user message:**\n> {last_user[:300]}")
        if last_assistant:
            parts.append(f"**Last assistant message:**\n> {last_assistant[:400]}")
        return "\n\n".join(parts) if parts else "_Empty transcript._"
    except Exception as e:
        return f"_Error reading transcript: {e}_"


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

    summary = extract_summary(transcript_path)

    note = f"""---
date: {now.strftime("%Y-%m-%d")}
time: {now.strftime("%H:%M")}
session_id: {session_id}
project: SVRPTW
tags: [session, svrptw, vrp, gart]
---

# Session {timestamp}

## Transcript Summary

{summary}

## Notes

<!-- Add manual notes here -->

## Links
- [[project-overview]]
- [[specs]]
- [[gart-integration]]
"""

    with open(note_path, "w", encoding="utf-8") as f:
        f.write(note)

    session_files = sorted(
        [f for f in os.listdir(SESSIONS_DIR) if f.endswith(".md")]
    )
    while len(session_files) > 50:
        os.remove(os.path.join(SESSIONS_DIR, session_files.pop(0)))


if __name__ == "__main__":
    main()
