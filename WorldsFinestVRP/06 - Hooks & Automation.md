---
title: Hooks & Automation
project: SVRPTW
tags: [hooks, automation, claude-code, process]
updated: 2026-05-14
---

# Hooks & Automation

How Claude Code is wired into this repo. **Read this first** if a hook is misbehaving — it's the map.

All hooks are configured in `.claude/settings.json` at the project root. Hook scripts live in `.claude/hooks/`. Settings is checked into the repo, so the wiring travels with the project.

## What fires when

| Event | Script | What it does |
|---|---|---|
| `UserPromptSubmit` | inline `echo` | Injects the **context-mode** banner into every user prompt (see [[08 - Context Mode]]). |
| `PreToolUse` (Bash) | `context_guard.py` | Warns on stderr when a Bash command is likely to flood context (`cat`, `head`, big `grep`). Advisory only — never blocks. |
| `Stop` | `obsidian_log.py` | Writes a timestamped session note into `Sessions/`. Captures last user + assistant messages, message count, tool-call count. Rolling cap of 100 notes. |
| `Stop` | graphify rebuild | Refreshes the `graphify-out/` knowledge graph. Silent failure (`|| true`). |
| `SessionStart` | `session_briefing.py` | Reads `00 - Index.md`, `05 - Decisions & Next Steps.md`, and the latest session note. Emits them as `additionalContext` so a fresh session opens already knowing where the project is. |

## Files at a glance

```
.claude/
├── settings.json           # the hook registry (checked in)
└── hooks/
    ├── context_guard.py    # PreToolUse: flood-warning for Bash
    ├── obsidian_log.py     # Stop: writes Sessions/<timestamp>-<id>.md
    └── session_briefing.py # SessionStart: injects vault state
```

## Interaction with global hooks

Global hooks live in `~/.claude/settings.json` and stack with these. After `claude-mem` is installed (see [[07 - Memory]]), it adds its own `SessionStart` hook that injects a transcript-derived memory summary. Both `SessionStart` hooks fire — claude-mem brings the *narrative* of what happened last time, `session_briefing.py` brings the *current state* of the vault. They complement.

## Editing a hook safely

1. Edit the `.py` under `.claude/hooks/`.
2. To test the Stop hook locally:
   ```powershell
   '{"session_id":"test","transcript_path":""}' | python .claude/hooks/obsidian_log.py
   ```
   A note should appear under `WorldsFinestVRP/Sessions/`.
3. To test SessionStart:
   ```powershell
   '{"session_id":"test","source":"startup"}' | python .claude/hooks/session_briefing.py
   ```
   Output should be valid JSON with `hookSpecificOutput.additionalContext`.

## Disabling temporarily

Comment out the offending block in `.claude/settings.json`. **Do not delete** — keep the wiring visible.

## See also

- [[07 - Memory]] — claude-mem plugin (persistent SessionStart memory)
- [[08 - Context Mode]] — context-mode injection + flood-warning conventions
- [[00 - Index]] — vault map
