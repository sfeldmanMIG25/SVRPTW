---
title: Context Mode
project: SVRPTW
tags: [context-mode, hooks, claude-code, process]
updated: 2026-05-14
---

# Context Mode

A convention this repo uses to keep Claude from blowing its context window on raw tool output. Two layers: a **prompt-level banner** that reminds Claude of the rules, and a **PreToolUse warning** that flags flood-prone Bash commands.

## The banner (UserPromptSubmit)

Every user prompt has this string injected as `additionalContext`:

> context-mode is ACTIVE. For any command with >20 lines of output, file analysis, or multi-step research use `ctx_batch_execute`/`ctx_search`/`ctx_execute` instead of Bash/Read. If `graphify-out/GRAPH_REPORT.md` exists, read it before answering architecture questions. Raw tool output floods context — keep large data in sandbox, return summaries only.

The injection is wired in `.claude/settings.json` under `hooks.UserPromptSubmit`:

```json
{
  "type": "command",
  "command": "echo '{\"hookSpecificOutput\":{\"hookEventName\":\"UserPromptSubmit\",\"additionalContext\":\"context-mode is ACTIVE…\"}}'"
}
```

This fires **by default on every prompt** — no opt-in needed. The banner travels with the repo via `.claude/settings.json` (project-level), so anyone cloning the repo gets the same behavior.

## The flood guard (PreToolUse → Bash)

`.claude/hooks/context_guard.py` watches every Bash invocation and prints a stderr warning when the command matches a flood pattern:

| Triggers a warning | Reason |
|---|---|
| `cat <file>` | Use `Read` (paginated, line-numbered) |
| `head` / `tail` | Use `Read` with `offset` + `limit` |
| `grep` | Use the `Grep` tool (ripgrep-backed, better output modes) |
| `find /` | Use `Glob` |
| `ls -R`, `ls -la /d` | Use `Glob` or `mcp__Desktop_Commander__list_directory` |
| `pip list`, `conda list` | Cap with `\| head -20` |

The hook is **advisory only** — `exit 0` always. It nudges, never blocks. Safe prefixes (`git`, `python`, `mkdir`, `rm`, `mv`, `cp`, `wc -l`) bypass the check entirely.

## When you want context mode OFF

If you're running a one-off investigation where you actually do want raw output (e.g., reading a long log in full), you can:

1. Acknowledge the banner in your prompt: "Ignore context-mode for this turn, I want the full file."
2. Or temporarily comment out the `UserPromptSubmit` block in `.claude/settings.json` for the session, then revert.

Claude treats the banner as a strong default, not an unbreakable rule — explicit instructions override it.

## When context mode bites you

Symptom: Claude refuses to use plain `Read`/`Bash` and keeps reaching for `ctx_*` tools that aren't available. Cause: the banner is on but the `ctx_*` MCP server isn't actually connected.

Fix: either install the ctx server (out of scope here) or remove the `UserPromptSubmit` block. The banner *references* `ctx_*` tools but doesn't enforce them — Claude will still fall back to `Read`/`Grep`/`Glob` if `ctx_*` is unavailable.

## See also

- [[06 - Hooks & Automation]] — the full hook registry
- [[07 - Memory]] — claude-mem (also injects at SessionStart)
- [[00 - Index]] — vault map
