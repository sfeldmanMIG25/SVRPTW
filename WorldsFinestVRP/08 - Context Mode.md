---
title: Context Mode
project: SVRPTW
tags: [context-mode, hooks, claude-code, process, mcp]
updated: 2026-05-25
---

# Context Mode

The repo runs the **`context-mode`** Claude Code plugin ([mksglu/context-mode](https://github.com/mksglu/context-mode)) to keep raw tool output out of the conversation. The plugin sandboxes flood-prone work into a local SQLite/FTS5 store and only returns summaries to the agent. Project-level `.claude/hooks/context_guard.py` is kept as a complementary advisory nudge on Bash invocations.

## Install

From a Claude Code session in `D:/SVRPTW/`:

```text
/plugin marketplace add mksglu/context-mode
/plugin install context-mode@context-mode
```

Restart the session (or `/reload-plugins`), then verify:

```text
/context-mode:ctx-doctor
```

All checks should be `[x]` — runtimes, hooks, FTS5, MCP registration.

Prerequisite: Claude Code v1.0.33+ (`claude --version`).

## What the plugin gives you

Auto-registers `SessionStart`, `PreToolUse`, `PostToolUse`, and `PreCompact` hooks plus **11 MCP tools**:

| Tool | Purpose |
|---|---|
| `ctx_execute` | Run code in a sandbox subprocess (Python, JS, etc.). Only stdout summary lands in context. |
| `ctx_execute_file` | Same, with a file as input. |
| `ctx_batch_execute` | Run several commands at once, query the combined output. |
| `ctx_index` | Index files/output into the local FTS5 knowledge base. |
| `ctx_search` | BM25 search over indexed content. |
| `ctx_fetch_and_index` | Fetch a URL and index it. |
| `ctx_stats` | Show savings + DB stats. |
| `ctx_doctor` | Verify install. |
| `ctx_upgrade` | Update the plugin. |
| `ctx_purge` | Clear local cache/index. |
| `ctx_insight` | Open the analytics dashboard (`/ctx-insight`). |

Data lives on disk only — no telemetry, no cloud calls. See [context-mode.com](https://context-mode.com/) for the marketing pitch and screenshots.

## What's still local to this repo

`.claude/hooks/context_guard.py` still fires on every Bash invocation as an advisory nudge:

| Triggers a warning | Suggested replacement |
|---|---|
| `cat <file>` | `Read` (paginated, line-numbered) |
| `head` / `tail` | `Read` with `offset` + `limit` |
| `grep` | the `Grep` tool (ripgrep-backed) |
| `find /` | `Glob` |
| `ls -R`, `ls -la /d` | `Glob` or `mcp__Desktop_Commander__list_directory` |
| `pip list`, `conda list` | cap with `\| head -20` |

Advisory only — `exit 0` always. Safe prefixes (`git`, `python`, `mkdir`, `rm`, `mv`, `cp`, `wc -l`) bypass.

The old `UserPromptSubmit` banner that injected a "context-mode is ACTIVE" reminder every prompt is **removed** as of 2026-05-25 — the plugin's SessionStart hook does the equivalent (and now the tools it references actually exist).

## Disabling context mode

- For one turn: just tell Claude "use Read directly for this file, full output."
- For a session: `/plugin disable context-mode@context-mode`, then `/reload-plugins`.
- Permanently: `/plugin uninstall context-mode@context-mode`.

## Status line (optional)

To show live context-savings % in the status bar, add to `~/.claude/settings.json`:

```json
{
  "statusLine": {
    "type": "command",
    "command": "context-mode statusline"
  }
}
```

## See also

- [[06 - Hooks & Automation]] — the full hook registry
- [[07 - Memory]] — claude-mem (also injects at SessionStart)
- [[00 - Index]] — vault map
