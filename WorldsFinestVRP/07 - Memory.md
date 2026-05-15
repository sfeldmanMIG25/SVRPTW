---
title: Memory (claude-mem)
project: SVRPTW
tags: [memory, claude-mem, plugin, process]
updated: 2026-05-14
---

# Memory — claude-mem

A persistent-memory plugin that summarizes each session and re-injects relevant snippets at the start of the next one. Without it, every new Claude Code session opens cold — no recollection of what you discussed yesterday. With it, you get a SessionStart-level briefing of past decisions, files touched, and unresolved threads.

## Install

```powershell
npx claude-mem install
```

The installer auto-fetches **Bun** and **uv** if they're missing — no manual prereq install. Once it finishes, **restart Claude Code** for the hook to register.

Settings live at `~/.claude-mem/settings.json`. A web viewer comes up at `http://localhost:37777` once a session has actually written some memory.

> Don't run `npm install -g claude-mem` — that installs the SDK only, not the plugin hooks. Use `npx claude-mem install`.

## How it interacts with our vault hooks

| Hook event | What runs |
|---|---|
| `SessionStart` | claude-mem injects a transcript-derived memory summary **AND** `session_briefing.py` injects the current vault state (see [[06 - Hooks & Automation]]). |
| `Stop` | claude-mem writes its own memory entry **AND** `obsidian_log.py` writes a session note into `Sessions/`. |

Two parallel records, two different lenses:

- **claude-mem** = continuity. "What were you working on, and where did you leave off?" Auto-summarized, semantic.
- **Obsidian Sessions/** = an audit trail. "What was the last message exchange, how many tool calls, what time?" Mechanical, human-readable, searchable in Obsidian.

## Verifying it's firing

1. Start a session, edit a couple of files, exit.
2. Start another session in the same project.
3. You should see a SessionStart injection that references the prior work.

> **Note**: memory injection starts on the **second** session in a given project, not the first. The first session collects observations, the second one gets them back. So if your second session in SVRPTW doesn't show memory, exit and start a third.

If you don't see it: check `~/.claude/settings.json` for `enabledPlugins: { "claude-mem@thedotmack": true }`. The hooks are loaded from the plugin's own `hooks/hooks.json` once it's enabled — there's no separate hook block in your settings.json.

## Worker autostart

The installer prints "Worker autostart skipped — start it manually with `npx claude-mem start`". You usually don't need to — the `SessionStart` hook the plugin ships with calls `worker-service.cjs start` itself, so the worker comes up the first time you launch Claude Code after install.

If memory injection still isn't firing after two sessions, start the worker once manually:

```powershell
$env:Path = "$env:USERPROFILE\.bun\bin;$env:USERPROFILE\.local\bin;$env:Path"
npx claude-mem start
```

Then visit `http://localhost:37777` to see live observations.

## Requirements on this machine (already installed)

- **Bun** at `C:\Users\catst\.bun\bin\bun.exe` (1.3.14)
- **uv** at `C:\Users\catst\.local\bin\uv.exe` (0.11.14)
- **bash** at `C:\Program Files\Git\bin\bash.exe` (Git Bash — claude-mem's hooks are `"shell": "bash"`)

All three are on the persistent user PATH.

## Viewer

Once a session is running:

```powershell
Start-Process http://localhost:37777
```

The viewer shows extracted memory entries grouped by project. Useful for spot-checking whether a fact was captured cleanly, or for clearing out stale entries.

## See also

- [[06 - Hooks & Automation]] — the local hook stack claude-mem layers on top of
- [[08 - Context Mode]] — keep memory injections from blowing the context budget
- [[00 - Index]] — vault map
