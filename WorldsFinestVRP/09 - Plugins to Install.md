---
title: Plugins to Install
project: SVRPTW
tags: [plugins, install, claude-code, academic]
updated: 2026-05-14
---

# Plugins to install (manual steps)

These two plugins ship as Claude Code marketplaces. Installation goes through `/plugin` inside Claude Code — there is no Bash command that performs them. Run them once and they're enabled for every future session.

## 1. Academic Research Skills (ARS)

By Cheng-I Wu (`Imbad0202`). ~6.7K-star canonical repo. Adds `/ars-plan`, `/ars-lit-review`, `academic-paper`, `academic-paper-reviewer`, and an `integrity_verification_agent` that catches fabricated references.

**Why it matters for GART 2.0**: the integrity agent will catch the kind of hallucinated citations you already had to hunt down manually (the Kou 2023/2022 and BHH-duplicate hunts).

### Install

Open Claude Code in any project and run:

```
/plugin marketplace add Imbad0202/academic-research-skills
/plugin install academic-research-skills
```

Then restart Claude Code.

### Recommended environment flags

The full pipeline spawns parallel subagents and runs long. Before running the full pipeline, set:

```powershell
$env:CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS = "1"
```

And launch with `claude --dangerously-skip-permissions` **for trusted runs only**.

### Smoke test

```
/ars-plan
```

Should kick off a Socratic dialogue about paper structure. For a single-shot test:

```
/ars-lit-review "TSP tour-length estimation"
```

### Budget warning

A full 10-stage pipeline run can exceed **200K input + 100K output tokens**. Run individual skills (`academic-paper`, `academic-paper-reviewer`) for incremental work.

---

## 2. academic-writing-agents (andrehuang)

Auto-triggers on `.tex` files. Adds `/academic` slash command — e.g. `/academic review my methodology section` or `Audit the bibliography for missing fields and arXiv updates`.

### Install

Inside Claude Code:

```
claude plugin install andrehuang-academic-writing-agents
```

Or directly from GitHub if not in the marketplace yet:

```
claude plugin install --url https://github.com/andrehuang/academic-writing-agents
```

---

## Sanity-check after install

After restarting Claude Code, run `/plugin list` (or check `~/.claude/settings.json` → `enabledPlugins`). You should see:

```json
{
  "enabledPlugins": {
    "claude-mem@thedotmack": true,
    "academic-research-skills@Imbad0202": true,
    "andrehuang-academic-writing-agents": true
  }
}
```

If any are missing, re-run the matching install command.

## See also

- [[07 - Memory]] — claude-mem (already installed; the third plugin we set up automatically)
- [[06 - Hooks & Automation]] — project-local hook stack
