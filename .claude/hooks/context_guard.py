"""
PreToolUse hook for Bash: warns when a command is likely to flood context.
Exits 0 (allow) always — this is advisory only, surfaces a warning via stderr.

Receives JSON on stdin:
  {"tool_name": "Bash", "tool_input": {"command": "..."}}
"""
import json
import sys

FLOOD_PATTERNS = [
    "cat ",
    "head ",
    "tail ",
    "grep ",
    "find /",
    "ls -la /d",
    "ls -R",
    "pip list",
    "conda list",
]

SAFE_PREFIXES = [
    "git ",
    "mkdir",
    "rm ",
    "mv ",
    "cp ",
    "python ",
    "wc -l",
]


def main():
    try:
        data = json.load(sys.stdin)
    except Exception:
        sys.exit(0)

    command = data.get("tool_input", {}).get("command", "")

    for safe in SAFE_PREFIXES:
        if command.strip().startswith(safe):
            sys.exit(0)

    for pattern in FLOOD_PATTERNS:
        if pattern in command:
            print(
                f"[context_guard] '{pattern}' can produce large output. "
                "Prefer ctx_batch_execute or pipe to | head -20.",
                file=sys.stderr,
            )
            break

    sys.exit(0)


if __name__ == "__main__":
    main()
