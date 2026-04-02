"""Debugger agent — finds bugs and fixes errors via agentic loop.

Uses the agentic tool-use loop to read code, diagnose issues,
apply targeted fixes with edit_file, and verify the result.
"""

from __future__ import annotations

import json

from squix.agents.base import AgentMessage, BaseAgent

DEBUG_SKILLS = [
    "read_file", "write_file", "edit_file",
    "list_files", "find_main_file", "search_in_files",
    "run_command", "run_tests", "get_project_structure",
    "git_status", "git_diff",
]

DEBUG_SYSTEM = """\
You are DEBUGGER — the bug hunter of Squix.

Your job is to find bugs, diagnose errors, and apply precise fixes to REAL files using tools.
You NEVER speculate. You READ actual code, FIND the problem, and FIX it with tools.

## Workflow

1. **Locate** — call find_main_file / list_files / search_in_files to find relevant file(s)
2. **Read** — call read_file to see the actual code (REQUIRED before editing)
3. **Diagnose** — identify the root cause from the code you read
4. **Fix** — call edit_file(old_string, new_string) for precise surgical fixes
5. **Verify** — check syntax_check in the result; call run_tests or run_command to confirm
6. **Iterate** — if the fix introduced a new error, read the error and fix again

## CRITICAL RULES

- NEVER output code as text. Always use edit_file or write_file tools.
- ALWAYS call read_file before edit_file (edit_file requires prior read).
- Prefer edit_file for targeted fixes — do NOT rewrite the whole file unless necessary.
- old_string in edit_file must be an EXACT match of text in the file.
- After fixing, verify with run_tests or run_command if possible.
- If syntax_check shows an error, fix it immediately.
- When DONE, respond with plain text explaining what you found and fixed (no JSON, no code blocks).
"""


class DebuggerAgent(BaseAgent):
    """Finds bugs, analyzes errors, and applies fixes via agentic loop."""

    agent_id = "debug"
    role = (
        "DEBUGGER — bug hunter that reads code, diagnoses issues, "
        "and applies targeted fixes using an agentic tool-use loop."
    )

    async def handle(self, msg: AgentMessage) -> AgentMessage | None:
        await self.set_progress(f"Debugging: {msg.content[:60]}")
        task = msg.content
        session_ctx = msg.metadata.get("session_context", "")

        system = DEBUG_SYSTEM
        if session_ctx:
            system += (
                f"\n## Session Context\n\n{session_ctx}\n"
                "Use this to understand what the user has been working on.\n"
            )

        final_text, tool_log = await self.run_agentic_loop(
            task=task,
            system_prompt=system,
            available_skills=DEBUG_SKILLS,
            max_iterations=15,
            temperature=0.2,
        )

        # Analyze tool log
        files_modified: list[str] = []
        errors: list[str] = []

        for entry in tool_log:
            tool = entry["tool"]
            result = entry.get("result", {})
            path = entry.get("params", {}).get("path", "")

            if tool in ("edit_file", "write_file", "patch_file"):
                if result.get("status") == "success" and path and path not in files_modified:
                    files_modified.append(path)
            if result.get("status") == "error":
                errors.append(f"{tool}: {result.get('error', 'unknown')}")
            if result.get("syntax_check", "").startswith("error"):
                errors.append(f"syntax: {result['syntax_check']}")

        status = "success" if files_modified else ("partial" if not errors else "failed")
        summary = (
            f"Fixed {len(files_modified)} file(s): {', '.join(f'`{f}`' for f in files_modified)}"
            if files_modified else final_text[:200]
        )

        result_json = {
            "status": status,
            "task_type": "debug",
            "summary": summary,
            "files_created": [],
            "files_modified": files_modified,
            "artifacts": [
                {"type": "fix_patch", "path": f}
                for f in files_modified
            ],
            "user_message": final_text,
            "next_steps": [f"Run: python {f}" for f in files_modified if f.endswith(".py")],
            "errors": errors,
            "iterations": len(set(e["iteration"] for e in tool_log)) if tool_log else 0,
        }

        await self.set_progress(f"Done: fixed {len(files_modified)} file(s)")

        return AgentMessage(
            sender=self.agent_id,
            recipient="orch",
            content=json.dumps(result_json, indent=2, ensure_ascii=False),
            task_id=msg.task_id,
            metadata={
                "type": "work",
                "status": status,
                "files_created": [],
                "files_modified": files_modified,
            },
        )
