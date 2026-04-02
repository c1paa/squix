"""Builder agent — writes code and implements solutions via agentic loop.

Uses the agentic tool-use loop: LLM decides which tools to call,
executes them, observes results, and repeats until the task is done.
"""

from __future__ import annotations

import json

from squix.agents.base import AgentMessage, BaseAgent

# Skills this agent is allowed to use in its loop
BUILD_SKILLS = [
    "read_file", "write_file", "edit_file", "patch_file",
    "list_files", "find_main_file", "search_in_files",
    "run_command", "run_tests", "get_project_structure",
    "git_status", "git_diff", "git_add", "git_commit",
]

BUILD_SYSTEM = """\
You are BUILDER — the code execution engine of Squix.

Your job is to write, create, and modify REAL FILES in the project using tools.
You NEVER output code as text. You ALWAYS use write_file or edit_file tools.

## Workflow

1. **Understand** — call list_files or get_project_structure to see what exists
2. **Read** — call read_file to read existing files (REQUIRED before edit_file)
3. **Write** — call write_file for new files, edit_file for modifications
4. **Verify** — check syntax_check in tool results; if error → fix with edit_file
5. **Repeat** — if something failed, read error and fix

## CRITICAL: How to create/write files

To create a new file or overwrite an existing one, call write_file:
```json
[{"tool": "write_file", "params": {"path": "filename.py", "content": "...full code..."}}]
```

To modify part of an existing file (MUST read_file first):
```json
[{"tool": "edit_file", "params": {"path": "filename.py", "old_string": "exact old text", "new_string": "new text"}}]
```

## Rules

- NEVER output code as plain text. Always use write_file or edit_file tools.
- If the user mentions a specific file (like main.py), use THAT file.
- For new files or complete rewrites → write_file with full content in "content" param
- For small changes to existing files → read_file first, then edit_file
- After writing Python → check syntax_check in result. If error → fix immediately.
- When DONE → respond with plain text summary of what you did (no JSON, no code blocks).
"""


class BuilderAgent(BaseAgent):
    """Writes code, implements features, creates files via agentic loop."""

    agent_id = "build"
    role = (
        "BUILDER — code executor that writes and modifies real files "
        "using an agentic tool-use loop."
    )

    async def handle(self, msg: AgentMessage) -> AgentMessage | None:
        await self.set_progress(f"Building: {msg.content[:80]}")
        task = msg.content
        session_ctx = msg.metadata.get("session_context", "")

        # Build the system prompt with optional session context
        system = BUILD_SYSTEM
        if session_ctx:
            system += (
                f"\n## Session Context\n\n{session_ctx}\n"
                "Use this to understand what the user has been working on.\n"
            )

        # Run the agentic loop — large max_tokens for full file writes
        final_text, tool_log = await self.run_agentic_loop(
            task=task,
            system_prompt=system,
            available_skills=BUILD_SKILLS,
            max_iterations=15,
            max_tokens=16384,
            temperature=0.3,
        )

        # Analyze tool log to build result contract
        files_created: list[str] = []
        files_modified: list[str] = []
        errors: list[str] = []

        for entry in tool_log:
            tool = entry["tool"]
            result = entry.get("result", {})
            path = entry.get("params", {}).get("path", "")

            if tool == "write_file" and result.get("status") == "success":
                if path and path not in files_created:
                    files_created.append(path)
            elif tool == "edit_file" and result.get("status") == "success":
                if path and path not in files_modified and path not in files_created:
                    files_modified.append(path)
            elif tool == "patch_file" and result.get("status") == "success":
                if path and path not in files_modified and path not in files_created:
                    files_modified.append(path)

            if result.get("status") == "error":
                errors.append(f"{tool}: {result.get('error', 'unknown')}")
            if result.get("syntax_check", "").startswith("error"):
                errors.append(f"syntax: {result['syntax_check']}")

        all_files = files_created + files_modified
        if all_files:
            summary = (
                f"{'Created' if files_created else 'Modified'} "
                f"{len(all_files)} file(s): {', '.join(f'`{f}`' for f in all_files)}"
            )
            status = "success"
        else:
            summary = final_text[:200]
            status = "success" if not errors else "failed"

        result_json = {
            "status": status,
            "task_type": "code_generate" if files_created else "code_edit",
            "summary": summary,
            "files_created": files_created,
            "files_modified": files_modified,
            "artifacts": [
                {"type": "code", "path": f, "language": _guess_lang(f)}
                for f in all_files
            ],
            "user_message": final_text,
            "next_steps": [f"Run: python {f}" for f in all_files if f.endswith(".py")],
            "errors": errors,
            "iterations": len(set(e["iteration"] for e in tool_log)) if tool_log else 0,
        }

        await self.set_progress(
            f"Done: {len(files_created)} created, {len(files_modified)} modified"
        )

        return AgentMessage(
            sender=self.agent_id,
            recipient="orch",
            content=json.dumps(result_json, indent=2, ensure_ascii=False),
            task_id=msg.task_id,
            metadata={
                "type": "work",
                "status": status,
                "files_created": files_created,
                "files_modified": files_modified,
            },
        )


def _guess_lang(path: str) -> str:
    ext = path.rsplit(".", 1)[-1].lower() if "." in path else ""
    return {
        "py": "python", "js": "javascript", "ts": "typescript",
        "rs": "rust", "go": "go", "rb": "ruby", "java": "java",
        "c": "c", "cpp": "cpp",
    }.get(ext, "text")
