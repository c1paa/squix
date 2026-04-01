"""Debugger agent — finds bugs and fixes errors.

Works through the skill layer (read_file, search_in_files, find_main_file,
get_project_structure).  Returns a structured result contract with diagnosis,
fix, and verification.
"""

from __future__ import annotations

import json
import re

from squix.agents.base import AgentMessage, BaseAgent


class DebuggerAgent(BaseAgent):
    """Finds bugs, analyzes errors, and returns diagnoses + fixes.

    THINK → ACT (read files) → THINK (diagnose) → ACT (fix file) → OBSERVE (verify) → HANDOFF
    """

    agent_id = "debug"
    role = (
        "DEBUGGER — you are the bug hunter of the Squix AI system. "
        "You READ real files, ANALYZE errors, identify root causes, and "
        "patch files in place. "
        "NEVER just speculate — always read the actual code first. "
        "Use find_main_file skill to locate the relevant file. "
        "Return a structured result contract with diagnosis + fix."
    )

    async def handle(self, msg: AgentMessage) -> AgentMessage | None:
        self.progress = f"Debugging: {msg.content[:60]}"
        task = msg.content

        # ─── THINK: classify problem ───
        think_result = await self._think(task)

        # ─── ACT: find target file ───
        target_result = await self._find_target_file(task)
        if target_result.get("status") != "success":
            return self._make_result(
                status="failed",
                error=target_result.get("error", "No target file found"),
                task_id=msg.task_id,
            )

        target_path = target_result.get("path", "")

        # Read the file
        file_read = await self.invoke_skill("read_file", {"path": target_path})
        if file_read.get("status") != "success":
            return self._make_result(
                status="failed",
                error=f"Cannot read '{target_path}': {file_read.get('error', 'unknown')}",
                task_id=msg.task_id,
            )

        content = file_read.get("content", "")

        # ─── THINK: diagnose + produce fix ───
        diagnose = await self._analyze(task, target_path, content)
        diagnosis_text = diagnose.get("diagnosis", "")
        fix_code = diagnose.get("fix_code", "")

        # ─── ACT: write fixed file ───
        files_created = []
        files_modified = []
        file_errors = []

        if fix_code:
            if self._workspace:
                try:
                    self._workspace.write_file(target_path, fix_code)
                    files_modified.append(target_path)
                    if self._primary:
                        self._primary.track_write(target_path)
                except Exception as e:
                    file_errors.append(f"Failed to write fix: {e}")

            # Verify syntax
            verify = ""
            if self._workspace and target_path.endswith(".py"):
                try:
                    rc, stdout, stderr = await self._workspace.run_command(
                        ["python", "-m", "py_compile", target_path],
                        timeout=10,
                    )
                    if rc == 0:
                        verify = "✓ Syntax check passed"
                    else:
                        verify = f"✗ Syntax error: {stderr.strip()[:200]}"
                except Exception as e:
                    verify = f"Verification failed: {e}"

        # ─── HANDOFF: structured result contract ───
        next_steps = [f"Run: python {target_path}"] if target_path.endswith(".py") else []
        user_msg = (
            f"Нашёл и исправил багу в `{target_path}`.\n\n"
            f"**Диагноз:** {diagnosis_text[:300]}\n\n"
            f"**Проверка:** {verify}"
            if fix_code else
            f"**Диагноз:** {diagnosis_text}\n\n"
            "Автоматический фикс не был применён. См. анализ выше."
        )

        result_json = {
            "status": "success" if fix_code else "partial",
            "task_type": "debug",
            "summary": f"Diagnosed and patched `{target_path}`",
            "files_created": files_created,
            "files_modified": files_modified,
            "artifacts": [
                {"type": "fix_patch", "path": target_path, "diagnosis": diagnosis_text[:300]},
            ],
            "user_message": user_msg,
            "next_steps": next_steps,
            "errors": file_errors,
        }

        return AgentMessage(
            sender=self.agent_id,
            recipient="orch",
            content=json.dumps(result_json, indent=2, ensure_ascii=False),
            task_id=msg.task_id,
            metadata={
                "type": "work",
                "status": "success" if fix_code else "partial",
                "files_created": files_created,
                "files_modified": files_modified,
            },
        )

    # ─── THINK ─────────────────────────────────────────────────────────

    async def _think(self, task: str) -> dict:
        messages = [
            {"role": "system", "content": (
                "You are a debugger classifier. Given a user message about a code issue, "
                "classify it. Respond with JSON ONLY:\n"
                '{"type": "error|bug|improvement|investigation", '
                '"likely_files": ["files that might be relevant"], '
                '"description": "brief problem summary"}'
            )},
            {"role": "user", "content": task},
        ]
        resp = await self.invoke_llm(messages, temperature=0.3)
        return self._try_parse_json(
            resp.text,
            {"type": "investigation", "likely_files": [], "description": task},
        )

    # ─── ACT: find file ────────────────────────────────────────────────

    async def _find_target_file(self, task: str) -> dict:
        # Primary tracker
        if self._primary:
            pf = self._primary.get_primary()
            if pf:
                return {"status": "success", "path": pf, "reason": "primary_tracker"}

        # File references in task text
        refs = re.findall(r'(?:`([\w./_\\-]+)`|([\w./_\\-]+\.\w{2,}))', task)
        for bare, quoted in refs:
            ref = (quoted or bare).replace("\\", "/")
            if self._workspace and (self._workspace.project_dir / ref).exists():
                return {"status": "success", "path": ref, "reason": "mentioned_in_task"}

        # find_main_file skill
        result = await self.invoke_skill("find_main_file")
        if result.get("status") == "success" and result.get("path"):
            return result

        # Project structure scan
        struct = await self.invoke_skill("get_project_structure")
        if struct.get("status") == "success":
            files = struct.get("structure", "").split("\n")
            for f in files:
                if any(f.endswith(e) for e in (".py", ".js", ".ts", ".go")):
                    return {"status": "success", "path": f, "reason": "first_source"}

        return {"status": "error", "error": "No source file found in project"}

    # ─── THINK: analyze ────────────────────────────────────────────────

    async def _analyze(self, task: str, path: str, content: str) -> dict:
        messages = [
            {"role": "system", "content": (
                "You are a senior debugger. Given the real file content and a problem "
                "description, identify the root cause and produce the fixed version.\n\n"
                "Respond with JSON ONLY:\n"
                '{"diagnosis": "clear explanation of the bug and root cause", '
                '"fix_code": "the COMPLETE fixed file content, ready to be written"}\n\n'
                "The fix_code must be the ENTIRE file — not a diff. "
                "Everything not changed must still be included."
            )},
            {"role": "user", "content": f"PROBLEM: {task}\n\nFILE: {path}\n```{content}\n```"},
        ]
        resp = await self.invoke_llm(messages, temperature=0.1)
        return self._try_parse_json(resp.text, {"diagnosis": resp.text, "fix_code": ""})

    # ─── Helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _try_parse_json(text: str, default: dict) -> dict:
        text = text.strip()
        if "```" in text:
            for part in text.split("```"):
                part = part.strip()
                if part.startswith("json"):
                    part = part[4:].strip()
                if part.startswith("{"):
                    try:
                        return json.loads(part)
                    except json.JSONDecodeError:
                        pass
        if text.startswith("{"):
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                pass
        return default

    def _make_result(self, status: str, task_id: str, error: str) -> AgentMessage:
        result_json = {
            "status": status,
            "summary": error,
            "files_created": [],
            "files_modified": [],
            "artifacts": [],
            "user_message": f"Не удалось выполнить отладку: {error}",
            "next_steps": [],
            "errors": [error],
        }
        return AgentMessage(
            sender=self.agent_id, recipient="orch",
            content=json.dumps(result_json, indent=2),
            task_id=task_id,
            metadata={
                "type": "work", "status": status,
                "error": status == "failed",
                "files_created": [],
                "files_modified": [],
            },
        )
