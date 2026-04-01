"""Builder agent — writes code and implements solutions.

Works through the skill layer (read_file, write_file, patch_file,
find_main_file).  Returns a structured result contract that Orch validates.
"""

from __future__ import annotations

import json
import re

from squix.agents.base import AgentMessage, BaseAgent


class BuilderAgent(BaseAgent):
    """Writes code, implements features, creates files.

    THINK → ACT (skills) → OBSERVE → HANDOFF (structured result contract)
    """

    agent_id = "build"
    role = (
        "BUILDER — you are the code executor of the Squix AI system. "
        "You write, modify, and create real files in the project. "
        "NEVER just describe what should be done — DO it using skills. "
        "ALWAYS use: find_main_file/read_file → generate → write_file cycle. "
        "Return a structured JSON result contract at the end."
    )

    async def handle(self, msg: AgentMessage) -> AgentMessage | None:
        await self.set_progress(f"Building: {msg.content[:80]}")
        task = msg.content
        session_ctx = msg.metadata.get("session_context", "")

        # Snapshot of files that already exist BEFORE we do anything
        preexisting_files: set[str] = set()
        if self._workspace:
            try:
                preexisting_files = {
                    f.relative_to(self._workspace.project_dir).as_posix()
                    for f in self._workspace.project_dir.rglob("*")
                    if f.is_file() and ".squix" not in str(f)
                }
            except Exception:
                pass

        # ─── ACT: find target file ───
        target_result = await self._find_target_file(task)
        if target_result.get("status") != "success":
            return self._make_result(
                status="failed",
                error=target_result.get("error", "No target file found"),
                task_id=msg.task_id,
            )

        target_path = target_result.get("path", "")
        reason = target_result.get("reason", "")
        await self.set_progress(f"Target file: {target_path}")

        # Read existing content
        existing = ""
        if self._workspace:
            try:
                existing = self._workspace.read_file(target_path)
                await self.set_progress(f"Read {target_path} ({len(existing)} bytes)")
            except Exception:
                pass

        # ─── THINK + ACT: generate code directly ───
        ctx_block = ""
        if session_ctx:
            ctx_block = (
                f"\n--- SESSION CONTEXT ---\n{session_ctx}\n"
                "--- END SESSION CONTEXT ---\n"
                "Use this to understand what the user has been working on.\n\n"
            )

        if existing:
            system = (
                f"You are BUILDER: a code executor. "
                f"Your job: modify '{target_path}'.\n"
                f"Task: {task[:200]}\n\n"
                f"{ctx_block}"
                f"CURRENT FILE:\n```{existing[:8000]}\n```\n\n"
                "INSTRUCTIONS:\n"
                "1. Produce the COMPLETE, FINAL version.\n"
                "2. Wrap in a code fence with the language tag.\n"
                "3. Include a filename hint before the fence, like 'File: <path>'.\n"
                "4. Do NOT explain — just produce the code.\n"
            )
        else:
            system = (
                f"You are BUILDER: a code executor. "
                f"Your job: create file '{target_path}'.\n"
                f"Task: {task[:200]}\n\n"
                f"{ctx_block}"
                "INSTRUCTIONS:\n"
                "1. Produce a COMPLETE, WORKING implementation.\n"
                "2. Wrap in a code fence with the language tag.\n"
                "3. Include a filename hint before the fence, like 'File: <path>'.\n"
                "4. Do NOT explain — just produce the code.\n"
            )

        await self.set_progress("Generating code...")
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": task},
        ]
        llm_result = await self.invoke_llm(messages, temperature=0.3, max_tokens=8196)
        await self.set_progress("Extracting and writing code...")

        # ─── ACT: extract code blocks and write ───
        wrote = self._extract_and_write(target_path, llm_result.text)

        # ─── OBSERVE: verify files exist ───
        file_errors = []
        for f in wrote:
            if self._workspace and not (self._workspace.project_dir / f).exists():
                file_errors.append(f"File was reported as written but does not exist: {f}")

        # ─── HANDOFF: structured result contract ───
        # Use preexisting snapshot to distinguish created vs modified
        files_created = [f for f in wrote if f not in preexisting_files]
        files_modified = [f for f in wrote if f in preexisting_files]
        # If target existed, was modified, and wasn't in wrote, it still counts
        if target_path in preexisting_files and target_path not in wrote:
            files_modified.append(target_path)

        if not wrote and not file_errors:
            return self._make_result(
                status="failed",
                error="No code blocks found in LLM response; no files wrote",
                task_id=msg.task_id,
            )

        if files_created:
            summary = f"Generated {len(files_created)} file(s): {', '.join(f'`{f}`' for f in files_created)}"
            user_msg = f"Готово, я создал {', '.join(f'`{f}`' for f in files_created)}."
            next_steps = [f"Run: python {f}" for f in files_created if f.endswith(".py")]
        elif files_modified:
            summary = f"Modified {len(files_modified)} file(s): {', '.join(f'`{f}`' for f in files_modified)}"
            user_msg = f"Готово, я обновил {', '.join(f'`{f}`' for f in files_modified)}."
            next_steps = [f"Run: python {f}" for f in files_modified if f.endswith(".py")]
        else:
            summary = f"No changes written to {target_path}"
            user_msg = "Не удалось внести изменения в файл."
            next_steps = []

        # Build structured JSON result
        result_json = {
            "status": "success",
            "task_type": "code_generate" if files_created else "code_edit",
            "summary": summary,
            "files_created": files_created,
            "files_modified": files_modified,
            "artifacts": [
                {"type": "code", "path": f, "language": self._guess_lang(f)}
                for f in wrote
            ],
            "user_message": user_msg,
            "next_steps": next_steps,
            "errors": file_errors,
        }

        await self.set_progress(f"Done: {len(files_created)} created, {len(files_modified)} modified")

        return AgentMessage(
            sender=self.agent_id,
            recipient="orch",
            content=json.dumps(result_json, indent=2, ensure_ascii=False),
            task_id=msg.task_id,
            metadata={
                "type": "work",
                "status": "success",
                "files_created": files_created,
                "files_modified": files_modified,
            },
        )

    # ─── ACT: find file ────────────────────────────────────────────────

    async def _find_target_file(self, task: str) -> dict:
        """Find target file OR generate a filename for new projects.

        Priority:
        1. Primary tracker (last file user worked with)
        2. File mentioned in task text (backtick refs)
        3. Scan project structure for main files
        4. LLM decides filename for new project / feature
        """
        # Primary tracker (most recent file user was working with)
        if self._primary:
            pf = self._primary.get_primary()
            if pf:
                return {"status": "success", "path": pf, "reason": "primary_tracker"}

        # Extract file references from task
        refs = re.findall(r'(?:`([\w./_\\-]+)`|([\w./_\\-]+\.\w{2,}))', task)
        for bare, quoted in refs:
            ref = (quoted or bare).replace("\\", "/")
            if self._workspace and (self._workspace.project_dir / ref).exists():
                return {"status": "success", "path": ref, "reason": "mentioned_in_task"}

        # Skill-based find
        result = await self.invoke_skill("find_main_file")
        if result.get("status") == "success" and result.get("path"):
            return result

        # Project structure scan
        struct = await self.invoke_skill("get_project_structure")
        if struct.get("status") == "success":
            files = struct.get("structure", "").split("\n")
            for f in files:
                if any(f.endswith(e) for e in (".py", ".js", ".ts", ".go", ".rs")):
                    return {"status": "success", "path": f, "reason": "first_source"}

        # No files exist — ask LLM to choose a filename for this task
        return await self._suggest_filename(task)

    async def _suggest_filename(self, task: str) -> dict:
        """Ask LLM to suggest a filename for a new project/feature."""
        messages = [
            {"role": "system", "content": (
                'Given a task description, suggest ONE appropriate filename '
                'for the main code file. Reply with ONLY the filename, nothing else.\n'
                'Examples:\n'
                '  "write a snake game" → snake.py\n'
                '  "create a calculator" → calculator.py\n'
                '  "make a web server" → server.py\n'
                'Use .py for Python, .js for JS, etc.'
            )},
            {"role": "user", "content": task},
        ]
        resp = await self.invoke_llm(messages, temperature=0.1)
        filename = resp.text.strip().strip('`').strip('"').strip("'").strip()
        if not filename or "." not in filename:
            # Fallback: snake_case from first word
            import re as re2
            match = re2.search(r'[a-zA-Zа-яА-Я]+', task)
            if match:
                filename = match.group(0).lower() + ".py"
            else:
                filename = "main.py"

        return {
            "status": "success",
            "path": filename.split("/")[-1],
            "reason": "llm_suggested",
        }

    # ─── ACT: extract and write ────────────────────────────────────────

    def _extract_and_write(self, default_target: str, text: str) -> list[str]:
        pattern = re.compile(r'```(?:\w+)?\s*\n([\s\S]*?)(?=\n```)', re.MULTILINE)
        file_hint = re.compile(
            r'(?:file[:\s]*|path[:\s]*)?[`"]?([\w./_\\-]+\.\w{2,})[`"]?\s*',
            re.IGNORECASE,
        )
        wrote = []
        blocks = list(pattern.finditer(text))

        if not blocks:
            # No code fence — write raw response as file
            if self._workspace:
                try:
                    self._workspace.write_file(default_target, text)
                    wrote.append(default_target)
                except Exception:
                    pass
            return wrote

        for i, block in enumerate(blocks):
            code = block.group(1)
            start = max(0, block.start() - 200)
            end = min(len(text), block.end() + 50)
            m = file_hint.search(text[start:end])

            filename = default_target
            if m:
                raw = m.group(1).replace("\\", "/")
                if self._workspace and (self._workspace.project_dir / raw).exists():
                    filename = raw
                else:
                    filename = raw.split("/")[-1]

            if self._workspace:
                try:
                    self._workspace.write_file(filename, code)
                    wrote.append(filename)
                    if self._primary:
                        self._primary.track_write(filename)
                except Exception:
                    pass
        return wrote

    # ─── Helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _guess_lang(path: str) -> str:
        ext = path.rsplit(".", 1)[-1].lower() if "." in path else ""
        return {
            "py": "python", "js": "javascript", "ts": "typescript",
            "rs": "rust", "go": "go", "rb": "ruby", "java": "java",
            "c": "c", "cpp": "cpp",
        }.get(ext, "text")

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
        try:
            return json.loads(text)
        except Exception:
            return default

    def _make_result(
        self, status: str, task_id: str, error: str, summary: str = "",
    ) -> AgentMessage:
        result_json = {
            "status": status,
            "summary": summary or error,
            "files_created": [],
            "files_modified": [],
            "artifacts": [],
            "user_message": (
                error if status == "failed" else summary
            ),
            "next_steps": [],
            "errors": [error] if status == "failed" else [],
        }
        return AgentMessage(
            sender=self.agent_id, recipient="orch",
            content=json.dumps(result_json, indent=2),
            task_id=task_id,
            metadata={
                "type": "work",
                "status": status,
                "error": status == "failed",
            },
        )
