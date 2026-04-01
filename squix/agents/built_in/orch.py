"""Orchestrator agent — the operational manager.

orch:
  - Receives tasks from talk (always, via delegate)
  - Decides which internal agent should handle it
  - Dispatches → collects → validates → formats user output
  - NEVER talks to the user directly — always returns final_result
  - If result is empty or verification fails → retry or error
"""

from __future__ import annotations

import asyncio
import json
import logging
import re

from squix.agents.base import AgentMessage, BaseAgent

logger = logging.getLogger("squix.agent.orch")


class OrchestratorAgent(BaseAgent):
    """Central operations dispatcher with result contract enforcement."""

    agent_id = "orch"
    role = (
        "ORCHESTRATOR — you are the operations manager of the Squix AI company. "
        "You receive tasks from Talk, decide which internal agents "
        "(build, debug, web, DB, AI, README) should handle them, dispatch in "
        "order, collect results, VALIDATE they actually created files / did work, "
        "and return a structured summary to the user. "
        "You NEVER talk directly to the user without a result. "
        "If a worker fails, retry once or return structured error."
    )

    # Result contract fields that must be present
    _REQUIRED_FIELDS = {"status"}

    async def handle(self, msg: AgentMessage) -> AgentMessage | None:
        msg_type = msg.metadata.get("type", "")

        # Extract task classifier metadata
        task_type = msg.metadata.get("task_type", "")
        output_mode = msg.metadata.get("output_mode", "")

        if msg_type == "plan":
            return await self._handle_plan(msg)
        elif msg_type == "work":
            return await self._handle_worker_result(msg)
        else:
            # This is the main path — delegated task from Talk
            return await self._handle_direct_task(msg, task_type, output_mode)

    # ── Main path: task from Talk ──────────────────────────────────────

    async def _handle_direct_task(
        self, msg: AgentMessage, task_type: str, output_mode: str,
    ) -> AgentMessage | None:
        """Primary flow: Talk → Orch → worker → validate → user."""
        task = msg.content
        await self.set_progress(f"Deciding worker for: {task[:60]}")

        # Step 1: Decide which worker to use
        worker = await self._decide_worker(task, task_type)

        if not self._send_fn:
            return self._format_failure(task, msg.task_id, "No send function")

        await self.set_progress(f"Dispatching to {worker}")

        # Step 2: Dispatch with session context
        session_ctx = msg.metadata.get("session_context", "")
        step_msg = AgentMessage(
            sender=self.agent_id,
            recipient=worker,
            content=task,
            task_id=msg.task_id,
            metadata={
                "type": "step",
                "orch_direct": True,
                "task_type": task_type,
                "session_context": session_ctx,
            },
        )
        await self._send_fn(step_msg)

        # Step 3: Wait for result
        try:
            result_msg = await asyncio.wait_for(
                self._inbox.get(), timeout=240,
            )
            result_content = result_msg.content
            result_meta = result_msg.metadata
        except TimeoutError:
            return self._format_failure(
                task, msg.task_id,
                f"Worker '{worker}' timeded out (240s)",
            )

        # Step 4: Validate result contract
        is_ok, error_reason = self._validate_result(
            result_content, result_meta, worker, output_mode,
        )

        if not is_ok:
            logger.warning(
                "Result validation failed for %s: %s", worker, error_reason,
            )
            # Retry once
            step_msg2 = AgentMessage(
                sender=self.agent_id,
                recipient=worker,
                content=(
                    f"Previous attempt did not meet the result contract. "
                    f"Error: {error_reason}.\n\n"
                    f"Original task: {task}\n\n"
                    f"Please try again and ensure your response includes "
                    f"structured output with status and the requested artifacts."
                ),
                task_id=msg.task_id,
                metadata={"type": "step", "orch_retry": True},
            )
            await self._send_fn(step_msg2)

            try:
                result_msg2 = await asyncio.wait_for(
                    self._inbox.get(), timeout=60,
                )
                result_content = result_msg2.content
                result_meta = result_msg2.metadata
            except TimeoutError:
                return self._format_failure(
                    task, msg.task_id,
                    f"Worker '{worker}' timeded out on retry",
                )

        # Step 5: Verify files exist (if files were claimed)
        files_created = result_meta.get("files_created", [])
        files_modified = result_meta.get("files_modified", [])
        file_errors = self._verify_files(created=files_created, modified=files_modified)

        # Step 6: Format user-facing output
        final_content = self._format_user_output(
            task=task,
            worker=worker,
            result_content=result_content,
            result_meta=result_meta,
            files_created=files_created,
            files_modified=files_modified,
            file_errors=file_errors,
            output_mode=output_mode,
        )

        return AgentMessage(
            sender=self.agent_id,
            recipient="user",
            content=final_content,
            task_id=msg.task_id,
            metadata={
                "type": "final_result",
                "worker": worker,
                "task_type": task_type,
                "files_created": files_created,
                "files_modified": files_modified,
                "file_errors": file_errors,
            },
        )

    # ── Plan execution ─────────────────────────────────────────────────

    async def _handle_plan(self, msg: AgentMessage) -> AgentMessage | None:
        """Execute a plan from the Planner agent."""
        await self.set_progress("Parsing plan...")
        steps = self._extract_steps(msg.content)

        if not steps:
            return AgentMessage(
                sender=self.agent_id, recipient="user",
                content=f"**Plan:**\n{msg.content}\n\n_No actionable steps found._",
                task_id=msg.task_id, metadata={"type": "final_result"},
            )

        results_text = []
        for i, step in enumerate(steps):
            target = step.get("agent", "build")
            task_text = step.get("task", "")

            if not self._send_fn:
                continue

            await self.set_progress(f"Step {i+1}/{len(steps)}: → {target}")

            await self._send_fn(AgentMessage(
                sender=self.agent_id, recipient=target,
                content=task_text, task_id=msg.task_id,
                metadata={"type": "step", "step_num": i + 1},
            ))

            try:
                r = await asyncio.wait_for(self._inbox.get(), timeout=60)
                results_text.append(
                    f"\n**Step {i+1} ({target}):** {task_text}\n"
                    f"**Result:** {r.content}\n",
                )
            except TimeoutError:
                results_text.append(
                    f"\n**Step {i+1} ({target}):** {task_text}\n"
                    f"**Result:** [timeout]\n",
                )

        return AgentMessage(
            sender=self.agent_id, recipient="user",
            content="\n".join(results_text), task_id=msg.task_id,
            metadata={"type": "final_result"},
        )

    # ── Worker result passthrough ──────────────────────────────────────

    async def _handle_worker_result(self, msg: AgentMessage) -> AgentMessage | None:
        """Relay a worker result when orch is relaying."""
        return AgentMessage(
            sender=self.agent_id, recipient="user",
            content=msg.content, task_id=msg.task_id,
            metadata={
                "type": "final_result", "worker": msg.sender,
            },
        )

    # ── Worker selection ───────────────────────────────────────────────

    async def _decide_worker(self, task: str, task_type: str) -> str:
        """Decide which worker handles this task."""
        lower = task.lower()

        # If task_type was set by Talk classifier, use it directly
        if task_type == "debugging":
            return "debug"
        if task_type in ("code_generate", "code_edit"):
            return "build"
        if task_type == "research":
            return "web"
        if task_type == "docs_write":
            return "README"

        # Fallback: keyword check
        if any(kw in lower for kw in _DEBUG_KW):
            return "debug"
        if any(kw in lower for kw in _BUILD_KW):
            return "build"
        if any(kw in lower for kw in _RESEARCH_KW):
            return "web"
        if any(kw in lower for kw in _DOCS_KW):
            return "README"

        # LLM decision
        messages = [
            {"role": "system", "content": (
                "You are the Orchestrator. Choose the worker agent for this task. "
                "Reply with ONE word: build, debug, web, DB, AI, README."
            )},
            {"role": "user", "content": task},
        ]
        resp = await self.invoke_llm(messages, temperature=0.1)
        choice = resp.text.strip().lower()
        valid = {"build", "debug", "web", "DB", "AI", "README", "plan", "idea", "product"}
        return choice if choice in valid else "build"

    # ── Result contract validation ─────────────────────────────────────

    def _validate_result(
        self,
        content: str,
        metadata: dict,
        worker: str,
        output_mode: str,
    ) -> tuple[bool, str]:
        """Check that the worker returned a non-empty, valid result.

        Returns (is_valid, reason_if_not).
        """
        # Empty check
        if not content or content.strip() == "":
            return False, f"{worker} returned empty content"

        # Check for errors in metadata
        if metadata.get("error"):
            return False, f"{worker} reported error: {content[:200]}"

        # Parse JSON result contract if present
        json_result = self._try_parse_result_json(content)
        if json_result:
            status = json_result.get("status", "")
            if status in ("failed", "error"):
                return False, json_result.get("user_message", content[:200])
            # If it has required fields, pass
            missing = self._REQUIRED_FIELDS - set(json_result.keys())
            if missing:
                return False, f"Missing fields: {missing}"
            return True, ""

        # Non-JSON content is acceptable as free-form result
        return True, ""

    @staticmethod
    def _try_parse_result_json(content: str) -> dict | None:
        """Try to extract JSON result contract from content."""
        text = content.strip()
        if "```" in text:
            parts = text.split("```")
            for part in parts:
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
        return None

    # ── File existence verification ────────────────────────────────────

    def _verify_files(
        self, created: list[str], modified: list[str],
    ) -> list[str]:
        """Verify that claimed files actually exist in workspace."""
        errors = []
        all_paths = created + modified
        for path in all_paths:
            if not self._workspace:
                continue
            try:
                full = self._workspace.project_dir / path
                if not full.exists():
                    errors.append(f"File not found after write: {path}")
            except Exception as e:
                errors.append(f"Error verifying {path}: {e}")
        return errors

    # ── User output formatting ─────────────────────────────────────────

    def _format_user_output(
        self,
        task: str,
        worker: str,
        result_content: str,
        result_meta: dict,
        files_created: list[str],
        files_modified: list[str],
        file_errors: list[str],
        output_mode: str,
    ) -> str:
        """Format the final result in human-readable form for the user."""
        parts = []

        # Try to parse structured result
        structured = self._try_parse_result_json(result_content)

        if structured:
            summary = structured.get("summary", "")
            user_msg = structured.get("user_message", "")
            next_steps = structured.get("next_steps", [])
            errors = structured.get("errors", [])

            if user_msg:
                parts.append(user_msg)
            elif summary:
                parts.append(summary)
            else:
                parts.append(result_content[:500])

            if next_steps:
                parts.append("**Next steps:**")
                for ns in next_steps:
                    parts.append(f"  - {ns}")
        else:
            # Free-form result
            parts.append(result_content[:1000])

        # File actions display
        if files_created:
            parts.append("\n📁 **Created:**")
            for f in files_created:
                parts.append(f"  - `{f}`")
        if files_modified:
            parts.append("\n✏️ **Modified:**")
            for f in files_modified:
                parts.append(f"  - `{f}`")

        if file_errors:
            parts.append("\n⚠️ **File errors:**")
            for e in file_errors:
                parts.append(f"  - {e}")

        return "\n".join(parts)

    def _format_failure(self, task: str, task_id: str, reason: str) -> AgentMessage:
        """Return structured failure to the user."""
        return AgentMessage(
            sender=self.agent_id, recipient="user",
            content=(
                f"❌ **Task failed:** {reason}\n\n"
                f"I started working on your request but could not complete it.\n"
                f"Task: {task[:300]}\n\n"
                f"Try rephrasing your request or check if there are any issues "
                f"with the project setup."
            ),
            task_id=task_id,
            metadata={"type": "final_result", "status": "failed"},
        )

    @staticmethod
    def _extract_steps(plan_text: str) -> list[dict[str, str]]:
        """Parse JSON or naive line splitting."""
        plan_text = plan_text.strip()
        if "```" in plan_text:
            parts = plan_text.split("```")
            for part in parts:
                part = part.strip()
                if part.startswith("json"):
                    part = part[4:].strip()
                if part.startswith("{") or part.startswith("["):
                    plan_text = part
                    break

        try:
            data = json.loads(plan_text)
            if isinstance(data, dict) and "steps" in data:
                return data["steps"]
            if isinstance(data, list):
                return data
        except (json.JSONDecodeError, TypeError):
            pass

        steps = []
        lines = re.split(r'\n\s*[\d\-\*•]+\.?\s*', plan_text)
        for line in lines:
            line = line.strip()
            if line and len(line) > 5:
                steps.append({"agent": "build", "task": line})
        return steps if steps else [{"agent": "build", "task": plan_text}]


# ── Keyword helpers for _decide_worker ─────────────────────────────────

_DEBUG_KW = (
    "not working", "не работает", "почини", "исправ",
    "ошибк", "баг", "глюк", "bug", "error",
    "debug", "broken", "краш", "проблем",
)
_BUILD_KW = (
    "write", "создай", "напиши", "сделай", "добавь",
    "build", "create", "implement", "игра", "game",
)
_RESEARCH_KW = (
    "найди", "search", "look up", "погугли",
)
_DOCS_KW = (
    "readme", "документацию", "documentation", "инструкцию",
)
