"""Orchestrator agent — the central task dispatcher managed by Planner."""

from __future__ import annotations

import asyncio
import json
import logging
import re

from squix.agents.base import AgentMessage, BaseAgent

logger = logging.getLogger("squix.agent.orch")


class OrchestratorAgent(BaseAgent):
    """Receives a plan from Planner and dispatches steps to worker agents.
    Collects worker results and produces a final summary."""

    agent_id = "orch"
    role = "Orchestrator — dispatch tasks to workers and coordinate the team."

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._worker_results: list[dict[str, str]] = []

    async def handle(self, msg: AgentMessage) -> AgentMessage | None:
        msg_type = msg.metadata.get("type", "")

        if msg_type == "plan":
            return await self._handle_plan(msg)
        elif msg_type == "work":
            return await self._handle_worker_result(msg)
        else:
            return await self._handle_direct(msg)

    async def _handle_plan(self, msg: AgentMessage) -> AgentMessage | None:
        """Parse plan into steps and dispatch each to the appropriate worker."""
        self.progress = "Parsing plan..."
        plan_text = msg.content
        steps = self._extract_steps(plan_text)

        if not steps:
            self.progress = "No steps found in plan"
            return AgentMessage(
                sender=self.agent_id,
                recipient="user",
                content=f"**Plan:**\n{plan_text}\n\n_No actionable steps found._",
                task_id=msg.task_id,
                metadata={"type": "final_result"},
            )

        # Dispatch each step to the appropriate agent and collect results
        self._worker_results = []
        results_text = [f"**Plan:**\n{plan_text}\n"]

        for i, step in enumerate(steps):
            target = step.get("agent", "build")
            task_text = step.get("task", "")

            if not self._send_fn:
                continue

            self.progress = f"Step {i+1}/{len(steps)}: → {target}"

            # Send to worker
            step_msg = AgentMessage(
                sender=self.agent_id,
                recipient=target,
                content=task_text,
                task_id=msg.task_id,
                metadata={"type": "step", "step_num": i + 1, "total_steps": len(steps)},
            )
            await self._send_fn(step_msg)

            # Wait for the worker's response via our inbox
            try:
                result_msg = await asyncio.wait_for(
                    self._inbox.get(), timeout=45
                )
                result_content = result_msg.content
            except asyncio.TimeoutError:
                result_content = f"[timeout waiting for {target}]"

            results_text.append(
                f"\n**Step {i+1} ({target}):** {task_text}\n"
                f"**Result:** {result_content}\n"
            )

        self.progress = "All steps complete"
        final = "\n".join(results_text)
        return AgentMessage(
            sender=self.agent_id,
            recipient="user",
            content=final,
            task_id=msg.task_id,
            metadata={"type": "final_result"},
        )

    async def _handle_worker_result(self, msg: AgentMessage) -> AgentMessage | None:
        """Accumulate a worker result."""
        self._worker_results.append({
            "agent": msg.sender,
            "result": msg.content,
        })
        return None

    async def _handle_direct(self, msg: AgentMessage) -> AgentMessage | None:
        """Handle a direct message — use LLM to respond."""
        self.progress = f"Handling from {msg.sender}"
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": msg.content},
        ]
        response = await self.invoke_llm(messages)
        return AgentMessage(
            sender=self.agent_id,
            recipient="user",
            content=response.text,
            task_id=msg.task_id,
            metadata={"type": "final_result"},
        )

    @staticmethod
    def _extract_steps(plan_text: str) -> list[dict[str, str]]:
        """Try to parse JSON steps from plan; fallback to naive line splitting."""
        plan_text = plan_text.strip()

        # Strip markdown code blocks
        if "```" in plan_text:
            parts = plan_text.split("```")
            for part in parts:
                part = part.strip()
                if part.startswith("json"):
                    part = part[4:].strip()
                if part.startswith("{") or part.startswith("["):
                    plan_text = part
                    break

        # Try JSON first
        try:
            data = json.loads(plan_text)
            if isinstance(data, dict) and "steps" in data:
                return data["steps"]
            if isinstance(data, list):
                return data
        except (json.JSONDecodeError, TypeError):
            pass

        # Fallback: numbered lines
        steps = []
        lines = re.split(r'\n\s*[\d\-\*•]+\.?\s*', plan_text)
        for line in lines:
            line = line.strip()
            if line and len(line) > 5:
                steps.append({"agent": "build", "task": line})

        return steps if steps else [{"agent": "build", "task": plan_text}]
