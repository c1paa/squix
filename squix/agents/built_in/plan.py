"""Planner agent — breaks large tasks into actionable steps for the orchestrator."""

from __future__ import annotations

import json
import logging

from squix.agents.base import AgentMessage, BaseAgent

logger = logging.getLogger("squix.agent.plan")


class PlannerAgent(BaseAgent):
    """Receives the user's high-level task and produces a structured plan
    with step-by-step instructions, then hands control to Orchestrator."""

    agent_id = "plan"
    role = "Task Planner — break big problems into actionable steps."

    async def handle(self, msg: AgentMessage) -> AgentMessage | None:
        self.progress = "Creating plan..."

        # Ask the LLM to produce a JSON plan
        instruction = (
            "Break this task into numbered steps. For each step specify:\n"
            "  - which agent should handle it: build (code), debug (fix bugs), "
            "web (search info), DB (knowledge), AI (ML tasks), README (docs)\n"
            "  - a clear task description for that agent\n\n"
            "Return ONLY a JSON object with key 'steps' containing a list of "
            '{"agent": "...", "task": "..."} objects.\n'
            "Example:\n"
            '{"steps": [{"agent": "build", "task": "Create a sort function"}, '
            '{"agent": "debug", "task": "Test the sort function"}]}'
        )
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"{msg.content}\n\n{instruction}"},
        ]
        response = await self.invoke_llm(messages)

        self.progress = "Plan created, sending to orchestrator"

        # Send the plan to orch
        return AgentMessage(
            sender=self.agent_id,
            recipient="orch",
            content=response.text,
            task_id=msg.task_id,
            metadata={
                "type": "plan",
                "original_input": msg.content,
            },
        )
