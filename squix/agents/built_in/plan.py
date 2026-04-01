"""Planner agent — breaks large tasks into actionable steps for the orchestrator."""

from __future__ import annotations

from squix.agents.base import AgentMessage, BaseAgent


class PlannerAgent(BaseAgent):
    """Receives the user's high-level task and produces a structured plan
    with step-by-step instructions, then hands control to Orchestrator."""

    agent_id = "plan"
    role = "Task Planner — break big problems into actionable steps."

    async def handle(self, msg: AgentMessage) -> AgentMessage | None:
        # Ask the LLM to produce a JSON plan
        prompt = (
            f"{self.system_prompt}\n\n"
            f"The user submitted this task:\n{msg.content}\n\n"
            "Break it down into numbered steps. For each step, specify:\n"
            "  - which agent should handle it (build, debug, web, etc.)\n"
            "  - what the expected output is\n"
            "Return ONLY a JSON object with key 'steps' containing a list of "
            '{"agent": "...", "task": "..."} objects.'
        )
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": msg.content},
        ]
        return AgentMessage(
            sender=self.agent_id,
            recipient="orch",
            content=prompt,
            task_id=msg.task_id,
            metadata={"type": "plan", "llm_messages": messages},
        )
