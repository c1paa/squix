"""Idea agent — brainstorms and discusses project ideas."""

from __future__ import annotations

from squix.agents.base import AgentMessage, BaseAgent


class IdeaAgent(BaseAgent):
    """Discusses project ideas, proposes alternatives, brainstorms."""

    agent_id = "idea"
    role = (
        "Idea Explorer — you brainstorm, discuss project ideas, and propose "
        "creative alternatives. Think outside the box. Help shape the product vision."
    )

    async def handle(self, msg: AgentMessage) -> AgentMessage | None:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": msg.content},
        ]
        return AgentMessage(
            sender=self.agent_id,
            recipient="orch",
            content=f"[idea] {msg.content}",
            task_id=msg.task_id,
            metadata={"type": "work", "llm_messages": messages},
        )
