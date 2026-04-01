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
        self.progress = f"Brainstorming: {msg.content[:50]}"
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": msg.content},
        ]
        response = await self.invoke_llm(messages)
        self.progress = "Ideas ready"

        # If delegated from talk, send result back to user
        if msg.metadata.get("original_sender") == "user":
            return AgentMessage(
                sender=self.agent_id,
                recipient="user",
                content=response.text,
                task_id=msg.task_id,
                metadata={"type": "result"},
            )

        return AgentMessage(
            sender=self.agent_id,
            recipient="orch",
            content=response.text,
            task_id=msg.task_id,
            metadata={"type": "work"},
        )
