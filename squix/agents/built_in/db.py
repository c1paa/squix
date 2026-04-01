"""Database/Knowledge agent — manages project knowledge and structured data."""

from __future__ import annotations

from squix.agents.base import AgentMessage, BaseAgent


class DatabaseAgent(BaseAgent):
    """Manages project knowledge, documentation, and structured information."""

    agent_id = "DB"
    role = (
        "Knowledge Base — you manage project knowledge and structured data. "
        "Store, retrieve, and organize project information. When asked about "
        "the project, provide summaries and context from what you know."
    )

    async def handle(self, msg: AgentMessage) -> AgentMessage | None:
        await self.set_progress(f"Knowledge lookup: {msg.content[:50]}")
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": msg.content},
        ]
        response = await self.invoke_llm(messages)
        await self.set_progress("Lookup complete")
        return AgentMessage(
            sender=self.agent_id,
            recipient="orch",
            content=response.text,
            task_id=msg.task_id,
            metadata={"type": "work"},
        )
