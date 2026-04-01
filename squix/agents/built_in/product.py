"""Product Manager agent — turns ideas into products, plans features and UX."""

from __future__ import annotations

from squix.agents.base import AgentMessage, BaseAgent


class ProductAgent(BaseAgent):
    """Turns ideas into products — plans features, UX, and roadmaps."""

    agent_id = "product"
    role = (
        "Product Manager — you turn ideas into real products. Plan features, "
        "define UX, create roadmaps, and think about user value. Be practical "
        "and strategic."
    )

    async def handle(self, msg: AgentMessage) -> AgentMessage | None:
        self.progress = f"Product analysis: {msg.content[:50]}"
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": msg.content},
        ]
        response = await self.invoke_llm(messages)
        self.progress = "Product analysis complete"

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
