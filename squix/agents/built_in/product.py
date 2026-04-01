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
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": msg.content},
        ]
        return AgentMessage(
            sender=self.agent_id,
            recipient="orch",
            content=f"[product] {msg.content}",
            task_id=msg.task_id,
            metadata={"type": "work", "llm_messages": messages},
        )
