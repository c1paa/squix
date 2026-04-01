"""Web researcher agent — searches for info online."""

from __future__ import annotations

from squix.agents.base import AgentMessage, BaseAgent


class WebAgent(BaseAgent):
    """Searches the web for information, documentation, and references."""

    agent_id = "web"
    role = (
        "Web Researcher — you search for and summarize web information. "
        "Provide concise answers with key facts. When asked to look something up, "
        "search, summarize the top findings, and cite sources if available."
    )

    async def handle(self, msg: AgentMessage) -> AgentMessage | None:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": msg.content},
        ]
        return AgentMessage(
            sender=self.agent_id,
            recipient="orch",
            content=f"[web] {msg.content}",
            task_id=msg.task_id,
            metadata={"type": "work", "llm_messages": messages},
        )
