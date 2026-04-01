"""Orchestrator agent — the central task dispatcher managed by Planner."""

from __future__ import annotations

from squix.agents.base import AgentMessage, BaseAgent


class OrchestratorAgent(BaseAgent):
    """Receives steps from Planner and routes them to the appropriate worker agents."""

    agent_id = "orch"
    role = "Orchestrator — dispatch tasks to workers and coordinate the team."

    async def handle(self, msg: AgentMessage) -> AgentMessage | None:
        # Orchestrator receives a plan from 'plan' or a delegation directive
        self.progress = f"Orchestrating: {msg.content[:80]}"
        # For now, return a message that the core engine will parse
        # to dispatch to workers
        return AgentMessage(
            sender=self.agent_id,
            recipient="system",  # handled by core dispatch loop
            content=msg.content,
            task_id=msg.task_id,
            metadata={"type": "dispatch_plan", "raw_plan": msg.content},
        )
