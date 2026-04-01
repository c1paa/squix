"""Core engine — the main brain that ties everything together."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from squix.agents.factory import AgentFactory
from squix.core.config import load as load_config
from squix.core.session import Session
from squix.memory.manager import MemoryManager
from squix.models.registry import ModelRegistry
from squix.observability.cost_tracker import CostTracker
from squix.observability.logger import SquixLogger
from squix.policy.engine import PolicyEngine
from squix.workspace.manager import WorkspaceManager

logger = logging.getLogger("squix.core.engine")


class SquixEngine:
    """Main engine — loads config, bootstraps agents, runs the orchestration loop."""

    def __init__(
        self,
        project_dir: Path | None = None,
        config_path: str | None = None,
        secrets: dict[str, str] | None = None,
    ) -> None:
        self.project_dir = project_dir or Path.cwd()
        self.config = load_config(config_path)
        self.secrets = secrets or {}

        # Subsystems
        self.registry = ModelRegistry(
            self.config.get("models", []),
            **self.secrets,
        )
        self.policy = PolicyEngine(self.config.get("policy", {}))
        self.memory = MemoryManager(self.config.get("memory", {}), self.project_dir)
        self.logger = SquixLogger(self.config.get("observability", {}))
        self.cost_tracker = CostTracker()
        self.workspace = WorkspaceManager(self.project_dir, self.config.get("workspace", {}))
        self.session: Session | None = None

        # Agent infrastructure
        self.agent_factory = AgentFactory(
            self.config.get("agents", []),
            self.config.get("agent_links", {}),
        )
        self.agents: dict[str, Any] = {}
        self._tasks: list[asyncio.Task] = []

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def startup(self) -> None:
        """Boot the entire system: load memory, create agents, start tasks."""
        self.logger.system("Squix starting up…")
        self.logger.system(f"Project dir: {self.project_dir}")

        # Prepare workspace
        self.workspace.init()

        # Restore session/state if exists
        self.session = await self.memory.load_session()
        if self.session:
            logger.info("Restored session: %s", self.session.session_id)
        else:
            self.session = await self.memory.create_session()

        # Create all agents
        self.agents = self.agent_factory.create_all(self.registry)
        self.logger.system(f"Agents loaded: {list(self.agents.keys())}")

        # Start agent run loops
        for agent in self.agents.values():
            task = asyncio.create_task(agent.run())
            self._tasks.append(task)

        # Health check providers
        health = await self.registry.health_check_all()
        for mid, ok in health.items():
            status = "OK" if ok else "UNREACHABLE"
            self.logger.system(f"  Model {mid}: {status}")

        self.logger.system("Squix ready.")

    async def shutdown(self) -> None:
        """Save state and stop all agents."""
        self.logger.system("Shutting down…")
        if self.session:
            await self.memory.save_session(self.session)

        # Cancel agent tasks gracefully
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        self.logger.system("Squix stopped.")

    # ------------------------------------------------------------------
    # Task submission
    # ------------------------------------------------------------------

    async def submit_task(self, user_input: str) -> dict[str, Any]:
        """Submit a user task and run the full pipeline: plan → orchestrate → execute."""
        task_id = self.session.next_task_id() if self.session else "t001"

        self.logger.task_started(task_id, user_input)

        # Step 1: Planner breaks it down
        from squix.agents.base import AgentMessage

        plan_msg = AgentMessage(
            sender="user",
            recipient="plan",
            content=user_input,
            task_id=task_id,
        )
        await self.agents["plan"].put_message(plan_msg)

        # Step 2: Wait for plan to come back from orchestrator
        result = await self._run_dispatch_loop(task_id, user_input)

        self.logger.task_completed(task_id)
        return result

    async def _run_dispatch_loop(self, task_id: str, user_input: str) -> dict[str, Any]:
        """Internal dispatch loop: plan → parse steps → send to workers → collect results."""
        from squix.agents.base import AgentMessage

        # This is a simplified synchronous-ish dispatch for MVP.
        # In a fuller version the orchestrator would manage the entire flow.
        results: list[dict[str, Any]] = []

        # Ask planner — it returns a plan message
        plan_response = await self.agents["plan"].handle(
            AgentMessage(sender="user", recipient="plan", content=user_input, task_id=task_id)
        )

        if not plan_response:
            return {"error": "Planner returned no plan", "task_id": task_id}

        # Parse the plan
        plan_text = plan_response.content
        self.logger.event("plan_created", {"task_id": task_id, "plan": plan_text[:200]})

        # Parse steps from the plan (try JSON, fallback to text)
        steps = self._extract_steps(plan_text)

        # Dispatch each step to the right agent
        for step in steps:
            target = step.get("agent", "build")
            task_text = step.get("task", "")

            if target not in self.agents:
                self.logger.warning(f"Agent '{target}' not found, defaulting to 'build'")
                target = "build"

            self.logger.event("dispatch", {
                "task_id": task_id,
                "step": step,
                "target": target,
            })

            # Pick model for this agent via policy
            model_id = self.policy.select_model(target, self.registry)

            # Ask the agent
            agent = self.agents[target]
            agent_msg = AgentMessage(
                sender="orch",
                recipient=target,
                content=task_text,
                task_id=task_id,
            )
            response = await agent.handle(agent_msg)

            if response:
                # Track cost if LLM call happened
                if "llm_messages" in response.metadata:
                    # In real flow, the adapter would be called here
                    pass
                results.append({
                    "agent": target,
                    "response": response.content,
                    "model": model_id,
                })

        return {
            "task_id": task_id,
            "plan": plan_text,
            "results": results,
        }

    @staticmethod
    def _extract_steps(plan_text: str) -> list[dict[str, str]]:
        """Try to parse JSON steps from plan; fallback to naive line splitting."""
        import json
        import re

        # Try JSON first
        try:
            data = json.loads(plan_text)
            if isinstance(data, dict) and "steps" in data:
                return data["steps"]
            if isinstance(data, list):
                return data
        except (json.JSONDecodeError, TypeError):
            pass

        # Fallback: numbered lines or bullet points
        steps = []
        lines = re.split(r'\n\s*[\d\-\*•]+\.?\s*', plan_text)
        for line in lines:
            line = line.strip()
            if line and len(line) > 5:
                steps.append({"agent": "build", "task": line})

        return steps if steps else [{"agent": "build", "task": plan_text}]

    def get_status(self) -> dict[str, Any]:
        """Return current system status for CLI display."""
        agent_states = {}
        for aid, agent in self.agents.items():
            agent_states[aid] = {
                "state": agent.state.value,
                "progress": agent.progress,
            }
        return {
            "session": self.session.session_id if self.session else None,
            "agents": agent_states,
            "total_cost": self.cost_tracker.total_cost,
            "tasks_completed": self.session.tasks_completed if self.session else 0,
        }
