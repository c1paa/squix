"""Core engine — the main brain that ties everything together."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from squix.agents.base import AgentMessage, AgentState
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

# Default timeout for waiting on agent responses (seconds)
_STEP_TIMEOUT = 60


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
        # Result queue — agents put final results here for engine to collect
        self._result_queue: asyncio.Queue[AgentMessage] = asyncio.Queue()
        # Event callback for UI to show real-time agent activity
        self._on_agent_event: Any = None

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

        # Configure file logging → session dir
        mem = self.config.get("memory", {}).get("storage_dir", ".squix")
        log_dir = (
            Path(self.project_dir) / mem / "sessions" / self.session.session_id
        )
        log_path = log_dir / "squix.log"
        self.logger.configure(log_path)

        # Create all agents with full wiring
        self.agents = self.agent_factory.create_all(
            registry=self.registry,
            send_fn=self._route_message,
            result_queue=self._result_queue,
            cost_tracker=self.cost_tracker,
            policy=self.policy,
        )
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
    # Message routing
    # ------------------------------------------------------------------

    async def _route_message(self, msg: AgentMessage) -> None:
        """Route a message from one agent to another via their inbox queues."""
        recipient = msg.recipient
        if recipient == "user":
            await self._result_queue.put(msg)
            return

        agent = self.agents.get(recipient)
        if agent is None:
            logger.warning("No agent '%s' found, routing to result queue", recipient)
            msg.metadata["routing_error"] = f"Agent '{recipient}' not found"
            await self._result_queue.put(msg)
            return

        self.logger.event("route", {
            "from": msg.sender,
            "to": recipient,
            "task_id": msg.task_id,
            "preview": msg.content[:100],
        })

        # Notify UI about agent activity
        if self._on_agent_event:
            self._on_agent_event(msg.sender, recipient, "routing")

        await agent.put_message(msg)

    # ------------------------------------------------------------------
    # Main user interaction
    # ------------------------------------------------------------------

    async def process_input(self, user_input: str) -> list[AgentMessage]:
        """Process user input through the Talk agent and collect all results.

        Talk classifies the input and either responds directly or delegates
        to other agents. Results flow back through the result queue.
        """
        task_id = self.session.next_task_id() if self.session else "t001"

        # Track in session
        if self.session:
            self.session.add_task(task_id, user_input)

        self.logger.task_started(task_id, user_input)

        # Drain any stale results from previous tasks
        while not self._result_queue.empty():
            try:
                self._result_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        # Send to Talk agent
        msg = AgentMessage(
            sender="user",
            recipient="talk",
            content=user_input,
            task_id=task_id,
        )
        talk_agent = self.agents.get("talk")
        if talk_agent is None:
            return [AgentMessage(
                sender="system",
                recipient="user",
                content="[error] Talk agent not available",
                task_id=task_id,
                metadata={"type": "error"},
            )]

        await talk_agent.put_message(msg)

        # Collect results — wait for responses with timeout
        results: list[AgentMessage] = []
        try:
            results = await self._collect_results(task_id)
        except asyncio.TimeoutError:
            results.append(AgentMessage(
                sender="system",
                recipient="user",
                content="[timeout] Task took too long to complete",
                task_id=task_id,
                metadata={"type": "error"},
            ))

        # Complete task in session
        if self.session:
            self.session.complete_task(task_id)
            await self.memory.save_session(self.session)

        self.logger.task_completed(task_id)
        return results

    async def _collect_results(self, task_id: str) -> list[AgentMessage]:
        """Collect results from the result queue for a specific task.

        Waits for results with a timeout. Stops when:
        - A final result (type=chat or type=result) is received
        - Or timeout is reached
        """
        results: list[AgentMessage] = []
        # Keep collecting until we get a final result or timeout
        deadline = _STEP_TIMEOUT
        while True:
            try:
                msg = await asyncio.wait_for(
                    self._result_queue.get(), timeout=deadline
                )
                # Only collect messages for this task (or accept all if task_id empty)
                if msg.task_id and msg.task_id != task_id:
                    continue
                results.append(msg)

                msg_type = msg.metadata.get("type", "")

                # If it's a direct chat response from talk, we're done
                if msg_type == "chat":
                    break

                # If it's an error, we're done
                if msg_type == "error":
                    break

                # If it's a final result from orch, we're done
                if msg_type == "final_result":
                    break

                # If it's a delegation result (intermediate), continue collecting
                # but with shorter timeout for subsequent results
                if msg_type == "result":
                    # This is a worker result — might be followed by more
                    # Give 5s for more results, then stop
                    deadline = 5
                    continue

                # For other types, keep waiting with shorter timeout
                deadline = _STEP_TIMEOUT

            except asyncio.TimeoutError:
                if results:
                    break
                raise

        return results

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def get_status(self) -> dict[str, Any]:
        """Return current system status for CLI display."""
        agent_states = {}
        for aid, agent in self.agents.items():
            agent_states[aid] = {
                "state": agent.state.value,
                "progress": agent.progress,
                "model": agent.model_prefers[0] if agent.model_prefers else "—",
                "neighbors": agent.neighbors,
            }
        return {
            "session": self.session.session_id if self.session else None,
            "agents": agent_states,
            "total_cost": self.cost_tracker.total_cost,
            "tasks_completed": self.session.tasks_completed if self.session else 0,
        }

    def get_active_chain(self) -> list[str]:
        """Return list of currently working agents (for UI display)."""
        return [
            aid for aid, agent in self.agents.items()
            if agent.state == AgentState.WORKING
        ]
