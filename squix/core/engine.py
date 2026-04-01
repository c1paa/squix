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
from squix.core.session_context import SessionContext
from squix.memory.manager import MemoryManager
from squix.models.registry import ModelRegistry
from squix.observability.cost_tracker import CostTracker
from squix.observability.logger import SquixLogger
from squix.policy.engine import PolicyEngine
from squix.skills.registry import SkillRegistry
from squix.workspace.manager import WorkspaceManager
from squix.workspace.primary_file_tracker import PrimaryFileTracker

logger = logging.getLogger("squix.core.engine")

# Default timeout for waiting on agent responses (seconds)
STEP_TIMEOUT = 30
DELEGATION_TIMEOUT = 5


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
        self.paid_model_ok = self.config.get("policy", {}).get("paid_model_ok", False)

        # Subsystems
        self.registry = ModelRegistry(
            self.config.get("models", []),
            paid_model_ok=self.paid_model_ok,
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
        self.primary_tracker = PrimaryFileTracker(self.project_dir)
        self.skills = SkillRegistry(workspace=self.workspace, primary_tracker=self.primary_tracker)
        self.session_context = SessionContext()
        self.agents: dict[str, Any] = {}
        self._tasks: list[asyncio.Task] = []
        # Result queue — agents put final results here for engine to collect
        self._result_queue: asyncio.Queue[AgentMessage] = asyncio.Queue()
        # Progress state for live execution UX
        self._current_progress: str = ""
        self._progress_events: list[tuple[str, str, str, str]] = []  # (agent_id, text, task_id, status)
        # Track current input for context update after completion
        self._current_user_input: str = ""

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
            workspace_manager=self.workspace,
            skills=self.skills,
            primary_tracker=self.primary_tracker,
        )
        self.logger.system(f"Agents loaded: {list(self.agents.keys())}")

        # Wire progress callbacks from engine to each agent
        for agent in self.agents.values():
            agent.set_progress_callback(self._on_progress)

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
        """Route a message from one agent to another via their inbox queues.

        Also mirrors delegation messages into the result queue so the UI
        sees the plan immediately without waiting for downstream agents.
        """
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

        # Track delegation in progress state
        self._current_progress = f"Delegating {msg.sender}→{recipient}"

        # Put delegation/forwarding info into the result queue so the UI
        # sees the plan immediately. Use a metadata copy so as not to
        # mutate the original message.
        routing_note = msg.metadata.copy()
        routing_note.update({
            "type": "routing",
            "from": msg.sender,
            "to": recipient,
        })
        await self._result_queue.put(AgentMessage(
            sender=msg.sender,
            recipient="user",
            content=f"→ {recipient}: {msg.content}",
            task_id=msg.task_id,
            metadata=routing_note,
        ))

        await agent.put_message(msg)

    # ------------------------------------------------------------------
    # Main user interaction
    # ------------------------------------------------------------------

    async def process_input(self, user_input: str) -> list[AgentMessage]:
        """Process user input through the Talk agent (auto mode).

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

        # Send to Talk agent with session context
        msg = AgentMessage(
            sender="user",
            recipient="talk",
            content=user_input,
            task_id=task_id,
            metadata={"session_context": self.session_context.format_for_talk()},
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
        except TimeoutError:
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

    # ------------------------------------------------------------------
    # Progress tracking
    # ------------------------------------------------------------------

    async def _on_progress(self, agent_id: str, text: str, task_id: str) -> None:
        """Handle progress update from an agent.

        Stores the current progress text and emits it to the result queue
        so the CLI can display it in real-time.
        """
        self._current_progress = text
        self._progress_events.append((agent_id, text, task_id or "", ""))
        # Emit to result queue for real-time streaming UI
        await self._result_queue.put(AgentMessage(
            sender=agent_id,
            recipient="user",
            content=text,
            task_id=task_id or "",
            metadata={"type": "progress", "agent_id": agent_id},
        ))

    def get_current_progress(self) -> str:
        """Get the latest progress text."""
        return self._current_progress

    def get_progress_events(self) -> list[tuple[str, str, str, str]]:
        """Get all progress events since last clear."""
        events = list(self._progress_events)
        self._progress_events.clear()
        return events

    async def submit_input(self, user_input: str) -> str:
        """Submit input to Talk agent without blocking for results.

        Returns the task_id. The CLI should read from ``_result_queue``
        directly to stream events in real-time.  Session context is
        injected into the message metadata so Talk/Orch/workers see
        what happened earlier in the session.
        """
        task_id = self.session.next_task_id() if self.session else "t001"
        if self.session:
            self.session.add_task(task_id, user_input)
        self.logger.task_started(task_id, user_input)
        self._current_user_input = user_input

        # Drain stale results from previous tasks
        while not self._result_queue.empty():
            try:
                self._result_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        # Clear progress state
        self._progress_events.clear()
        self._current_progress = ""

        # Build message with session context
        msg = AgentMessage(
            sender="user",
            recipient="talk",
            content=user_input,
            task_id=task_id,
            metadata={
                "session_context": self.session_context.format_for_talk(),
            },
        )
        talk_agent = self.agents.get("talk")
        if talk_agent is None:
            await self._result_queue.put(AgentMessage(
                sender="system",
                recipient="user",
                content="[error] Talk agent not available",
                task_id=task_id,
                metadata={"type": "error"},
            ))
            return task_id

        await talk_agent.put_message(msg)
        return task_id

    def update_session_context(
        self,
        response: str,
        files: list[str] | None = None,
        agent_chain: list[str] | None = None,
    ) -> None:
        """Update session context after a task completes.

        Called by the CLI after streaming finishes so subsequent tasks
        have full conversation history.
        """
        self.session_context.add_exchange(
            user_input=self._current_user_input,
            response=response,
            agent_chain=agent_chain,
            files=files,
        )

    async def complete_task(self, task_id: str) -> None:
        """Mark a task as complete in the session."""
        if self.session:
            self.session.complete_task(task_id)
            await self.memory.save_session(self.session)
        self.logger.task_completed(task_id)

    async def chat_only(self, user_input: str) -> list[AgentMessage]:
        """Direct LLM chat response — no delegation, no pipeline.

        Used in Talk Mode: bypasses classification and talks directly.
        """
        task_id = self.session.next_task_id() if self.session else "t001"
        if self.session:
            self.session.add_task(task_id, user_input)

        model_ids = self.registry.get_model_ids()
        model_id = model_ids[0] if model_ids else "default"
        adapter = self.registry.get_adapter(model_id)
        if adapter is None:
            return [AgentMessage(
                sender="system", recipient="user",
                content="[no model available]",
                task_id=task_id, metadata={"type": "error"},
            )]

        import re
        context = self.session_context.format_for_talk()
        system_prompt = (
            "You are Squix — a helpful AI assistant. "
            "Answer concisely and naturally. "
            "If you don't know something, say so briefly."
        )
        if context:
            system_prompt += f"\n\n{context}"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ]

        try:
            resp = await adapter.chat(messages, temperature=0.7, max_tokens=1024)
            text = resp.text.strip()
            text = re.sub(r"^['\"\u201c\u201d]", "", text).strip()
            text = re.sub(r"['\"\u201d]\s*$", "", text).strip()
            # Track cost
            self.cost_tracker.record(
                model_id=model_id,
                input_tokens=resp.input_tokens,
                output_tokens=resp.output_tokens,
                cost=resp.cost,
            )
            if self.session:
                self.session.complete_task(task_id)
                await self.memory.save_session(self.session)
            return [AgentMessage(
                sender="squix", recipient="user",
                content=text, task_id=task_id,
                metadata={"type": "chat", "model": model_id},
            )]
        except Exception as e:
            return [AgentMessage(
                sender="system", recipient="user",
                content=f"[error] {e}", task_id=task_id,
                metadata={"type": "error"},
            )]

    async def plan_only(self, user_input: str) -> list[AgentMessage]:
        """Get plan from Talk agent but do NOT delegate to workers.

        Used in Plan Mode: show what WOULD happen without executing.
        """
        task_id = self.session.next_task_id() if self.session else "t001"
        if self.session:
            self.session.add_task(task_id, user_input)

        # Send to Talk for classification only
        msg = AgentMessage(
            sender="user", recipient="talk",
            content=user_input, task_id=task_id,
        )
        talk_agent = self.agents.get("talk")
        if talk_agent is None:
            return [AgentMessage(
                sender="system", recipient="user",
                content="[error] Talk agent not available",
                task_id=task_id, metadata={"type": "error"},
            )]

        await talk_agent.put_message(msg)

        try:
            results = await asyncio.wait_for(
                self._collect_results_plan_mode(task_id), timeout=STEP_TIMEOUT,
            )
        except TimeoutError:
            results = [AgentMessage(
                sender="system", recipient="user",
                content="[timeout]", task_id=task_id,
                metadata={"type": "error"},
            )]

        # If Talk delegated, annotate: show plan but mark as "not executed"
        final: list[AgentMessage] = []
        for r in results:
            rtype = r.metadata.get("type", "")
            if rtype == "delegate":
                # Talk wanted to delegate — show what it planned
                target = r.recipient
                final.append(AgentMessage(
                    sender="plan", recipient="user",
                    content=f"Would delegate to: {target}\nTask: {r.content}",
                    task_id=task_id,
                    metadata={"type": "result", "plan_only": True},
                ))
                break
            elif rtype == "chat":
                # Talk responded directly — pass through as-is
                final.append(r)
            else:
                final.append(r)

        if self.session:
            self.session.complete_task(task_id)
            await self.memory.save_session(self.session)
        return final

    async def _collect_results_plan_mode(self, task_id: str) -> list[AgentMessage]:
        """Collect results for plan mode — breaks at delegate type.

        Unlike _collect_results, for plan_mode we want to see the delegation
        decision immediately, not wait for the actual worker execution.
        """
        results: list[AgentMessage] = []
        deadline = 5  # short — we just need the classification

        while True:
            try:
                msg = await asyncio.wait_for(
                    self._result_queue.get(), timeout=deadline,
                )
                if msg.task_id and msg.task_id != task_id:
                    continue
                results.append(msg)
                msg_type = msg.metadata.get("type", "")

                # Break at chat, error, final_result, or delegate
                if msg_type in ("chat", "error", "final_result", "delegate"):
                    break

            except TimeoutError:
                if results:
                    break
                raise

        return results

    async def interactive_steps(self, user_input: str) -> list[AgentMessage]:
        """Process input step-by-step, asking confirmation before each worker.

        Used in Interactive Mode: user approves each worker before execution.
        The CLI will display intermediate plans and prompt for y/n.
        For Talk, we still classify first; delegation results indicate what
        steps would run, and the CLI handles the interactive flow separately.
        """
        # For now, same pipeline but results include per-step granularity
        # The CLI will display these and ask the user. Actual per-step
        # confirmation is handled in the CLI (it sees delegation results).
        return await self.process_input(user_input)

    async def _collect_results(self, task_id: str) -> list[AgentMessage]:
        """Collect results from the result queue for a specific task.

        Stops when:
        - A final result (type=final_result) is received
        - A chat response (type=chat) is received
        - An error (type=error) is received
        - Or overall timeout is reached

        NOTE: type=delegate is NOT a stop condition — it means work just started.
        We continue waiting until an actual result or final_result arrives.
        """
        results: list[AgentMessage] = []
        # Total deadline for the whole task (multi-agent pipeline can be slow)
        overall_deadline = 120  # 2 minutes max for complex tasks
        deadline = STEP_TIMEOUT  # initial wait
        routing_received = False
        started = asyncio.get_event_loop().time()

        while True:
            elapsed = asyncio.get_event_loop().time() - started
            remaining = max(0, overall_deadline - elapsed)
            if remaining <= 0:
                break

            try:
                msg = await asyncio.wait_for(
                    self._result_queue.get(),
                    timeout=min(deadline, remaining),
                )
                if msg.task_id and msg.task_id != task_id:
                    continue
                results.append(msg)

                msg_type = msg.metadata.get("type", "")

                if msg_type in ("chat", "error", "final_result"):
                    break

                if msg_type == "delegate":
                    # Delegation means work just started — don't break!
                    routing_received = True
                    deadline = 120  # Full pipeline: worker → orch → user
                    continue

                if msg_type == "routing":
                    routing_received = True
                    deadline = 120  # Allow multi-hop worker→orch→user flow
                    continue

                if msg_type == "result":
                    deadline = 60
                    continue

                deadline = STEP_TIMEOUT

            except TimeoutError:
                if results:
                    break
                raise

        # If we got a routing/delegate notice but no final result, add a summary
        if routing_received and not any(
            r.metadata.get("type") in ("final_result", "chat", "error")
            for r in results
        ):
            results.append(AgentMessage(
                sender="system",
                recipient="user",
                content=(
                    "[task dispatched to worker — processing your request, "
                    "result will follow on next turn]"
                ),
                task_id=task_id,
                metadata={"type": "final_result", "summary": True},
            ))

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
