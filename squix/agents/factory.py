"""Agent factory — creates agents from config with wiring."""

from __future__ import annotations

from typing import Any

from squix.agents.base import BaseAgent
from squix.agents.built_in.ai import AISpecialistAgent
from squix.agents.built_in.build import BuilderAgent
from squix.agents.built_in.db import DatabaseAgent
from squix.agents.built_in.debug import DebuggerAgent
from squix.agents.built_in.idea import IdeaAgent
from squix.agents.built_in.orch import OrchestratorAgent
from squix.agents.built_in.plan import PlannerAgent
from squix.agents.built_in.product import ProductAgent
from squix.agents.built_in.readme import ReadmeAgent
from squix.agents.built_in.web import WebAgent
from squix.models.registry import ModelRegistry

_AGENT_MAP: dict[str, type[BaseAgent]] = {
    "orch": OrchestratorAgent,
    "plan": PlannerAgent,
    "build": BuilderAgent,
    "debug": DebuggerAgent,
    "web": WebAgent,
    "DB": DatabaseAgent,
    "AI": AISpecialistAgent,
    "idea": IdeaAgent,
    "product": ProductAgent,
    "README": ReadmeAgent,
}


class AgentFactory:
    """Creates agents from config and wires up communication links."""

    def __init__(
        self,
        agent_configs: list[dict[str, Any]],
        agent_links: dict[str, list[str]],
    ) -> None:
        self._configs: dict[str, dict[str, Any]] = {}
        for ac in agent_configs:
            self._configs[ac["id"]] = ac
        self._links = agent_links

    def create_all(self, registry: ModelRegistry) -> dict[str, BaseAgent]:
        """Create all enabled agents."""
        agents: dict[str, BaseAgent] = {}
        for aid, cfg in self._configs.items():
            if not cfg.get("enabled", True):
                continue
            cls = _AGENT_MAP.get(aid)
            if cls is None:
                continue

            count = cfg.get("count", 1)
            for i in range(count):
                agent_id = f"{aid}.{i}" if count > 1 else aid
                neighbors = self._links.get(aid, [])
                agent = cls(
                    agent_id=agent_id,
                    model_prefers=cfg.get("model_prefers", []),
                    neighbors=neighbors,
                )
                agents[agent_id] = agent

        return agents
