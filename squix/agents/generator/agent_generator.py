"""Agent generator — create, clone, enable/disable agents per project."""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from string import Template
from typing import Any

import yaml

logger = logging.getLogger("squix.agents.generator")

TEMPLATE_BUILDER = Template("""\
from squix.agents.base import BaseAgent, AgentMessage


class ${class_name}(BaseAgent):
    agent_id = "${agent_id}"
    role = "${role}"

    async def handle(self, msg: AgentMessage) -> AgentMessage | None:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": msg.content},
        ]
        return AgentMessage(
            sender=self.agent_id,
            recipient="orch",
            content=f"[${agent_id}] {msg.content}",
            task_id=msg.task_id,
            metadata={"type": "work", "llm_messages": messages},
        )
""")


class AgentGenerator:
    """Allows users to create, clone, enable/disable agents at runtime."""

    def __init__(self, config_path: Path | None = None) -> None:
        self.config_path = config_path

    def create_custom_agent(
        self,
        agent_id: str,
        role: str,
        model_prefers: list[str] | None = None,
        neighbors: list[str] | None = None,
        system_prompt: str | None = None,
        output_dir: Path | None = None,
    ) -> dict[str, Any]:
        """Generate a new custom agent module and return its config."""
        class_name = agent_id.capitalize().replace("_", "") + "Agent"

        code = TEMPLATE_BUILDER.substitute(
            class_name=class_name,
            agent_id=agent_id,
            role=role,
        )

        if output_dir is None:
            output_dir = Path(__file__).parent.parent / "agents" / "custom"
        output_dir.mkdir(parents=True, exist_ok=True)

        agent_file = output_dir / f"{agent_id}.py"
        agent_file.write_text(code, encoding="utf-8")

        config_entry = {
            "id": agent_id,
            "name": class_name,
            "role": role,
            "model_prefers": model_prefers or [],
            "enabled": True,
            "custom": True,
            "neighbors": neighbors or [],
        }

        logging.info("Created custom agent: %s at %s", agent_id, agent_file)
        return config_entry

    def clone_agent(
        self,
        source_id: str,
        new_id: str,
        config_path: Path,
    ) -> dict[str, Any]:
        """Clone an existing agent from the config (e.g., more builders)."""
        with open(config_path) as f:
            config = yaml.safe_load(f)

        source = None
        for ac in config.get("agents", []):
            if ac["id"] == source_id:
                source = ac
                break

        if source is None:
            raise ValueError(f"Agent '{source_id}' not found in config")

        new_config = copy.deepcopy(source)
        new_config["id"] = new_id
        new_config["count"] = new_config.get("count", 1) + 1
        new_config["name"] = f"{source.get('name', new_id)} (clone)"

        # Update links if the source was in the communication graph
        for ac in config.get("agents", []):
            links = config.get("agent_links", {}).get(ac["id"], [])
            if source_id in links and new_id not in links:
                links.append(new_id)

        # Add the new agent config entry
        config["agents"].append(new_config)

        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        logger.info("Cloned agent %s → %s", source_id, new_id)
        return new_config

    def toggle_agent(self, config_path: Path, agent_id: str, enabled: bool) -> None:
        """Enable or disable an agent in the config file."""
        with open(config_path) as f:
            config = yaml.safe_load(f)

        for ac in config.get("agents", []):
            if ac["id"] == agent_id:
                ac["enabled"] = enabled
                logger.info("Agent %s: enabled=%s", agent_id, enabled)
                break

        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    def list_config(self, config_path: Path) -> list[dict[str, Any]]:
        """List all agents and their status from a config file."""
        with open(config_path) as f:
            config = yaml.safe_load(f)
        return config.get("agents", [])
