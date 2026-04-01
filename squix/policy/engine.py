"""Policy engine — routing rules, fallback, and escalation logic."""

from __future__ import annotations

import logging
from typing import Any

from squix.models.registry import ModelRegistry

logger = logging.getLogger("squix.policy")


class PolicyEngine:
    """Decides which model to use for a given agent/task, handles fallback and escalation."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}
        self.cheap_threshold = self.config.get("cheap_model_threshold", 500)
        self.fallback_enabled = self.config.get("fallback_enabled", True)
        self.max_retries = self.config.get("max_retries", 3)
        self.escalation_enabled = self.config.get("escalation_enabled", True)
        self.escalation_models = self.config.get("escalation_models", [])

    def select_model(
        self,
        agent_id: str,
        registry: ModelRegistry,
        *,
        prefer_cheap: bool = False,
        escalated: bool = False,
    ) -> str | None:
        """Select the best model for a given agent.

        Returns a model_id or None if no model is available.
        """
        if escalated and self.escalation_models:
            for mid in self.escalation_models:
                adapter = registry.get_adapter(mid)
                if adapter is not None:
                    return mid

        # Fallback: return first available model
        model_ids = registry.get_model_ids()
        for mid in model_ids:
            adapter = registry.get_adapter(mid)
            if adapter is not None:
                return mid

        return None

    def select_model_for_agent(
        self,
        model_prefers: list[str],
        registry: ModelRegistry,
        *,
        prefer_cheap: bool = False,
    ) -> str | None:
        """Pick the first available model from the agent's preference list."""
        for mid in model_prefers:
            adapter = registry.get_adapter(mid)
            if adapter is not None:
                return mid

        # Fallback to any available model
        model_ids = registry.get_model_ids()
        for mid in model_ids:
            adapter = registry.get_adapter(mid)
            if adapter is not None:
                logger.warning(
                    "No preferred model available, falling back to %s", mid
                )
                return mid

        return None

    def should_fallback(self, error: str, attempt: int) -> bool:
        """Check if we should retry with a different model."""
        if not self.fallback_enabled:
            return False
        return attempt < self.max_retries

    def next_fallback_model(
        self,
        current_model: str,
        model_prefers: list[str],
        registry: ModelRegistry,
    ) -> str | None:
        """Get the next available model for fallback (skipping the failed one)."""
        for mid in model_prefers:
            if mid != current_model:
                adapter = registry.get_adapter(mid)
                if adapter is not None:
                    return mid

        # Try any model in registry
        for mid in registry.get_model_ids():
            if mid != current_model:
                adapter = registry.get_adapter(mid)
                if adapter is not None:
                    return mid

        return None

    def needs_escalation(self, confidence: float | None) -> bool:
        """Decide if a task needs escalation to a more powerful model."""
        if not self.escalation_enabled:
            return False
        return bool(confidence is not None and confidence < 0.3)

    def get_escalation_model(self) -> str | None:
        """Return the top escalation model."""
        if self.escalation_models:
            return self.escalation_models[0]
        return None
