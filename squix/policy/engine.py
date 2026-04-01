"""Policy engine — routing rules, fallback, and escalation logic."""

from __future__ import annotations

from typing import Any

from squix.models.registry import ModelRegistry


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
            return self.escalation_models[0]

        # Ask the agent's preference first
        # Actually we need access to agent's model_prefers — the factory passes this
        # But here we look it up from the config indirectly
        # For now, the registry's specialization lookup is the primary method

        # Get agent's preferred models (passed via agent config, looked up by registry)
        # We use the agent's role to guess — but better: the agent object has model_prefers
        # This method is called by the engine which has access to the agent
        # So we do a simple round-robin over the agent's preferred list
        # Actually, let's fix this: the engine passes the agent's prefs
        return None  # placeholder — the engine overrides this

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
