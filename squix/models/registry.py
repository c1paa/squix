"""Model registry — manages available models, their metadata, and instantiation."""

from __future__ import annotations

import logging
from typing import Any

from squix.models.base import ModelAdapter

logger = logging.getLogger("squix.models.registry")


class ModelRegistry:
    """Keeps track of all registered models and creates adapters on demand."""

    def __init__(
        self,
        model_configs: list[dict[str, Any]],
        paid_model_ok: bool = False,
        **secrets: Any,
    ) -> None:
        """
        Args:
            model_configs: list of model entries from config
            paid_model_ok: whether to include paid models (user-approved)
            secrets: provider credentials (e.g. openrouter_api_key=...)
        """
        self._configs: dict[str, dict[str, Any]] = {}
        self._adapters: dict[str, ModelAdapter] = {}
        self._secrets = secrets
        self._paid_model_ok = paid_model_ok
        self._blocked_paid_models: list[str] = []

        for m in model_configs:
            if m.get("paid", False) and not paid_model_ok:
                self._blocked_paid_models.append(m["id"])
                logger.info(
                    "⛔ Blocked paid model '%s' (set paid_model_ok: true to allow)",
                    m["id"],
                )
                continue
            self._configs[m["id"]] = m

    @property
    def blocked_paid_models(self) -> list[str]:
        """Return list of paid model IDs that were blocked."""
        return list(self._blocked_paid_models)

    def get_model_ids(self) -> list[str]:
        return list(self._configs.keys())

    def get_config(self, model_id: str) -> dict[str, Any] | None:
        return self._configs.get(model_id)

    def find_by_specialization(self, spec: str) -> list[str]:
        """Return model IDs matching a specialization, sorted by priority (lower = better)."""
        matches = []
        for mid, cfg in self._configs.items():
            if spec in cfg.get("specialization", []):
                matches.append((mid, cfg.get("priority", 99)))
        matches.sort(key=lambda x: x[1])
        return [m[0] for m in matches]

    def get_adapter(self, model_id: str) -> ModelAdapter | None:
        if model_id in self._adapters:
            return self._adapters[model_id]

        cfg = self._configs.get(model_id)
        if cfg is None:
            logger.warning("Unknown model_id: %s", model_id)
            return None

        provider = cfg.get("provider", "").lower()

        if provider == "openrouter":
            from squix.api.openrouter import OpenRouterAdapter

            api_key = self._secrets.get("openrouter_api_key", "")
            if not api_key:
                logger.warning("OpenRouter API key not set for %s; using stub adapter", model_id)
                return self._create_stub_adapter(model_id)
            self._adapters[model_id] = OpenRouterAdapter(
                model_id,
                api_key=api_key,
                cost_input_per_1k=cfg.get("cost_per_1k_input", 0),
                cost_output_per_1k=cfg.get("cost_per_1k_output", 0),
            )

        elif provider == "ollama":
            from squix.api.ollama import OllamaAdapter

            raw_id = model_id.split("/", 1)[-1] if "/" in model_id else model_id
            self._adapters[model_id] = OllamaAdapter(raw_id)

        else:
            logger.warning("Unknown provider '%s' for model %s", provider, model_id)
            # Fall through to stub adapter
            return self._create_stub_adapter(model_id)

        return self._adapters[model_id]

    def _create_stub_adapter(self, model_id: str) -> ModelAdapter:
        """Create a stub adapter for local/offline mode (testing)."""
        from squix.models.base import ModelAdapter, ModelResponse

        class StubAdapter(ModelAdapter):
            async def chat(self, messages, temperature=0.7, max_tokens=None, **kwargs):
                return ModelResponse(
                    text="[stub response]",
                    input_tokens=0,
                    output_tokens=0,
                    model_id=model_id,
                    cost=0.0,
                )

            async def health_check(self) -> bool:
                return True  # Always "healthy" in stub mode

        self._adapters[model_id] = StubAdapter(model_id)
        return self._adapters[model_id]

    async def health_check_all(self) -> dict[str, bool]:
        """Check which models/providers are reachable."""
        results: dict[str, bool] = {}
        for model_id in self._configs:
            adapter = self.get_adapter(model_id)
            if adapter:
                results[model_id] = await adapter.health_check()
            else:
                results[model_id] = False
        return results
