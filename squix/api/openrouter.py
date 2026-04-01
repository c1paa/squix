"""OpenRouter API adapter."""

from __future__ import annotations

import logging
from typing import Any

import httpx

from squix.models.base import ModelAdapter, ModelResponse

logger = logging.getLogger("squix.api.openrouter")


class OpenRouterAdapter(ModelAdapter):
    """Adapter for the OpenRouter API."""

    API_URL = "https://openrouter.ai/api/v1"

    def __init__(self, model_id: str, *, api_key: str, **kwargs: Any) -> None:
        super().__init__(model_id, **kwargs)
        self.api_key = api_key
        self._session = httpx.AsyncClient(
            base_url=self.API_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "HTTP-Referer": "https://github.com/squix-ai/squix",
                "X-Title": "Squix",
            },
            timeout=120.0,
        )
        # Parse cost from model_id metadata (loaded via registry)
        self._cost_input_per_1k: float = kwargs.get("cost_input_per_1k", 0.0)
        self._cost_output_per_1k: float = kwargs.get("cost_output_per_1k", 0.0)

    async def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        payload: dict[str, Any] = {
            "model": self.model_id,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        payload.update(kwargs)

        logger.debug("OpenRouter request → model=%s tokens~=%s",
                      self.model_id, len(messages))

        resp = await self._session.post("/chat/completions", json=payload)
        resp.raise_for_status()
        data = resp.json()

        choice = data["choices"][0]
        usage = data.get("usage", {})
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)

        cost = (
            (input_tokens / 1000) * self._cost_input_per_1k
            + (output_tokens / 1000) * self._cost_output_per_1k
        )

        return ModelResponse(
            text=choice["message"]["content"],
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model_id=self.model_id,
            cost=cost,
            metadata=data,
        )

    async def health_check(self) -> bool:
        try:
            resp = await self._session.get("/auth/key")
            return resp.status_code == 200
        except Exception:
            logger.exception("OpenRouter health check failed")
            return False
