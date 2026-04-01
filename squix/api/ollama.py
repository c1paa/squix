"""Ollama (local) API adapter."""

from __future__ import annotations

import logging
from typing import Any

import httpx

from squix.models.base import ModelAdapter, ModelResponse

logger = logging.getLogger("squix.api.ollama")


class OllamaAdapter(ModelAdapter):
    """Adapter for local Ollama instances."""

    def __init__(
        self,
        model_id: str,
        *,
        base_url: str = "http://localhost:11434",
        **kwargs: Any,
    ) -> None:
        super().__init__(model_id, **kwargs)
        self.base_url = base_url
        self._session = httpx.AsyncClient(base_url=base_url, timeout=300.0)

    async def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        # Ollama chat endpoint
        payload: dict[str, Any] = {
            "model": self.model_id.split("/", 1)[-1] if "/" in self.model_id else self.model_id,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
            },
        }
        if max_tokens is not None:
            payload["options"]["num_predict"] = max_tokens

        logger.debug("Ollama request → model=%s", self.model_id)

        resp = await self._session.post("/api/chat", json=payload)
        resp.raise_for_status()
        data = resp.json()

        text = data.get("message", {}).get("content", "")
        # Ollama doesn't always give token counts
        input_tokens = data.get("prompt_eval_count", 0)
        output_tokens = data.get("eval_count", 0)

        return ModelResponse(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model_id=self.model_id,
            cost=0.0,  # local, free
            metadata=data,
        )

    async def health_check(self) -> bool:
        try:
            resp = await self._session.get("/api/tags")
            return resp.status_code == 200
        except Exception:
            logger.exception("Ollama health check failed")
            return False
