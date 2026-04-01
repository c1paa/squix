"""Abstract base class for all model adapters (providers)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ModelResponse:
    """Standardized response from any model call."""

    text: str
    input_tokens: int
    output_tokens: int
    model_id: str
    cost: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class ModelAdapter(ABC):
    """Unified interface for any AI model provider (OpenRouter, Ollama, …)."""

    def __init__(self, model_id: str, **kwargs: Any) -> None:
        self.model_id = model_id
        self.kwargs = kwargs

    @abstractmethod
    async def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        """Send a chat completion request and return :class:`ModelResponse`."""
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        """Return True if the model/provider is reachable."""
        ...
