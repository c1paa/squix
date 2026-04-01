"""Shared fixtures for Squix tests."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from squix.agents.base import AgentMessage
from squix.core.config import load as load_config
from squix.core.engine import SquixEngine
from squix.core.session import Session
from squix.memory.manager import MemoryManager
from squix.models.base import ModelResponse
from squix.models.registry import ModelRegistry
from squix.observability.cost_tracker import CostTracker
from squix.policy.engine import PolicyEngine


@pytest.fixture
def project_dir(tmp_path: Path) -> Path:
    """Create a temporary project directory."""
    return tmp_path


@pytest.fixture
def minimal_config(project_dir: Path) -> dict:
    """Return a minimal config that works with mocks."""
    return {
        "models": [
            {
                "id": "qwen/qwen3-235b-a22b",
                "provider": "openrouter",
                "cost_per_1k_input": 0.001,
                "cost_per_1k_output": 0.002,
                "max_context": 8192,
                "specialization": ["general"],
                "priority": 1,
            }
        ],
        "agents": [
            {
                "id": "talk",
                "type": "talk",
                "enabled": True,
                "model_prefers": ["qwen/qwen3-235b-a22b"],
                "count": 1,
            }
        ],
        "agent_links": {},
        "policy": {"fallback_enabled": False, "escalation_enabled": False},
        "memory": {
            "storage_dir": ".squix",
            "sessions_dir": "sessions",
            "auto_save_interval": 60,
        },
        "observability": {"log_level": "ERROR", "show_costs": False, "show_agent_status": False},
        "workspace": {"output_dir": ".squix/artifacts"},
    }


@pytest.fixture
def mock_adapter():
    """Return a mock model adapter for non-API testing."""
    mock = AsyncMock()
    mock.chat.return_value = ModelResponse(
        text="Hello from mock!",
        input_tokens=10,
        output_tokens=8,
        model_id="mock",
    )
    return mock


@pytest.fixture
def mock_registry(mock_adapter):
    """Return a registry that always returns a mock adapter."""
    reg = AsyncMock(spec=ModelRegistry)
    reg.get_adapter.return_value = mock_adapter
    reg.get_model_ids.return_value = ["mock"]
    return reg


@pytest.fixture
def memory(project_dir: Path) -> MemoryManager:
    """Create a real memory manager using temp directory."""
    return MemoryManager({"storage_dir": ".squix", "sessions_dir": "sessions"}, project_dir)


@pytest.fixture
def policy():
    """Create a real policy engine."""
    return PolicyEngine({"fallback_enabled": False, "escalation_enabled": False})


@pytest.fixture
def cost_tracker() -> CostTracker:
    """Create a fresh cost tracker."""
    return CostTracker()
