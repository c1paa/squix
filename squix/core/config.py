"""Configuration loader — merges default + user YAML configs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_DEFAULT_CONFIG_NAME = "default_config.yml"

# Default config lives in the squix package directory
_package_dir = Path(__file__).resolve().parent.parent
_default_path = _package_dir / _DEFAULT_CONFIG_NAME


def load_default() -> dict[str, Any]:
    """Load the built-in default configuration."""
    with open(_default_path) as f:
        return yaml.safe_load(f)


def load_user(path: str | Path | None = None) -> dict[str, Any]:
    """Load user override config from a file path.

    If *path* is None, looks for ``squix.yml`` in the current working
    directory.  Returns an empty dict if the file does not exist.
    """
    path = Path.cwd() / "squix.yml" if path is None else Path(path)

    if not path.exists():
        return {}

    with open(path) as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def load(overrides: str | Path | None = None) -> dict[str, Any]:
    """Load merged configuration.

    User settings deeply override the defaults (dict merge).
    """
    default = load_default()
    user = load_user(overrides)

    if user:
        _deep_merge(default, user)

    return default


def _deep_merge(base: dict, override: dict) -> None:
    """Merge *override* into *base* in-place (recursive)."""
    for key, value in override.items():
        if (
            key in base
            and isinstance(base[key], dict)
            and isinstance(value, dict)
        ):
            _deep_merge(base[key], value)
        else:
            base[key] = value
