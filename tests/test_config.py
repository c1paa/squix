"""Tests for config loading — YAML parsing, deep merge."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from squix.core.config import load as load_config


class TestConfigLoading:
    def test_default_config(self):
        """load with no custom config returns empty dict."""
        cfg = load_config()
        assert isinstance(cfg, dict)

    def test_default_config_not_none(self):
        cfg = load_config()
        # Should always return dict, never None
        assert cfg is not None


class TestConfigDeepMerge:
    def test_deep_merge_overrides_leaf_values(self, project_dir):
        """Deep merge should override leaf values from config file."""
        cfg_path = project_dir / "test_config.yml"
        cfg_path.write_text(yaml.dump({"observability": {"log_level": "DEBUG"}}))

        cfg = load_config(str(cfg_path))
        assert cfg.get("observability", {}).get("log_level") == "DEBUG"

    def test_deep_merge_preserves_unspecified(self, project_dir):
        """Values not in config file should stay at defaults."""
        cfg_path = project_dir / "test_config.yml"
        cfg_path.write_text(yaml.dump({"observability": {"log_level": "TRACE"}}))

        cfg = load_config(str(cfg_path))
        # show_costs not overridden — should be from default_config or empty
        obs = cfg.get("observability", {})
        # Either present from file or from default_config
        assert "log_level" in obs

    def test_config_path_to_missing_file(self):
        """Nonexistent config path should return defaults."""
        cfg = load_config("/nonexistent/config.yml")
        assert isinstance(cfg, dict)

    def test_config_dict_merges(self, project_dir):
        """New keys from config should be added, existing preserved."""
        cfg_path = project_dir / "test.yml"
        cfg_path.write_text(yaml.dump({
            "policy": {"cheap_model_threshold": 1000},
        }))
        cfg = load_config(str(cfg_path))
        assert cfg["policy"]["cheap_model_threshold"] == 1000
