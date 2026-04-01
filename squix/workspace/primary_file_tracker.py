"""PrimaryFileTracker — remembers the user's main working file.

Behaviour:
  - Every read_file / write_file call from any skill updates the tracker.
  - The most recent file is considered the "primary" file.
  - When the user says 'my code' / 'the file I was editing', use this.
  - On startup, load any remembered state from .squix/primary_file.json.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class PrimaryFileTracker:
    """Tracks the most recently accessed / modified project file."""

    def __init__(self, project_dir: Path, history_size: int = 10) -> None:
        self._project_dir = project_dir
        self._history_size = history_size
        # Ordered list: most recent first
        self._history: list[dict[str, Any]] = []
        self._current: str | None = None
        # Load persisted state
        self._load()

    def track_access(self, path: str) -> None:
        """Record that an agent read/accessed *path*."""
        entry = {"path": path, "action": "access", "ts": time.time()}
        self._push(entry)

    def track_write(self, path: str) -> None:
        """Record that an agent modified *path*."""
        entry = {"path": path, "action": "write", "ts": time.time()}
        self._current = path
        self._push(entry)

    def get_primary(self) -> str | None:
        """Return the most likely primary file."""
        if self._current:
            return self._current
        if self._history:
            return self._history[0]["path"]
        return None

    def get_history(self) -> list[dict[str, Any]]:
        return list(self._history)

    def save(self) -> None:
        """Persist state to disk."""
        store = {
            "current": self._current,
            "history": self._history,
        }
        p = self._project_dir / ".squix" / "primary_file.json"
        p.parent.mkdir(exist_ok=True)
        p.write_text(json.dumps(store, indent=2), "utf-8")

    # ── Internals ───────────────────────────────────────────────────────

    def _push(self, entry: dict[str, Any]) -> None:
        # Avoid duplicates at the top
        if self._history and self._history[0]["path"] == entry["path"]:
            # Just update timestamp
            self._history[0]["ts"] = entry["ts"]
            return
        self._history.insert(0, entry)
        self._history = self._history[: self._history_size]
        self.save()

    def _load(self) -> None:
        p = self._project_dir / ".squix" / "primary_file.json"
        try:
            data = json.loads(p.read_text("utf-8"))
            self._current = data.get("current")
            self._history = data.get("history", [])[: self._history_size]
        except Exception:
            pass
