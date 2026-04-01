"""Workspace manager — file operations, code execution, artifact storage."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger("squix.workspace")


class WorkspaceManager:
    """Manages the project workspace: file I/O, code execution, and artifacts."""

    def __init__(
        self,
        project_dir: Path,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.project_dir = project_dir
        self.config = config or {}
        self.artifacts_dir = project_dir / self.config.get("artifacts_dir", ".squix/artifacts")

    def init(self) -> None:
        """Create workspace directories."""
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

    # ---- File operations ----

    def read_file(self, path: str) -> str:
        """Read a file safely, resolving relative to project_dir."""
        full = self._resolve(path)
        if not full.exists():
            raise FileNotFoundError(f"File not found: {full}")
        return full.read_text(encoding="utf-8", errors="replace")

    def write_file(self, path: str, content: str) -> Path:
        """Write content to a file, creating parent dirs as needed."""
        full = self._resolve(path)
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text(content, encoding="utf-8")
        logger.info("Wrote file: %s", full)
        return full

    def list_files(self, path: str = ".", max_depth: int = 3) -> list[str]:
        """List files in a directory (relative to project dir)."""
        full = self._resolve(path)
        if not full.is_dir():
            return [str(full)]
        result = []
        for f in full.rglob("*"):
            rel = f.relative_to(self.project_dir)
            depth = len(rel.parts)
            if depth <= max_depth:
                result.append(str(rel))
        return sorted(result)

    def save_artifact(self, name: str, content: str, task_id: str = "") -> Path:
        """Save an artifact (result, file snippet, etc.) to the artifacts directory."""
        task_folder = self.artifacts_dir / (task_id or "misc")
        task_folder.mkdir(exist_ok=True)
        path = task_folder / name
        path.write_text(content, encoding="utf-8")
        return path

    # ---- Code execution ----

    async def run_command(
        self,
        command: str | list[str],
        timeout: int = 60,
        cwd: str | None = None,
    ) -> tuple[int, str, str]:
        """Run a shell command safely in the workspace.

        Returns: (return_code, stdout, stderr)
        """
        work_dir = self._resolve(cwd) if cwd else self.project_dir
        cmds = ["bash", "-c", command] if isinstance(command, str) else command

        logger.info("Executing: %s in %s", cmds, work_dir)

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmds,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=work_dir,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )
            return (
                proc.returncode or 0,
                stdout.decode("utf-8", errors="replace"),
                stderr.decode("utf-8", errors="replace"),
            )
        except TimeoutError:
            proc.kill()
            return -1, "", f"Command timed out after {timeout}s"
        except Exception as e:
            logger.exception("Command execution failed")
            return -1, "", str(e)

    async def run_python(self, code: str, timeout: int = 30) -> tuple[int, str, str]:
        """Execute Python code and return output."""
        return await self.run_command(
            ["python", "-c", code],
            timeout=timeout,
        )

    def _resolve(self, path: str | Path) -> Path:
        """Resolve a path relative to the project directory."""
        p = Path(path)
        if p.is_absolute():
            return p
        return (self.project_dir / p).resolve()
