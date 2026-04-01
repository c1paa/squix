"""Skill registry — callable skill instances wired to the workspace."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from squix.skills.definitions import SKILLS, SkillDef

logger = logging.getLogger("squix.skills.registry")


class SkillRegistry:
    """Central registry that executes skills on behalf of agents.

    Usage:
        skills = SkillRegistry(workspace=ws)
        result = await skills.call("read_file", {"path": "main.py"}, agent="debug")
    """

    def __init__(self, workspace: Any = None, primary_tracker: Any = None) -> None:
        self._workspace = workspace
        self._primary = primary_tracker
        self._skill_log: list[dict[str, Any]] = []  # audit trail

    # ── Public API ──────────────────────────────────────────────────────

    def get_skill_def(self, name: str) -> SkillDef | None:
        return SKILLS.get(name)

    def is_allowed(self, skill_name: str, agent_id: str) -> bool:
        sdef = SKILLS.get(skill_name)
        if sdef is None:
            return False
        return agent_id in sdef.allowed_agents

    def list_allowed(self, agent_id: str) -> list[SkillDef]:
        return [s for s in SKILLS.values() if agent_id in s.allowed_agents]

    async def call(
        self,
        skill_name: str,
        params: dict[str, Any],
        agent_id: str = "",
    ) -> Any:
        """Call a skill by name. Raises PermissionError / ValueError."""
        sdef = SKILLS.get(skill_name)
        if sdef is None:
            raise ValueError(f"Unknown skill: {skill_name}")
        if agent_id and agent_id not in sdef.allowed_agents:
            logger.warning(
                "Agent %s denied skill %s", agent_id, skill_name,
            )
            raise PermissionError(
                f"Agent '{agent_id}' is not allowed to use '{skill_name}'",
            )

        # Audit log
        self._skill_log.append({
            "skill": skill_name,
            "agent": agent_id,
            "params": {k: (v[:200] if isinstance(v, str) and len(v) > 200 else v)
                       for k, v in params.items()},
        })

        executor = getattr(self, f"_exec_{skill_name}", None)
        if executor is None:
            raise NotImplementedError(f"Skill '{skill_name}' has no executor")

        return await executor(params)

    @property
    def history(self) -> list[dict[str, Any]]:
        return list(self._skill_log)

    # ── Executors ───────────────────────────────────────────────────────

    async def _exec_read_file(self, params: dict[str, Any]) -> dict[str, Any]:
        if not self._workspace:
            return {"error": "Workspace not available"}
        path = params["path"]
        try:
            content = self._workspace.read_file(path)
            # Track as primary file
            if self._primary:
                self._primary.track_access(path)
            return {"status": "success", "content": content, "path": path}
        except FileNotFoundError:
            return {"status": "error", "error": f"File not found: {path}", "path": path}
        except Exception as e:
            return {"status": "error", "error": str(e), "path": path}

    async def _exec_write_file(self, params: dict[str, Any]) -> dict[str, Any]:
        if not self._workspace:
            return {"error": "Workspace not available"}
        path = params["path"]
        content = params["content"]
        try:
            written = self._workspace.write_file(path, content)
            if self._primary:
                self._primary.track_write(path)
            return {"status": "success", "path": path, "lines": content.count("\n") + 1}
        except Exception as e:
            return {"status": "error", "error": str(e), "path": path}

    async def _exec_patch_file(self, params: dict[str, Any]) -> dict[str, Any]:
        if not self._workspace:
            return {"error": "Workspace not available"}
        path = params["path"]
        patch = params.get("patch", "")
        try:
            # First try to read existing file
            try:
                existing = self._workspace.read_file(path)
            except FileNotFoundError:
                existing = ""

            # If patch is a unified diff, apply it; otherwise treat as replacement
            if patch.startswith("---") or patch.startswith("diff "):
                result = self._apply_unified_diff(existing, patch)
            else:
                result = patch  # full replacement

            self._workspace.write_file(path, result)
            if self._primary:
                self._primary.track_write(path)
            return {"status": "success", "path": path}
        except Exception as e:
            return {"status": "error", "error": str(e), "path": path}

    async def _exec_list_files(self, params: dict[str, Any]) -> dict[str, Any]:
        if not self._workspace:
            return {"error": "Workspace not available"}
        path = params.get("path", ".")
        max_depth = int(params.get("max_depth", 2))
        files = self._workspace.list_files(path, max_depth=max_depth)
        return {"status": "success", "files": files}

    async def _exec_search_in_files(self, params: dict[str, Any]) -> dict[str, Any]:
        if not self._workspace:
            return {"error": "Workspace not available"}
        query = params["query"]
        glob_pat = params.get("glob", "")
        try:
            regex = re.compile(query, re.IGNORECASE)
        except re.error:
            regex = re.compile(re.escape(query), re.IGNORECASE)

        matches: list[dict[str, Any]] = []
        files = self._workspace.list_files(max_depth=5)
        for fpath in files:
            if glob_pat and not Path(fpath).match(glob_pat):
                continue
            try:
                content = self._workspace.read_file(fpath)
            except Exception:
                continue
            for i, line in enumerate(content.split("\n"), 1):
                if regex.search(line):
                    matches.append({"file": fpath, "line": i, "text": line.strip()})
                    if len(matches) >= 100:
                        break
            if len(matches) >= 100:
                break
        return {"status": "success", "matches": matches, "count": len(matches)}

    async def _exec_run_command(self, params: dict[str, Any]) -> dict[str, Any]:
        if not self._workspace:
            return {"error": "Workspace not available"}
        command = params["command"]
        try:
            rc, stdout, stderr = await self._workspace.run_command(command, timeout=60)
            return {"status": "success", "returncode": rc, "stdout": stdout, "stderr": stderr}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    async def _exec_run_tests(self, params: dict[str, Any]) -> dict[str, Any]:
        if not self._workspace:
            return {"error": "Workspace not available"}
        # Try common test commands
        cmds = [
            ["python", "-m", "pytest", "-x", "-q", "--tb=short"],
            ["make", "test"],
        ]
        for cmd in cmds:
            rc, stdout, stderr = await self._workspace.run_command(cmd, timeout=120)
            if rc == 0 or "no test" not in stderr.lower():
                return {
                    "status": "success",
                    "command": " ".join(cmd),
                    "returncode": rc,
                    "stdout": stdout[:4000],
                    "stderr": stderr[:4000],
                }
        return {
            "status": "success",
            "command": "none found",
            "returncode": 1,
            "stdout": "",
            "stderr": "No test runner found.",
        }

    async def _exec_get_project_structure(self, params: dict[str, Any]) -> dict[str, Any]:
        if not self._workspace:
            return {"error": "Workspace not available"}
        files = self._workspace.list_files(max_depth=3)
        structure = "\n".join(files[:100])
        # Try README for extra context
        readme = ""
        for name in ["README.md", "README"]:
            try:
                readme = self._workspace.read_file(name)[:1000]
                break
            except Exception:
                pass
        return {
            "status": "success",
            "structure": structure,
            "readme": readme,
            "files_count": len(files),
        }

    async def _exec_find_main_file(self, params: dict[str, Any]) -> dict[str, Any]:
        if not self._workspace:
            return {"error": "Workspace not available"}

        # Priority tracker → recently modified files
        if self._primary:
            pf = self._primary.get_primary()
            if pf and (self._workspace.project_dir / pf).is_file():
                return {"status": "success", "path": pf, "reason": "primary_tracker"}

        # Heuristic scan
        files = self._workspace.list_files(max_depth=3)
        candidates: list[tuple[int, str]] = []

        # 1) Known entry point names
        entry_names = {
            "main.py", "app.py", "index.py", "server.py", "__main__.py",
            "main.js", "app.js", "index.js", "server.js",
            "main.ts", "app.ts", "index.ts",
            "main.go", "main.rs", "program.java", "main.c", "main.cpp",
        }
        for f in files:
            fname = Path(f).name
            if fname in entry_names:
                candidates.append((10, f))

        # 2) pyproject.toml / package.json → infer entry
        cfg_files = [f for f in files if Path(f).name in ("pyproject.toml", "package.json", "Cargo.toml")]
        for cf in sorted(cfg_files):
            try:
                content = self._workspace.read_file(cf)
                # Look for patterns like "main" or "script"
                for kw in ["main", "script", "entry"]:
                    idx = content.find(kw)
                    if idx != -1:
                        chunk = content[idx:idx+100]
                        for token in re.findall(r'[\w./_\\-]+\.\w{2,}', chunk):
                            candidates.append((5, token.strip("\"'")))
            except Exception:
                pass

        # 3) Most recently modified .py / .js / .ts
        source_exts = (".py", ".js", ".ts", ".go", ".rs")
        src = [f for f in files if any(f.endswith(e) for e in source_exts)]
        if src:
            candidates.append((1, src[0]))

        if candidates:
            candidates.sort(key=lambda x: -x[0])
            best = candidates[0][1]
            return {"status": "success", "path": best, "reason": "heuristic_scan"}

        # 4) Any source file
        if src:
            return {"status": "success", "path": src[0], "reason": "first_source"}

        return {"status": "not_found", "error": "No main file found"}

    async def _exec_save_memory(self, params: dict[str, Any]) -> dict[str, Any]:
        key = params["key"]
        value = params["value"]
        if self._workspace:
            mem_path = self._workspace.project_dir / ".squix" / "memory.json"
            mem_path.parent.mkdir(exist_ok=True)
            import json
            mem = {}
            if mem_path.exists():
                try:
                    mem = json.loads(mem_path.read_text("utf-8"))
                except Exception:
                    mem = {}
            mem[key] = value
            mem_path.write_text(json.dumps(mem, ensure_ascii=False, indent=2), "utf-8")
        return {"status": "success", "key": key}

    async def _exec_load_memory(self, params: dict[str, Any]) -> dict[str, Any]:
        key = params["key"]
        if self._workspace:
            mem_path = self._workspace.project_dir / ".squix" / "memory.json"
            if mem_path.exists():
                import json
                try:
                    mem = json.loads(mem_path.read_text("utf-8"))
                    return {"status": "success", "key": key, "value": mem.get(key, "")}
                except Exception:
                    pass
        return {"status": "not_found", "key": key, "value": ""}

    async def _exec_search_web(self, params: dict[str, Any]) -> dict[str, Any]:
        query = params.get("query", "")
        return {"status": "stub", "query": query, "results":[]}

    # ── Helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _apply_unified_diff(original: str, diff_text: str) -> str:
        """Very basic unified diff applier. Falls back to replacement."""
        # For a proper implementation, use `patch` or python-diff library.
        # For now, if it looks like a diff, try a line-by-line apply.
        # If parsing fails, just return the diff as a full replacement.
        lines = original.split("\n")
        result: list[str] = list(lines)

        current_line = 0
        for line in diff_text.split("\n"):
            if line.startswith("---") or line.startswith("+++") or line.startswith("@@"):
                m = re.match(r"@@\s*-\d+(?:,\d+)?\s*\+(\d+)", line)
                if m:
                    current_line = int(m.group(1)) - 1
                continue
            if line.startswith("+"):
                content = line[1:]
                if current_line < len(result):
                    result.insert(current_line, content)
                else:
                    result.append(content)
                current_line += 1
            elif line.startswith("-"):
                if 0 <= current_line < len(result):
                    result.pop(current_line)
            elif not line.startswith("\\"):
                current_line += 1

        return "\n".join(result)
