"""Skill definitions — the catalog of all available tools."""

from __future__ import annotations

from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class SkillParam:
    name: str
    type: str = "string"
    required: bool = True
    description: str = ""


@dataclass
class SkillDef:
    """Definition of a callable skill that agents can invoke."""
    name: str
    description: str
    allowed_agents: list[str]
    params: list[SkillParam] = field(default_factory=list)
    is_dangerous: bool = False  # write_file, run_command etc.
    is_read_only: bool = True
    is_concurrent_safe: bool = True


# ---------------------------------------------------------------------------
# Skill catalog — keep this as the single source of truth.
# ---------------------------------------------------------------------------

SKILLS: dict[str, SkillDef] = {
    # ── File access ────────────────────────────────────────────────────
    "read_file": SkillDef(
        name="read_file",
        description=(
            "Read the full contents of a file in the current project. "
            "If the file does not exist, returns an error. "
            "Always use this before editing."
        ),
        params=[
            SkillParam("path", description="Relative path inside the project, e.g. main.py"),
        ],
        allowed_agents=["build", "debug", "README", "web", "DB", "AI", "idea", "product", "plan", "orch"],
    ),

    "write_file": SkillDef(
        name="write_file",
        description=(
            "Write content to a file in the project. "
            "Creates parent directories automatically. "
            "Overwrites the file if it already exists. "
            "Use this to create or completely replace a file."
        ),
        params=[
            SkillParam("path", description="Relative path inside the project"),
            SkillParam("content", description="Full file content to write"),
        ],
        allowed_agents=["build", "debug", "README"],
        is_dangerous=True,
        is_read_only=False,
        is_concurrent_safe=False,
    ),

    "edit_file": SkillDef(
        name="edit_file",
        description=(
            "Replace exact text in a file. You MUST read_file first. "
            "Finds old_string in the file and replaces it with new_string. "
            "If old_string appears more than once, set replace_all=true. "
            "Fails if old_string is not found — use read_file to check first."
        ),
        params=[
            SkillParam("path", description="File to edit"),
            SkillParam("old_string", description="Exact text to find in the file"),
            SkillParam("new_string", description="Replacement text"),
            SkillParam("replace_all", type="boolean", required=False,
                       description="Replace all occurrences (default false)"),
        ],
        allowed_agents=["build", "debug"],
        is_dangerous=True,
        is_read_only=False,
        is_concurrent_safe=False,
    ),

    "patch_file": SkillDef(
        name="patch_file",
        description=(
            "Apply a small patch (diff/unified format) to an existing file. "
            "Use this for incremental changes instead of rewriting the whole file. "
            "If the tool is not available, fall back to write_file."
        ),
        params=[
            SkillParam("path", description="Path of the file to patch"),
            SkillParam("patch", description="Unified diff or full replacement text"),
        ],
        allowed_agents=["build", "debug"],
        is_dangerous=True,
        is_read_only=False,
        is_concurrent_safe=False,
    ),

    "list_files": SkillDef(
        name="list_files",
        description=(
            "List files and directories in the project. "
            "Returns a tree-like view up to max_depth."
        ),
        params=[
            SkillParam("path", description="Root path, default '.'", required=False),
            SkillParam("max_depth", description="How deep to traverse", required=False),
        ],
        allowed_agents=["build", "debug", "README", "plan", "orch", "idea", "product", "web", "DB", "AI", "talk"],
    ),

    "search_in_files": SkillDef(
        name="search_in_files",
        description=(
            "Search for a pattern/regex across all files in the project. "
            "Returns matching file paths and line numbers."
        ),
        params=[
            SkillParam("query", description="Text or regex pattern to search"),
            SkillParam("glob", description="File glob filter, e.g. *.py", required=False),
        ],
        allowed_agents=["debug", "build", "DB"],
    ),

    # ── Code execution ─────────────────────────────────────────────────
    "run_command": SkillDef(
        name="run_command",
        description=(
            "Execute a shell command in the project directory. "
            "Returns exit code, stdout, stderr. "
            "Use carefully — commands run in the same environment."
        ),
        params=[
            SkillParam("command", description="Shell command to run"),
        ],
        allowed_agents=["debug", "build"],
        is_dangerous=True,
        is_read_only=False,
        is_concurrent_safe=False,
    ),

    "run_tests": SkillDef(
        name="run_tests",
        description=(
            "Run the project's test suite (pytest, make test, etc.) "
            "and return the output."
        ),
        params=[],
        allowed_agents=["debug", "build"],
    ),

    # ── Context helpers ────────────────────────────────────────────────
    "get_project_structure": SkillDef(
        name="get_project_structure",
        description=(
            "Return a concise tree view of the project: files, README, config."
        ),
        params=[],
        allowed_agents=["debug", "build", "plan", "orch", "idea", "product", "README", "AI", "web", "DB", "talk"],
    ),

    "find_main_file": SkillDef(
        name="find_main_file",
        description=(
            "Heuristically identify the most likely main/active source file "
            "in the project. Returns the path. Use this when the user says "
            "'my code', 'the main file', 'fix the file I was editing'."
        ),
        params=[],
        allowed_agents=["debug", "build", "plan", "orch", "talk"],
    ),

    # ── Memory / knowledge ─────────────────────────────────────────────
    "save_memory": SkillDef(
        name="save_memory",
        description=(
            "Persist a key-value fact about the project for future reference. "
            "Use this to remember decisions, file locations, architecture notes."
        ),
        params=[
            SkillParam("key", description="Unique key name"),
            SkillParam("value", description="Content to store"),
        ],
        allowed_agents=["DB", "plan", "orch"],
        is_read_only=False,
        is_concurrent_safe=False,
    ),

    "load_memory": SkillDef(
        name="load_memory",
        description="Retrieve a previously saved memory fact by key.",
        params=[
            SkillParam("key", description="Key to retrieve"),
        ],
        allowed_agents=["DB", "plan", "orch", "debug", "build"],
    ),

    # ── Web / external ─────────────────────────────────────────────────
    "search_web": SkillDef(
        name="search_web",
        description=(
            "Search the web for external information. "
            "Use when the project needs docs, tutorials, API references."
        ),
        params=[
            SkillParam("query", description="Search query"),
        ],
        allowed_agents=["web"],
    ),

    # ── Git operations ────────────────────────────────────────────────
    "git_status": SkillDef(
        name="git_status",
        description="Show current git status (modified, staged, untracked files).",
        params=[],
        allowed_agents=["build", "debug", "plan", "orch"],
    ),

    "git_diff": SkillDef(
        name="git_diff",
        description="Show git diff of changes. Use staged=true for staged changes only.",
        params=[
            SkillParam("staged", type="boolean", required=False,
                       description="Show only staged changes"),
        ],
        allowed_agents=["build", "debug"],
    ),

    "git_add": SkillDef(
        name="git_add",
        description="Stage a file for commit.",
        params=[
            SkillParam("path", description="File path to stage"),
        ],
        allowed_agents=["build", "debug"],
        is_dangerous=True,
        is_read_only=False,
    ),

    "git_commit": SkillDef(
        name="git_commit",
        description="Create a git commit with the given message.",
        params=[
            SkillParam("message", description="Commit message"),
        ],
        allowed_agents=["build", "debug"],
        is_dangerous=True,
        is_read_only=False,
        is_concurrent_safe=False,
    ),
}


# ── Helper: build agent-permission map ──────────────────────────────────

def skills_for_agent(agent_id: str) -> list[SkillDef]:
    """Return the list of skills an agent is allowed to use."""
    return list(SKILLS.values())
