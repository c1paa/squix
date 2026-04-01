"""Project scanner — analyzes the project directory for /init command."""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

logger = logging.getLogger("squix.core.init_scanner")

# Map file extensions → language
_EXT_LANG: dict[str, str] = {
    ".py": "Python",
    ".js": "JavaScript",
    ".ts": "TypeScript",
    ".tsx": "TypeScript (React)",
    ".jsx": "JavaScript (React)",
    ".rs": "Rust",
    ".go": "Go",
    ".java": "Java",
    ".kt": "Kotlin",
    ".rb": "Ruby",
    ".php": "PHP",
    ".c": "C",
    ".cpp": "C++",
    ".h": "C/C++ Header",
    ".cs": "C#",
    ".swift": "Swift",
    ".dart": "Dart",
    ".lua": "Lua",
    ".sh": "Shell",
    ".yml": "YAML",
    ".yaml": "YAML",
    ".json": "JSON",
    ".toml": "TOML",
    ".md": "Markdown",
    ".html": "HTML",
    ".css": "CSS",
    ".scss": "SCSS",
    ".sql": "SQL",
}

# Config files that identify frameworks/tools
_FRAMEWORK_FILES: dict[str, str] = {
    "pyproject.toml": "Python (pyproject)",
    "setup.py": "Python (setup.py)",
    "requirements.txt": "Python (requirements)",
    "Pipfile": "Python (Pipenv)",
    "package.json": "Node.js",
    "tsconfig.json": "TypeScript",
    "Cargo.toml": "Rust",
    "go.mod": "Go",
    "pom.xml": "Java (Maven)",
    "build.gradle": "Java (Gradle)",
    "Gemfile": "Ruby",
    "composer.json": "PHP",
    "Makefile": "Make",
    "Dockerfile": "Docker",
    "docker-compose.yml": "Docker Compose",
    ".github": "GitHub Actions",
    ".gitignore": "Git",
    "next.config.js": "Next.js",
    "next.config.ts": "Next.js",
    "vite.config.ts": "Vite",
    "webpack.config.js": "Webpack",
    "tailwind.config.js": "Tailwind CSS",
    "tailwind.config.ts": "Tailwind CSS",
    ".env": "Environment config",
    ".env.example": "Environment config",
}

# Directories to skip
_SKIP_DIRS = {
    ".git", ".squix", "node_modules", "__pycache__", ".venv", "venv",
    ".mypy_cache", ".pytest_cache", ".ruff_cache", "dist", "build",
    ".next", ".nuxt", "target", ".tox", "env", ".env",
}


class ProjectScanner:
    """Scans a project directory and produces a profile."""

    def __init__(self, project_dir: Path) -> None:
        self.project_dir = project_dir

    def scan(self) -> dict[str, Any]:
        """Run a full scan and return project profile."""
        files = self._collect_files()

        # Language stats
        lang_counter: Counter = Counter()
        total_lines = 0
        for f in files:
            ext = f.suffix.lower()
            lang = _EXT_LANG.get(ext)
            if lang:
                lang_counter[lang] += 1
            try:
                lines = len(f.read_text(errors="replace").splitlines())
                total_lines += lines
            except Exception:
                pass

        # Framework detection
        frameworks = []
        for fname, framework in _FRAMEWORK_FILES.items():
            if (self.project_dir / fname).exists():
                frameworks.append(framework)

        # Primary language
        primary_lang = lang_counter.most_common(1)[0][0] if lang_counter else "Unknown"

        # Find key files
        has_readme = any(
            (self.project_dir / name).exists()
            for name in ("README.md", "README.rst", "README.txt", "README")
        )
        has_tests = any(
            f.name.startswith("test_") or f.name.endswith("_test.py")
            or "tests" in f.parts or "test" in f.parts
            for f in files
        )
        has_docs = (self.project_dir / "docs").is_dir()
        has_ci = (
            (self.project_dir / ".github" / "workflows").is_dir()
            or (self.project_dir / ".gitlab-ci.yml").exists()
        )

        profile = {
            "project_name": self.project_dir.name,
            "primary_language": primary_lang,
            "languages": dict(lang_counter.most_common(10)),
            "frameworks": frameworks,
            "total_files": len(files),
            "total_lines": total_lines,
            "has_readme": has_readme,
            "has_tests": has_tests,
            "has_docs": has_docs,
            "has_ci": has_ci,
        }

        return profile

    def scan_and_save(self) -> dict[str, Any]:
        """Scan and save profile to .squix/project_profile.json."""
        profile = self.scan()
        squix_dir = self.project_dir / ".squix"
        squix_dir.mkdir(parents=True, exist_ok=True)
        profile_path = squix_dir / "project_profile.json"
        with open(profile_path, "w") as f:
            json.dump(profile, f, indent=2)
        logger.info("Project profile saved to %s", profile_path)
        return profile

    def _collect_files(self) -> list[Path]:
        """Collect all files, skipping ignored directories."""
        files = []
        for item in self.project_dir.rglob("*"):
            # Skip ignored directories
            if any(part in _SKIP_DIRS for part in item.parts):
                continue
            if item.is_file():
                files.append(item)
        return files

    @staticmethod
    def format_profile(profile: dict[str, Any]) -> str:
        """Format profile as a human-readable summary."""
        lines = [
            f"Project: {profile['project_name']}",
            f"Language: {profile['primary_language']}",
            f"Files: {profile['total_files']}  |  Lines: {profile['total_lines']:,}",
        ]

        if profile.get("frameworks"):
            lines.append(f"Stack: {', '.join(profile['frameworks'])}")

        langs = profile.get("languages", {})
        if langs:
            lang_parts = [f"{lang} ({count})" for lang, count in list(langs.items())[:5]]
            lines.append(f"Languages: {', '.join(lang_parts)}")

        flags = []
        if profile.get("has_readme"):
            flags.append("README")
        if profile.get("has_tests"):
            flags.append("Tests")
        if profile.get("has_docs"):
            flags.append("Docs")
        if profile.get("has_ci"):
            flags.append("CI/CD")
        if flags:
            lines.append(f"Has: {', '.join(flags)}")

        return "\n".join(lines)
