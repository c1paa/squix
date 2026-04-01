"""Squix — main entry point.

Usage:
    python -m squix.main              # run from any project directory
    squix                             # after pip install

Optional environment variables:
    SQUIX_CONFIG      — path to a user config file (default: squix.yml in cwd)
    SQUIX_DEBUG       — set to 1 to enable debug logging
    OPENROUTER_API_KEY — OpenRouter API key (required for OpenRouter models)
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path


def main() -> None:
    """CLI entry point."""
    project_dir = Path.cwd()
    config_path = os.environ.get("SQUIX_CONFIG")
    debug = os.environ.get("SQUIX_DEBUG", "0") == "1"

    # Configure root logging
    log_level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    )

    # Gather secrets from environment (accept both OPENROUTER_API_KEY and OPENAI_API_KEY)
    secrets = {}
    api_key = (
        os.environ.get("OPENROUTER_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
    )
    if api_key:
        secrets["openrouter_api_key"] = api_key

    # Build and run the engine + CLI
    from squix.core.engine import SquixEngine
    from squix.ui.cli import SquixCLI

    engine = SquixEngine(
        project_dir=project_dir,
        config_path=config_path,
        secrets=secrets,
    )

    cli = SquixCLI(engine)

    try:
        asyncio.run(_run(engine, cli))
    except KeyboardInterrupt:
        asyncio.run(engine.shutdown())
        sys.exit(0)


async def _run(engine, cli) -> None:
    """Bootstrap engine and start interactive CLI."""
    await engine.startup()
    await cli.repl()


# Make `python -m squix.main` work
if __name__ == "__main__":
    main()
