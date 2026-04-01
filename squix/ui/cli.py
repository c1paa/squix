"""CLI layer — interactive session, commands, rich TUI with rich library."""

from __future__ import annotations

import asyncio
import sys
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

console = Console()

class SquixCLI:
    """Interactive CLI interface for Squix."""

    def __init__(self, engine: Any) -> None:
        self.engine = engine
        self.running = False

    # ------------------------------------------------------------------ #
    #  Banner
    # ------------------------------------------------------------------ #

    def print_banner(self) -> None:
        console.print(Panel(
            "[bold cyan]Squix[/bold cyan] — [dim]AI Agent OS v0.1.0[/dim]\n"
            "[dim]Orchestrate AI models and agents as a managed team[/dim]",
            border_style="cyan",
        ))

    # ------------------------------------------------------------------ #
    #  Interactive loop
    # ------------------------------------------------------------------ #

    async def repl(self) -> None:
        """Start the interactive REPL loop."""
        self.running = True
        self.print_banner()

        # Show initial status
        self._show_status()

        console.print("\n[dim]Type your task, or use commands:[/dim]")
        console.print("  [bold]/status[/bold]   — show agents & system state")
        console.print("  [bold]/cost[/bold]     — show cost breakdown")
        console.print("  [bold]/models[/bold]   — list available models")
        console.print("  [bold]/agents[/bold]   — list agents & their roles")
        console.print("  [bold]/history[/bold]  — show task history")
        console.print("  [bold]/help[/bold]     — show all commands")
        console.print("  [bold]/quit[/bold]     — exit Squix")
        console.print()

        while self.running:
            try:
                user_input = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: Prompt.ask("[bold green]squix[/bold green]")
                )
            except (EOFError, KeyboardInterrupt):
                break

            user_input = user_input.strip()
            if not user_input:
                continue

            if user_input.startswith("/"):
                await self._handle_command(user_input)
            else:
                await self._handle_task(user_input)

    async def _handle_command(self, cmd: str) -> None:
        """Process a slash-command."""
        parts = cmd.split(maxsplit=1)
        command = parts[0].lower()

        commands = {
            "/status": self._show_status,
            "/cost": self._show_cost,
            "/models": self._show_models,
            "/agents": self._show_agents,
            "/history": self._show_history,
            "/help": self._show_help,
            "/quit": self._quit,
            "/exit": self._quit,
            "/q": self._quit,
        }

        handler = commands.get(command)
        if handler:
            result = handler()
            if asyncio.iscoroutine(result):
                await result
        else:
            console.print(f"[yellow]Unknown command: {command}[/yellow]")
            console.print("Type [bold]/help[/bold] for available commands.")

    async def _handle_task(self, text: str) -> None:
        """Submit a user task to the engine."""
        try:
            result = await self.engine.submit_task(text)

            if "error" in result:
                console.print(f"[red]Error: {result['error']}[/red]")
                return

            # Show the plan
            plan = result.get("plan", "")
            console.print(f"\n[bold blue]📋 Plan:[/bold blue]\n{plan[:500]}")

            # Show results
            results = result.get("results", [])
            if results:
                console.print("\n[bold green]📝 Results:[/bold green]")
                for r in results:
                    console.print(
                        Panel(
                            r.get("response", "")[:300],
                            title=f"[cyan]{r.get('agent', '?')}[/cyan] "
                                  f"[dim]({r.get('model', '?')})[/dim]",
                            border_style="green",
                        )
                    )

            # Final cost
            status = self.engine.get_status()
            status.get("total_cost", 0)
            if self.engine.cost_tracker.total_cost > 0:
                summary = self.engine.cost_tracker.summary()
                console.print(f"\n[yellow]💰 Total cost: ${summary['total_cost']:.4f}[/yellow]")

        except Exception as e:
            console.print(f"[bold red]✗ Task failed: {e}[/bold red]")
            self.engine.logger.error(str(e))

    # ------------------------------------------------------------------ #
    #  Status / info panels
    # ------------------------------------------------------------------ #

    def _show_status(self) -> None:
        """Print current system status."""
        status = self.engine.get_status()
        table = Table(title="System Status")
        table.add_column("Agent", style="cyan")
        table.add_column("State", style="green")
        table.add_column("Progress")

        for aid, info in status.get("agents", {}).items():
            table.add_row(aid, info["state"], info.get("progress", ""))

        table.add_row(
            "---", "---",
            f"Tasks: {status.get('tasks_completed', 0)} | "
            f"Cost: ${status.get('total_cost', 0):.4f}",
        )
        console.print(table)

    def _show_cost(self) -> None:
        """Print cost breakdown."""
        summary = self.engine.cost_tracker.summary()
        if not summary["by_model"]:
            console.print("[dim]No costs recorded yet.[/dim]")
            return

        table = Table(title=f"Cost Breakdown — Total: ${summary['total_cost']:.4f}")
        table.add_column("Model", style="cyan")
        table.add_column("Cost ($)", justify="right")
        table.add_column("Input Tokens", justify="right")
        table.add_column("Output Tokens", justify="right")

        for mid, info in summary["by_model"].items():
            table.add_row(
                mid,
                f"${info['cost']:.6f}",
                str(info["input_tokens"]),
                str(info["output_tokens"]),
            )
        console.print(table)

    def _show_models(self) -> None:
        """List all registered models."""
        ids = self.engine.registry.get_model_ids()
        table = Table(title="Registered Models")
        table.add_column("ID", style="cyan")
        table.add_column("Provider", style="magenta")
        table.add_column("Specialization")
        table.add_column("Priority", justify="right")

        for mid in ids:
            cfg = self.engine.registry.get_config(mid) or {}
            specs = ", ".join(cfg.get("specialization", []))
            table.add_row(
                mid,
                cfg.get("provider", "?"),
                specs,
                str(cfg.get("priority", "")),
            )
        console.print(table)

    def _show_agents(self) -> None:
        """List all agents and their roles."""
        table = Table(title="Agents")
        table.add_column("ID", style="cyan")
        table.add_column("Role")
        table.add_column("State", style="green")
        table.add_column("Neighbors")

        for aid, agent in self.engine.agents.items():
            table.add_row(
                aid,
                agent.role[:60],
                agent.state.value,
                ", ".join(agent.neighbors[:3]),
            )
        console.print(table)

    def _show_history(self) -> None:
        """Show task history."""
        session = self.engine.session
        if not session or not session.tasks:
            console.print("[dim]No tasks in history.[/dim]")
            return

        table = Table(title=f"Task History (session: {session.session_id})")
        table.add_column("ID", style="cyan")
        table.add_column("Input")
        table.add_column("Status", style="green")
        table.add_column("Completed")

        for t in session.tasks:
            table.add_row(
                t.id,
                t.user_input[:60],
                t.status,
                t.completed_at or "—",
            )
        console.print(table)

    def _show_help(self) -> None:
        """Show available commands."""
        console.print(Panel(
            "[bold]Commands:[/bold]\n"
            "  /status    — show agents & system state\n"
            "  /cost      — show cost breakdown\n"
            "  /models    — list available models\n"
            "  /agents    — list agents & roles\n"
            "  /history   — show task history\n"
            "  /help      — this message\n"
            "  /quit      — exit Squix\n\n"
            "[dim]Or just type your task naturally — Squix will plan and execute it.[/dim]",
            title="Help",
            border_style="blue",
        ))

    async def _quit(self) -> None:
        """Exit the CLI."""
        console.print("\n[yellow]Shutting down Squix…[/yellow]")
        self.running = False
        await self.engine.shutdown()
        console.print("[green]Goodbye![/green]")
        sys.exit(0)
