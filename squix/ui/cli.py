"""CLI layer — Claude Code-style interface with agent visualization."""

from __future__ import annotations

import asyncio
import contextlib
import os
import shutil
import sys
from typing import Any

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completion, WordCompleter
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

console = Console()

# --------------------------------------------------------------------------- #
#  Theme definitions
# --------------------------------------------------------------------------- #

THEMES: dict[str, dict[str, Any]] = {
    "dark": {
        "label": "Dark (default)",
        "description": "Classic dark terminal — clean, readable, familiar",
        "user_color": "cyan",
        "user_border": "cyan",
        "plan_color": "white",
        "plan_border": "blue",
        "agent_color": "green",
        "agent_border": "green",
        "error_color": "red",
        "error_border": "red",
        "banner_style": "cyan",
        "prompt_style": "green",
        "table_title": "bold cyan",
        "dim": "dim",
    },
    "monokai": {
        "label": "Monokai",
        "description": "Warm orange/pink on dark bg — like the classic code theme",
        "user_color": "yellow",
        "user_border": "yellow",
        "plan_color": "white",
        "plan_border": "magenta",
        "agent_color": "green",
        "agent_border": "yellow",
        "error_color": "red",
        "error_border": "red",
        "banner_style": "magenta",
        "prompt_style": "yellow",
        "table_title": "bold magenta",
        "dim": "dim",
    },
    "dracula": {
        "label": "Dracula",
        "description": "Purple/cyan on dark — moody and vibrant",
        "user_color": "magenta",
        "user_border": "magenta",
        "plan_color": "white",
        "plan_border": "cyan",
        "agent_color": "green",
        "agent_border": "cyan",
        "error_color": "red",
        "error_border": "red",
        "banner_style": "magenta",
        "prompt_style": "cyan",
        "table_title": "bold cyan",
        "dim": "dim",
    },
    "nord": {
        "label": "Nord",
        "description": "Cool blue-gray arctic palette — calm and minimal",
        "user_color": "blue",
        "user_border": "blue",
        "plan_color": "white",
        "plan_border": "blue",
        "agent_color": "green",
        "agent_border": "blue",
        "error_color": "red",
        "error_border": "red",
        "banner_style": "blue",
        "prompt_style": "blue",
        "table_title": "bold blue",
        "dim": "dim",
    },
}

# --------------------------------------------------------------------------- #
#  Mode definitions
# --------------------------------------------------------------------------- #

MODE_ORDER = ["auto", "plan", "interactive", "talk"]

MODES: dict[str, dict[str, str]] = {
    "auto": {
        "label": "Auto",
        "icon": ">",
        "color": "green",
        "description": "Full AI control, auto-execute all steps",
    },
    "plan": {
        "label": "Plan",
        "icon": "#",
        "color": "yellow",
        "description": "Only plan, no execution",
    },
    "interactive": {
        "label": "Interactive",
        "icon": "?",
        "color": "cyan",
        "description": "Ask before executing each step",
    },
    "talk": {
        "label": "Talk",
        "icon": "~",
        "color": "magenta",
        "description": "Chat only, no task execution",
    },
}

# --------------------------------------------------------------------------- #
#  Command completer
# --------------------------------------------------------------------------- #

COMMANDS = {
    "/status": "agents & system state (live table)",
    "/cost": "cost breakdown",
    "/models": "available models",
    "/agents": "agents & their roles",
    "/graph": "ASCII graph of agent connections",
    "/history": "task history",
    "/init": "scan project & create profile",
    "/session": "current session info",
    "/theme": "change theme",
    "/mode": "cycle interaction mode",
    "/clear": "clear screen",
    "/help": "show all commands",
    "/quit": "exit Squix",
}

COMMAND_NAMES = list(COMMANDS.keys())


class SquixCompleter(WordCompleter):
    """Custom completer that shows command descriptions as meta."""

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        if not text.startswith("/"):
            return
        prefix = text[1:]
        for name, desc in COMMANDS.items():
            if name[1:].startswith(prefix.lower()):
                yield Completion(
                    name[1:],
                    start_position=-len(prefix),
                    display=name,
                    display_meta=desc,
                )


# --------------------------------------------------------------------------- #
#  Theme picker UI
# --------------------------------------------------------------------------- #

class ThemePicker:
    """Arrow-key theme picker."""

    def __init__(self, current_theme: str) -> None:
        self._keys = list(THEMES.keys())
        self._selected = self._keys.index(current_theme) if current_theme in self._keys else 0
        self._chosen: str | None = None

    def run(self) -> str | None:
        from prompt_toolkit.application import Application
        from prompt_toolkit.formatted_text import HTML
        from prompt_toolkit.layout import FormattedTextControl, Layout, Window
        from prompt_toolkit.layout.containers import HSplit
        from prompt_toolkit.layout.dimension import Dimension

        kb = KeyBindings()

        @kb.add("up")
        @kb.add("k")
        def _up(_event: Any) -> None:
            self._selected = (self._selected - 1) % len(self._keys)
            app.invalidate()

        @kb.add("down")
        @kb.add("j")
        def _down(_event: Any) -> None:
            self._selected = (self._selected + 1) % len(self._keys)
            app.invalidate()

        @kb.add("enter")
        def _apply(_event: Any) -> None:
            self._chosen = self._keys[self._selected]
            app.exit()

        @kb.add("escape")
        @kb.add("c-c")
        def _cancel(_event: Any) -> None:
            self._chosen = None
            app.exit()

        def _build_text():
            lines = ["\n  Select a theme:\n\n"]
            for i, tid in enumerate(self._keys):
                t = THEMES[tid]
                if i == self._selected:
                    lines.append(f"  > {t['label']}\n    {t['description']}\n\n")
                else:
                    lines.append(f"    {t['label']}\n    {t['description']}\n\n")
            lines.append("  up/down select  |  Enter apply  |  Esc cancel\n")
            return "".join(lines)

        control = FormattedTextControl(lambda: _build_text())
        app: Application[Any] = Application(
            layout=Layout(HSplit([Window(content=control, height=Dimension())])),
            key_bindings=kb,
            full_screen=True,
            mouse_support=False,
        )

        try:
            app.run()
        except KeyboardInterrupt:
            self._chosen = None
        return self._chosen


# --------------------------------------------------------------------------- #
#  Agent Graph Renderer
# --------------------------------------------------------------------------- #

def render_agent_graph(agents: dict[str, Any], links: dict[str, list[str]]) -> str:
    """Render an ASCII graph of agent connections with state indicators."""
    # State indicators
    state_icons = {
        "idle": ".",
        "working": "*",
        "waiting": "~",
        "error": "!",
        "done": "+",
    }

    lines = []
    lines.append("  Agent Connections Graph")
    lines.append("  " + "=" * 50)
    lines.append("")

    # Build adjacency display
    for aid, agent in sorted(agents.items()):
        icon = state_icons.get(agent.state.value, "?")
        model = agent.model_prefers[0].split("/")[-1][:12] if agent.model_prefers else "---"
        neighbors = links.get(aid, agent.neighbors)

        # Agent box
        state_str = agent.state.value.upper()
        lines.append(f"  [{icon}] {aid:<10} ({state_str:<7}) model: {model}")

        if neighbors:
            arrow_parts = []
            for n in neighbors[:6]:
                n_agent = agents.get(n)
                n_icon = state_icons.get(n_agent.state.value, "?") if n_agent else "?"
                arrow_parts.append(f"[{n_icon}]{n}")
            arrows = "  ".join(arrow_parts)
            lines.append(f"      --> {arrows}")
        lines.append("")

    # Legend
    lines.append("  Legend: [.] idle  [*] working  [~] waiting  [!] error  [+] done")

    return "\n".join(lines)


def render_agent_table(agents: dict[str, Any], cost_tracker: Any = None) -> Table:
    """Render a Rich table of agent states."""
    table = Table(title="Agent Status", title_style="bold cyan", border_style="dim")
    table.add_column("Agent", style="bold cyan", width=10)
    table.add_column("State", width=9)
    table.add_column("Model", style="dim", width=16)
    table.add_column("Progress", width=40)
    table.add_column("Cost", justify="right", width=10)

    state_colors = {
        "idle": "dim",
        "working": "bold green",
        "waiting": "yellow",
        "error": "bold red",
        "done": "green",
    }

    for aid, agent in sorted(agents.items()):
        state = agent.state.value
        color = state_colors.get(state, "white")
        model = agent.model_prefers[0].split("/")[-1][:16] if agent.model_prefers else "---"
        progress = (agent.progress[:40] if agent.progress else "---")
        cost = ""
        if cost_tracker:
            agent_cost = cost_tracker._agent_costs.get(aid, 0.0)
            if agent_cost > 0:
                cost = f"${agent_cost:.4f}"

        table.add_row(
            aid,
            Text(state, style=color),
            model,
            progress,
            cost,
        )

    return table


# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #

class SquixCLI:
    """Chat-style CLI with agent visualization and Claude Code-like modes."""

    def __init__(self, engine: Any) -> None:
        self.engine = engine
        self.running = False
        self.theme = "dark"
        self.current_mode: str = "auto"
        self._session: PromptSession | None = None

    # ------------------------------------------------------------------ #
    #  Helpers
    # ------------------------------------------------------------------ #

    def _th(self) -> dict[str, Any]:
        return THEMES.get(self.theme, THEMES["dark"])

    def _mode_info(self) -> dict[str, str]:
        return MODES.get(self.current_mode, MODES["auto"])

    def clear(self) -> None:
        os.system("cls" if os.name == "nt" else "clear")

    def print_msg(self, label: str, text: str, *,
                  color: str | None = None, border: str | None = None) -> None:
        th = self._th()
        c = color or th.get("agent_color", "green")
        b = border or th.get("agent_border", "green")
        cols = shutil.get_terminal_size().columns - 6
        lines = []
        for line in text.splitlines():
            while line:
                chunk = line[:cols]
                lines.append(chunk)
                line = line[len(chunk):]
        body = "\n".join(lines) if lines else text
        console.print(Panel(
            body,
            title=f"[bold {c}]{label}[/bold {c}]",
            border_style=b,
        ))
        console.print()

    # ------------------------------------------------------------------ #
    #  Banner + hints
    # ------------------------------------------------------------------ #

    def print_banner(self) -> None:
        self.clear()
        th = self._th()
        bs = th["banner_style"]
        mi = self._mode_info()
        mode_tag = f'[{mi["color"]}]{mi["icon"]} {mi["label"]}[/{mi["color"]}]'

        session_id = ""
        if self.engine.session:
            session_id = f"  session: {self.engine.session.session_id[:8]}"

        project_name = self.engine.project_dir.name
        agent_count = len(self.engine.agents)

        console.print(Panel(
            f"[bold {bs}]Squix[/bold {bs}]  [dim]AI Agent OS v0.1.0[/dim]  {mode_tag}\n"
            f"[dim]{project_name}  |  {agent_count} agents  |  {self.theme}{session_id}[/dim]\n"
            f"[dim]Shift-Tab: mode  |  Tab: complete  |  /help: commands[/dim]",
            border_style=bs,
        ))
        console.print()

    def _show_hints(self) -> None:
        th = self._th()
        ps = th["prompt_style"]
        hints = (
            f"[dim]/[/dim][bold {ps}]status[/bold {ps}]    agents state      "
            f"[dim]/[/dim][bold {ps}]graph[/bold {ps}]   agent connections\n"
            f"[dim]/[/dim][bold {ps}]cost[/bold {ps}]      costs              "
            f"[dim]/[/dim][bold {ps}]init[/bold {ps}]    scan project\n"
            f"[dim]/[/dim][bold {ps}]models[/bold {ps}]    models             "
            f"[dim]/[/dim][bold {ps}]theme[/bold {ps}]   change theme\n"
            f"[dim]/[/dim][bold {ps}]agents[/bold {ps}]    agents & roles     "
            f"[dim]/[/dim][bold {ps}]help[/bold {ps}]    all commands\n"
            f"[dim]/[/dim][bold {ps}]history[/bold {ps}]   task history       "
            f"[dim]/[/dim][bold {ps}]quit[/bold {ps}]    exit"
        )
        console.print(Panel(hints, border_style="dim", title="Commands"))
        console.print()

    # ------------------------------------------------------------------ #
    #  REPL
    # ------------------------------------------------------------------ #

    async def repl(self) -> None:
        self.running = True
        self.print_banner()
        self._show_hints()

        kb = KeyBindings()

        @kb.add(Keys.BackTab)
        def _cycle_mode(_event):
            self._cycle_mode()
            # Redraw prompt area
            if self._session and self._session.app:
                self._session.app.invalidate()

        self._session = PromptSession(key_bindings=kb)

        while self.running:
            try:
                mi = self._mode_info()
                prompt_text = f"squix [{mi['label']}] > "

                completer = SquixCompleter(
                    COMMAND_NAMES,
                    ignore_case=True,
                )
                loop = asyncio.get_event_loop()
                user_input = await loop.run_in_executor(
                    None,
                    lambda c=completer, p=prompt_text: self._session.prompt(
                        p,
                        completer=c,
                    ),
                )
            except (EOFError, KeyboardInterrupt):
                break

            user_input = user_input.strip()
            if not user_input:
                continue

            if user_input.startswith("/"):
                await self._handle_command(user_input)
            else:
                await self._handle_input(user_input)

    def _cycle_mode(self) -> None:
        idx = MODE_ORDER.index(self.current_mode)
        self.current_mode = MODE_ORDER[(idx + 1) % len(MODE_ORDER)]
        mi = self._mode_info()
        console.print(f"[{mi['color']}]{mi['icon']} Mode: {mi['label']} — {mi['description']}[/{mi['color']}]")
        console.print()

    # ------------------------------------------------------------------ #
    #  Input handling
    # ------------------------------------------------------------------ #

    async def _handle_input(self, text: str) -> None:
        """Process user input through the agent system."""
        th = self._th()

        # Show what user typed
        console.print(f"[dim]You:[/dim] {text}")
        console.print()

        # Show working indicator
        active_chain = self.engine.get_active_chain()

        try:
            # Process through Talk agent → full pipeline
            with console.status("[bold green]Thinking...", spinner="dots"):
                results = await self.engine.process_input(text)

            if not results:
                console.print("[dim]No response.[/dim]")
                console.print()
                return

            # Display results
            for msg in results:
                msg_type = msg.metadata.get("type", "")
                sender = msg.sender

                if msg_type == "error":
                    self.print_msg(
                        f"Error ({sender})",
                        msg.content,
                        color=th["error_color"],
                        border=th["error_border"],
                    )
                elif msg_type == "chat":
                    # Direct chat response from talk
                    self.print_msg("Squix", msg.content,
                                   color=th["agent_color"], border=th["agent_border"])
                elif msg_type == "final_result":
                    # Task pipeline result
                    self.print_msg(f"Result ({sender})", msg.content,
                                   color=th["agent_color"], border=th["agent_border"])
                elif msg_type == "result":
                    self.print_msg(f"{sender}", msg.content,
                                   color=th["agent_color"], border=th["agent_border"])
                else:
                    self.print_msg(sender, msg.content,
                                   color=th["agent_color"], border=th["agent_border"])

            # Show cost if any
            if self.engine.cost_tracker.total_cost > 0:
                total = self.engine.cost_tracker.total_cost
                calls = self.engine.cost_tracker.total_calls
                console.print(f"  [dim]$ {total:.4f}  |  {calls} calls[/dim]")
                console.print()

        except Exception as e:
            self.print_msg("Error", str(e),
                           color=th["error_color"], border=th["error_border"])

    # ------------------------------------------------------------------ #
    #  Commands
    # ------------------------------------------------------------------ #

    async def _handle_command(self, cmd: str) -> None:
        parts = cmd.split(maxsplit=1)
        command = parts[0].lower()

        commands = {
            "/status": self._show_status,
            "/cost": self._show_cost,
            "/models": self._show_models,
            "/agents": self._show_agents,
            "/graph": self._show_graph,
            "/history": self._show_history,
            "/init": self._run_init,
            "/session": self._show_session,
            "/help": self._show_help,
            "/theme": self._show_theme_menu,
            "/mode": self._cycle_mode,
            "/quit": self._quit,
            "/exit": self._quit,
            "/q": self._quit,
            "/clear": self._clear_screen,
        }

        handler = commands.get(command)
        if handler:
            # Handle /theme <name>
            if command == "/theme" and len(parts) > 1:
                name = parts[1].strip().lower()
                if name in THEMES:
                    self.theme = name
                    self.print_banner()
                    self._show_hints()
                    console.print(f"[bold green]>[/bold green] Theme: {THEMES[name]['label']}")
                else:
                    self.print_msg("Error", f"Unknown theme: {name}",
                                   color="red", border="red")
                return
            result = handler()
            if asyncio.iscoroutine(result):
                await result
            return

        th = self._th()
        self.print_msg("Error", f"Unknown command: {command}",
                       color=th["error_color"], border=th["error_border"])

    # ------------------------------------------------------------------ #
    #  Info screens
    # ------------------------------------------------------------------ #

    def _clear_screen(self) -> None:
        self.print_banner()
        self._show_hints()

    def _show_status(self) -> None:
        """Live table of agent states."""
        table = render_agent_table(self.engine.agents, self.engine.cost_tracker)
        console.print(table)

        # Summary line
        status = self.engine.get_status()
        total = status.get("total_cost", 0)
        tasks = status.get("tasks_completed", 0)
        active = self.engine.get_active_chain()
        active_str = " -> ".join(active) if active else "none"
        console.print(f"\n  [dim]Tasks: {tasks}  |  Cost: ${total:.4f}  |  Active: {active_str}[/dim]")
        console.print()

    def _show_cost(self) -> None:
        th = self._th()
        summary = self.engine.cost_tracker.summary()

        if not summary["by_model"]:
            console.print("[dim]No costs recorded yet.[/dim]")
            console.print()
            return

        table = Table(
            title=f"Cost Breakdown — Total: ${summary['total_cost']:.4f}  ({summary['total_calls']} calls)",
            title_style=th["table_title"],
        )
        table.add_column("Model", style=th["banner_style"])
        table.add_column("Cost ($)", justify="right")
        table.add_column("In Tokens", justify="right")
        table.add_column("Out Tokens", justify="right")
        table.add_column("Calls", justify="right")
        for mid, info in summary["by_model"].items():
            table.add_row(
                mid.split("/")[-1],
                f"${info['cost']:.6f}",
                str(info["input_tokens"]),
                str(info["output_tokens"]),
                str(info["calls"]),
            )
        console.print(table)

        # Agent costs
        if summary["by_agent"]:
            agent_table = Table(title="Cost by Agent", title_style=th["table_title"])
            agent_table.add_column("Agent", style=th["banner_style"])
            agent_table.add_column("Cost ($)", justify="right")
            for aid, cost in sorted(summary["by_agent"].items(), key=lambda x: -x[1]):
                agent_table.add_row(aid, f"${cost:.6f}")
            console.print(agent_table)

        console.print()

    def _show_models(self) -> None:
        th = self._th()
        ids = self.engine.registry.get_model_ids()
        table = Table(title="Registered Models", title_style=th["table_title"])
        table.add_column("ID", style=th["banner_style"])
        table.add_column("Provider", style="magenta")
        table.add_column("Specialization")
        table.add_column("Priority", justify="right")
        for mid in ids:
            cfg = self.engine.registry.get_config(mid) or {}
            table.add_row(
                mid,
                cfg.get("provider", "?"),
                ", ".join(cfg.get("specialization", [])),
                str(cfg.get("priority", "")),
            )
        console.print(table)
        console.print()

    def _show_agents(self) -> None:
        th = self._th()
        table = Table(title="Agents", title_style=th["table_title"])
        table.add_column("ID", style=th["banner_style"], width=10)
        table.add_column("Role", width=50)
        table.add_column("State", style="green", width=9)
        table.add_column("Neighbors", width=30)
        for aid, agent in sorted(self.engine.agents.items()):
            neighbors = ", ".join(agent.neighbors[:5])
            if len(agent.neighbors) > 5:
                neighbors += "..."
            table.add_row(
                aid,
                agent.role[:50],
                agent.state.value,
                neighbors,
            )
        console.print(table)
        console.print()

    def _show_graph(self) -> None:
        """Show ASCII graph of agent connections."""
        links = self.engine.config.get("agent_links", {})
        graph_text = render_agent_graph(self.engine.agents, links)
        console.print(Panel(graph_text, title="[bold cyan]Agent Graph[/bold cyan]",
                           border_style="cyan"))
        console.print()

    def _show_history(self) -> None:
        th = self._th()
        session = self.engine.session
        if not session or not session.tasks:
            console.print("[dim]No tasks in history.[/dim]")
            console.print()
            return

        table = Table(
            title=f"Task History (session: {session.session_id[:8]})",
            title_style=th["table_title"],
        )
        table.add_column("ID", style=th["banner_style"])
        table.add_column("Input", width=50)
        table.add_column("Status", style="green")
        table.add_column("Completed")
        for t in session.tasks:
            table.add_row(
                t.id,
                t.user_input[:50],
                t.status,
                t.completed_at or "---",
            )
        console.print(table)
        console.print()

    async def _run_init(self) -> None:
        """Scan the project and create a profile."""
        from squix.core.init_scanner import ProjectScanner

        console.print("[dim]Scanning project...[/dim]")
        scanner = ProjectScanner(self.engine.project_dir)

        with console.status("[bold green]Scanning files...", spinner="dots"):
            profile = scanner.scan_and_save()

        formatted = ProjectScanner.format_profile(profile)
        self.print_msg("Project Profile", formatted,
                       color="cyan", border="cyan")

    def _show_session(self) -> None:
        session = self.engine.session
        if not session:
            console.print("[dim]No active session.[/dim]")
            console.print()
            return

        info = (
            f"Session ID: {session.session_id}\n"
            f"Created: {session.created_at}\n"
            f"Tasks: {len(session.tasks)}\n"
            f"Completed: {session.tasks_completed}\n"
            f"Total Cost: ${self.engine.cost_tracker.total_cost:.4f}"
        )
        self.print_msg("Session", info, color="cyan", border="cyan")

    def _show_help(self) -> None:
        help_text = (
            "[bold]Commands:[/bold]\n"
            "  /status    — agents state (live table)\n"
            "  /graph     — ASCII graph of agent connections\n"
            "  /cost      — cost breakdown by model & agent\n"
            "  /models    — list available models\n"
            "  /agents    — list agents & roles\n"
            "  /history   — task history\n"
            "  /init      — scan project, create profile\n"
            "  /session   — current session info\n"
            "  /theme     — change theme (arrow-key selector)\n"
            "  /mode      — cycle interaction mode\n"
            "  /clear     — clear screen\n"
            "  /help      — this message\n"
            "  /quit      — exit Squix\n\n"
            "[bold]Modes (Shift-Tab):[/bold]\n"
            "  Auto        — full AI control, auto-execute\n"
            "  Plan        — only plan, no execution\n"
            "  Interactive — approve each step\n"
            "  Talk        — chat only, no tasks\n\n"
            "[dim]Type / and press Tab to autocomplete commands.[/dim]"
        )
        console.print(Panel(help_text, title="Help", border_style="blue"))
        console.print()

    def _show_theme_menu(self) -> None:
        if not sys.stdin.isatty():
            console.print("[dim]Type /theme <name> (dark/monokai/dracula/nord).[/dim]")
            return

        picker = ThemePicker(self.theme)
        chosen = picker.run()
        self.print_banner()
        if chosen:
            self.theme = chosen
            self._show_hints()
            console.print(f"[bold green]>[/bold green] Theme: {THEMES[chosen]['label']}")
        else:
            console.print("[dim]Theme unchanged.[/dim]")
            self._show_hints()
        console.print()

    async def _quit(self) -> None:
        self.running = False
        await self.engine.shutdown()
        console.print("\n[yellow]Goodbye![/yellow]")
        sys.exit(0)
