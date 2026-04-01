# Squix — AI Agent OS

> Orchestrate multiple AI models and agents as a managed team.

Squix is an **operating system for AI** — it lets you build an "AI company" out of different models (local and cloud), distribute tasks across them, manage limits and costs, and see the entire execution flow.

## Core Concept

Instead of using one LLM for everything, Squix coordinates a **team of specialized agents** — planner, orchestrator, builder, debugger, researcher, product manager, documentation writer, and more — each using the right model for their task.

```
User → Planner → Orchestrator → [Builder | Debugger | Web | DB | AI | Product | README]
```

The **Planner** breaks big tasks into steps, the **Orchestrator** dispatches them to worker agents, and the **Policy Engine** routes each request to the best model based on cost, capability, and availability.

## Features

- 🔌 **Multi-provider**: OpenRouter (cloud) + Ollama (local) via unified interface
- 👥 **10 built-in agents**: Orchestrator, Planner, Builder, Debugger, Web Researcher, Knowledge Base, AI Specialist, Idea Explorer, Product Manager, Documentation Writer
- 🧠 **Agent communication graph**: Predefined topology — agents know who to talk to
- ⚖️ **Policy Engine**: Smart routing (cheap vs expensive models), fallback, escalation
- 💰 **Cost tracking**: Per-model and per-agent token cost tracking with limits
- 💾 **Session memory**: Save/restore state across restarts — crash recovery
- 🔍 **Observability**: Structured logs, real-time status, cost breakdown
- 📁 **Workspace**: File I/O, code execution, artifact storage in the project directory
- 🛠️ **Agent Generator**: Create custom agents, clone existing ones, enable/disable per project
- ⌨️ **Rich CLI**: Interactive REPL with beautiful terminal UI

## Quick Start

### Prerequisites

- Python 3.11+
- An OpenRouter API key (for cloud models) — [get one here](https://openrouter.ai/keys)
- [Ollama](https://ollama.ai/) installed and running (for local models, optional)

### Installation

```bash
# From the project directory
pip install -e ".[dev]"
```

### Running

```bash
# Set your API key
export OPENROUTER_API_KEY="sk-..."

# Run from any project directory
python -m squix.main
# or if installed:
squix
```

### CLI Commands

| Command | Description |
|---------|-------------|
| (type any text) | Submit a task |
| `/status` | Show agents & system state |
| `/cost` | Show cost breakdown |
| `/models` | List available models |
| `/agents` | List agents & roles |
| `/history` | Show task history |
| `/help` | Show available commands |
| `/quit` or `/exit` | Exit Squix |

## Configuration

### Basic config (squix.yml)

Place a `squix.yml` in your project root to override defaults:

```yaml
models:
  - id: openrouter/anthropic/claude-sonnet-4-20250514
    provider: openrouter
    cost_per_1k_input: 0.003
    cost_per_1k_output: 0.015
    specialization: ["code", "reasoning"]
    priority: 5

agents:
  - id: build
    count: 2  # use 2 builders in parallel

policy:
  fallback_enabled: true
  max_retries: 3
  escalation_enabled: true
```

### Disabling agents per project

```yaml
agents:
  - id: web
    enabled: false
  - id: DB
    enabled: false
```

See `squix/default_config.yml` for the full default configuration.

### Environment Variables

| Variable | Description |
|----------|-------------|
| `OPENROUTER_API_KEY` | OpenRouter API key |
| `SQUIX_CONFIG` | Path to custom config file |
| `SQUIX_DEBUG` | Set to `1` for debug logging |

## Architecture

```
squix/
 ├── core/           Engine + config loader + session management
 ├── agents/
 │    ├── built_in/  10 pre-built agents
 │    └── generator/ Create/clone/custom agents
 ├── models/         Base adapter + OpenRouter/Ollama + registry
 ├── policy/         Routing rules, fallback, escalation
 ├── memory/         State persistence (JSON), save/restore
 ├── observability/  Logging, cost tracking
 ├── workspace/      File I/O, code execution, artifacts
 ├── api/            Provider adapters (OpenRouter, Ollama)
 ├── ui/             Rich CLI interface
 └── main.py         Entry point
```

### Agent Communication Topology

```
idea → product, web
product → web, DB, plan, orch
web → product, DB, orch, idea
plan → product, orch
DB → web, product, orch
orch → AI, README, debug, build, plan, product, web, DB
build → debug, AI, orch
debug → orch, build, AI
AI → orch, build, debug
README → orch
```

**Flow**: User → Planner (splits into steps) → Orchestrator (dispatches) → Workers (execute) → Results collected

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run linter
ruff check .

# Run tests
pytest
```

## Roadmap

- [x] Core architecture and agent framework
- [x] OpenRouter & Ollama adapters
- [x] Policy engine with routing and fallback
- [x] Memory layer with JSON persistence
- [x] Cost tracking and observability
- [x] Rich interactive CLI
- [x] Workspace and code execution
- [ ] Streaming responses (real-time agent output)
- [ ] Visual agent communication graph in terminal
- [ ] Web UI dashboard
- [ ] More model providers (Direct OpenAI, Anthropic, Gemini)
- [ ] Vector DB for knowledge base
- [ ] Marketplace of pre-built agent teams
- [ ] Plugin system for custom tools

## License

MIT
