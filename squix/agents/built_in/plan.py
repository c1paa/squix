"""Planner agent — breaks large tasks into actionable steps."""

from __future__ import annotations

import json

from squix.agents.base import AgentMessage, BaseAgent


class PlannerAgent(BaseAgent):
    """Receives the user's high-level task and produces a structured plan.

    Flow:
    1. THINK — understand the task
    2. ACT (skills) — read project structure, find main file
    3. THINK — create step-by-step plan
    4. HANDOFF — send plan to Orch
    """

    agent_id = "plan"
    role = (
        "PLANNER — you are the strategist of the Squix AI company. "
        "You receive large, ambiguous tasks and break them into concrete, "
        "sequential steps. Each step specifies which agent handles it "
        "(build/debug/web/DB/AI/README) and exactly what they should do. "
        "You NEVER talk to the user. Your output goes ONLY to Orch. "
        "ALWAYS check the project structure before planning."
    )

    async def handle(self, msg: AgentMessage) -> AgentMessage | None:
        self.progress = "Analyzing task and project structure..."

        # ── Step 1: ACT (read context) ──
        struct_result = await self.invoke_skill("get_project_structure")
        if struct_result.get("status") == "success":
            context = f"Project structure:\n```\n{struct_result.get('structure', '')}\n```\n\n"
            if struct_result.get("readme"):
                context += f"README:\n```\n{struct_result.get('readme', '')[:1000]}\n```\n\n"
        else:
            context = ""

        # ── Step 2: THINK — create plan ──
        messages = [
            {"role": "system", "content": (
                "You are a senior technical planner. "
                "Given a task and project context, produce a STEP-BY-STEP plan.\n"
                "Each step must specify:\n"
                "  - agent: which internal agent handles it (build, debug, web, DB, AI, README)\n"
                "  - task: clear, specific instruction for that agent\n\n"
                "Return ONLY valid JSON. Format:\n"
                '{"steps": [{"agent": "build", "task": "..."}, ...]}'
            )},
            {"role": "user", "content": f"TASK: {msg.content}\n\n{context}"},
        ]
        response = await self.invoke_llm(messages, temperature=0.5)

        # Parse plan
        try:
            text = response.text.strip()
            if "```" in text:
                text = text.split("```")[1].strip()
                if text.startswith("json"):
                    text = text[4:].strip()
            plan = json.loads(text)
        except Exception:
            # Fallback: wrap raw text as single step
            plan = {"steps": [{"agent": "build", "task": msg.content}]}

        self.progress = "Plan created, sending to Orch"

        return AgentMessage(
            sender=self.agent_id,
            recipient="orch",
            content=json.dumps(plan, indent=2) if isinstance(plan, dict) else response.text,
            task_id=msg.task_id,
            metadata={
                "type": "plan",
                "original_input": msg.content,
                "num_steps": len(plan.get("steps", []) if isinstance(plan, dict) else []),
            },
        )
