"""Talk agent — the ONLY external user interface.

Talk:
  1. Receives ALL user messages
  2. Uses keyword pre-check for fast routing
  3. Falls back to LLM classification
  4. ALWAYS delegates to Orch (never directly to workers)
  5. Orch handles dispatch, validation, and result formatting
"""

from __future__ import annotations

import json
import logging

from squix.agents.base import AgentMessage, BaseAgent

logger = logging.getLogger("squix.agent.talk")

# ── Keyword pre-check: catch obvious tasks regardless of conversational wrapper ──
_DEBUG_KW = (
    "not working", "не работает", "почини", "исправ",
    "ошибк", "баг", "глюк", "bug", "error",
    "debug", "broken", "краш", "проблем", "проблема",
    "проблемы", "падет", "падает", "fix", "починить",
    "исправить", "почему не", "what is wrong", "wrong with",
    "why does", "why doesn", "why doesn't", "what's wrong", "что не так",
    "почему не",
)
_BUILD_KW = (
    "write", "создай", "напиши", "сделай", "добавь",
    "удали", "измени", "build", "create", "implement",
    "refactor", "optimize", "добавить", "удалить",
    "изменить", "рефактор", "игра", "game", "игру",
    "приложение", "app", "сайт", "site", "скрипт", "script",
    "функцию", "function", "класс", "class", "модуль", "module",
)
_GREETING_KW = (
    "привет", "hello", "hi ", "hey", "добрый", "доброе",
    "здравствуйте", "как дела", "how are you",
)
_PLAN_KW = (
    "спроектируй", "спланируй", "архитектур", "design",
    "планирую", "запланируй", "спроектируй", "план проекта",
    "как сделать", "how to make", "how would you",
    "как лучше", "что лучше", "как бы ты",
)
_RESEARCH_KW = (
    "найди", "найди в интернете", "погугли", "search for",
    "look up", "документацию", "documentation",
    "разработк", "research", "исследуй",
)
_DOCS_KW = (
    "readme", "документацию", "documentation", "напиши readme",
    "create readme", "обнови документацию", "update documentation",
    "инструкцию", "manual", "manual for", "инструкция для",
)


class TalkAgent(BaseAgent):
    """Sole user-facing interface. Classify → delegate to Orch."""

    agent_id = "talk"
    role = (
        "TALK — you are the ONLY interface between the user and the Squix AI "
        "company. Your job: receive messages from the user, classify them, "
        "and either answer directly (greetings, simple questions) or delegate "
        "to ORCH (the operations manager). "
        "You NEVER delegate directly to build/debug/web/DB — always through ORCH. "
        "You NEVER produce code or make file changes. "
        "When you receive a result from Orch, return a concise "
        "summary to the user."
    )

    def _default_system_prompt(self) -> str:
        return (
            "You are Talk, the router agent in the Squix AI system.\n"
            "Classify messages and either respond directly or delegate to ORCH.\n\n"
            "Agents (for reference only — always delegate to orch, never to them):\n"
            "  plan (big tasks), debug (bugs/errors), build (code creation),\n"
            "  web (external info), DB (project knowledge), AI (ML tasks),\n"
            "  idea (brainstorming), product (features/UX),\n"
            "  orch (task coordination), README (docs).\n\n"
            "When delegating, send to ORCH with:\n"
            '  {"action": "orch", "task_type": "...", "reason": "...", '
            '"output_mode": "..." }\n\n'
            "task_type options: simple_chat, code_edit, code_generate, "
            "debugging, research, docs_write, product_discussion\n"
            "output_mode options: text_response, file_patch, file_create, "
            "multi_file_project_output, summary_with_artifacts"
        )

    async def handle(self, msg: AgentMessage) -> AgentMessage | None:
        self.progress = "Classifying user input..."

        # ─── 1. Keyword pre-check ───
        kw_result = self._classify_by_keywords(msg.content)
        if kw_result:
            task_type, output_mode, reason = kw_result
            if task_type == "simple_chat":
                direct = await self._generate_chat_response(msg.content)
                self.progress = "Responded directly (keyword match)"
                return AgentMessage(
                    sender=self.agent_id, recipient="user",
                    content=direct, task_id=msg.task_id,
                    metadata={"type": "chat", "reason": reason},
                )
            # Delegate to ORCH (never directly to workers)
            self.progress = f"Delegating to orch (keyword match: {task_type})"
            return AgentMessage(
                sender=self.agent_id,
                recipient="orch",
                content=msg.content,
                task_id=msg.task_id,
                metadata={
                    "type": "delegate",
                    "reason": reason,
                    "task_type": task_type,
                    "output_mode": output_mode,
                },
            )

        # ─── 2. LLM classification ───
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": msg.content},
        ]
        response = await self.invoke_llm(messages, temperature=0.3)
        action, task_type, output_mode, reason, direct_response = self._parse_classification(response.text)

        if action == "talk":
            if not direct_response:
                direct_response = await self._generate_chat_response(msg.content)
            self.progress = "Responded directly"
            return AgentMessage(
                sender=self.agent_id,
                recipient="user",
                content=direct_response,
                task_id=msg.task_id,
                metadata={"type": "chat", "task_type": task_type, "reason": reason},
            )
        else:
            # ALWAYS delegate to orch (never to workers directly)
            self.progress = f"Delegating to orch (LLM: {task_type})"
            return AgentMessage(
                sender=self.agent_id,
                recipient="orch",
                content=msg.content,
                task_id=msg.task_id,
                metadata={
                    "type": "delegate",
                    "task_type": task_type,
                    "output_mode": output_mode,
                    "reason": reason,
                },
            )

    # ── Keyword pre-check ──────────────────────────────────────────────

    def _classify_by_keywords(self, text: str) -> tuple[str, str, str] | None:
        """Fast pre-check.

        Returns (task_type, output_mode, reason) or None.
        """
        lower = text.lower()

        # Debug takes priority
        if any(kw in lower for kw in _DEBUG_KW):
            return "debugging", "file_patch", "debug_keyword"
        # Docs before build (README.md requests)
        if any(kw in lower for kw in _DOCS_KW):
            return "docs_write", "file_create", "docs_keyword"
        # Build / generation
        if any(kw in lower for kw in _BUILD_KW):
            return "code_generate", "file_create", "build_keyword"
        # Research
        if any(kw in lower for kw in _RESEARCH_KW):
            return "research", "summary_with_artifacts", "research_keyword"
        # Plan / design
        if any(kw in lower for kw in _PLAN_KW):
            return "product_discussion", "text_response", "plan_keyword"
        # Greeting
        if any(kw in lower for kw in _GREETING_KW):
            return "simple_chat", "text_response", "greeting"
        return None

    # ── LLM classification parser ──────────────────────────────────────

    def _parse_classification(
        self, text: str,
    ) -> tuple[str, str, str, str, str]:
        """Parse LLM JSON → (action, task_type, output_mode, reason, response)."""
        text = text.strip()
        if "```" in text:
            parts = text.split("```")
            for part in parts:
                part = part.strip()
                if part.startswith("json"):
                    part = part[4:].strip()
                if part.startswith("{"):
                    text = part
                    break

        defaults = ("talk", "simple_chat", "text_response", "parse_fallback", "")

        try:
            data = json.loads(text)
            action = data.get("action", "talk")
            task_type = data.get("task_type", "simple_chat")
            output_mode = data.get("output_mode", "text_response")
            reason = data.get("reason", "llm_default")
            response = data.get("response", "")
        except (json.JSONDecodeError, TypeError, AttributeError):
            return defaults

        # Sanitize
        valid_actions = {
            "talk", "plan", "idea", "product", "web", "DB",
            "build", "debug", "orch", "AI", "README",
        }
        if action not in valid_actions:
            action = "talk"
        # Force everything to go through orch
        if action != "talk":
            action = "orch"

        return action, task_type, output_mode, reason, response

    # ── Direct chat generator ──────────────────────────────────────────

    async def _generate_chat_response(self, user_input: str) -> str:
        messages = [
            {"role": "system", "content": (
                "You are Squix — a helpful AI assistant. "
                "Answer concisely and naturally. "
                "If you don't know something, say so briefly."
            )},
            {"role": "user", "content": user_input},
        ]
        response = await self.invoke_llm(messages, temperature=0.7)
        return response.text.strip()
