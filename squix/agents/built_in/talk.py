"""Talk agent — the single entry point for all user messages.

Classifies user input and either responds directly or delegates to
the appropriate specialist agent (plan, idea, product, etc.).
"""

from __future__ import annotations

import json
import logging

from squix.agents.base import AgentMessage, BaseAgent

logger = logging.getLogger("squix.agent.talk")

# Classification prompt that explains all available agents
_CLASSIFY_PROMPT = """\
You are Talk — the router agent in the Squix AI system.
Your job is to classify the user's message and decide how to handle it.

Available agents and when to use them:
- "talk" — answer yourself for: greetings, simple questions, casual chat, general knowledge
- "plan" — delegate for: complex tasks that need multiple steps (coding, building, creating)
- "idea" — delegate for: brainstorming, discussing project ideas, exploring concepts
- "product" — delegate for: product planning, features, UX, roadmaps, turning ideas into products
- "web" — delegate for: when user needs web search, external information, documentation lookup
- "DB" — delegate for: project knowledge questions, retrieving stored information
- "build" — delegate for: simple single-step coding tasks (write a function, fix one thing)
- "debug" — delegate for: debugging, error analysis, finding bugs

Rules:
1. If the message is casual chat, greeting, or a simple question — handle it yourself ("talk")
2. If it's a complex task requiring planning — send to "plan"
3. If it's about ideas or brainstorming — send to "idea"
4. If it's about product/feature planning — send to "product"
5. If it's a simple coding task (one step) — send to "build"
6. If it's about debugging/errors — send to "debug"

Respond with ONLY a JSON object (no markdown, no explanation):
{"action": "talk" or agent_id, "reason": "brief reason"}

If action is "talk", also include a "response" field with your answer to the user.
{"action": "talk", "reason": "greeting", "response": "your answer here"}
"""


class TalkAgent(BaseAgent):
    """Single entry point for all user messages. Classifies and routes."""

    agent_id = "talk"
    role = (
        "Talk — the user-facing router. Classify messages and either "
        "answer directly or delegate to the right specialist agent."
    )

    def _default_system_prompt(self) -> str:
        return _CLASSIFY_PROMPT

    async def handle(self, msg: AgentMessage) -> AgentMessage | None:
        self.progress = "Classifying user input..."

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": msg.content},
        ]
        response = await self.invoke_llm(messages, temperature=0.3)

        # Parse the classification
        action, reason, direct_response = self._parse_classification(response.text)

        if action == "talk":
            # Respond directly to the user
            if not direct_response:
                # LLM didn't include a response, generate one
                direct_response = await self._generate_chat_response(msg.content)
            self.progress = "Responded directly"
            return AgentMessage(
                sender=self.agent_id,
                recipient="user",
                content=direct_response,
                task_id=msg.task_id,
                metadata={"type": "chat", "reason": reason},
            )
        else:
            # Delegate to another agent
            self.progress = f"Delegating to {action}"
            return AgentMessage(
                sender=self.agent_id,
                recipient=action,
                content=msg.content,
                task_id=msg.task_id,
                metadata={"type": "delegate", "reason": reason, "original_sender": "user"},
            )

    def _parse_classification(self, text: str) -> tuple[str, str, str]:
        """Parse the LLM classification response. Returns (action, reason, response)."""
        text = text.strip()

        # Try to extract JSON from the response
        # Sometimes LLM wraps in markdown code blocks
        if "```" in text:
            parts = text.split("```")
            for part in parts:
                part = part.strip()
                if part.startswith("json"):
                    part = part[4:].strip()
                if part.startswith("{"):
                    text = part
                    break

        try:
            data = json.loads(text)
            action = data.get("action", "talk")
            reason = data.get("reason", "")
            response = data.get("response", "")
            # Validate the action is a known agent
            valid_targets = {"talk", "plan", "idea", "product", "web", "DB",
                             "build", "debug", "orch", "AI", "README"}
            if action not in valid_targets:
                action = "talk"
            return action, reason, response
        except (json.JSONDecodeError, TypeError, AttributeError):
            # If parsing fails, treat the whole text as a direct response
            logger.warning("Talk: failed to parse classification, treating as direct response")
            return "talk", "parse_fallback", text

    async def _generate_chat_response(self, user_input: str) -> str:
        """Generate a direct chat response when talk handles it itself."""
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
