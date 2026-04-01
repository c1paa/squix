"""Squix Skills — tool layer that agents call to perform real actions.

Skills are the bridge between LLM reasoning and actual world mutations.
Each skill is a callable function with:
  - name, description, parameters
  - agent permissions (which agents may invoke it)
  - execution logic

Agents never "talk about" actions — they call the skill directly.
"""

from squix.skills.definitions import SKILLS, SkillDef
from squix.skills.registry import SkillRegistry

__all__ = ["SkillRegistry", "SKILLS", "SkillDef"]
