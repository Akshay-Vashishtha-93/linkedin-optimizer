"""Multi-stage Content Lab pipeline for LinkedIn post drafting.

Replaces the single-shot LLM call with a 4-stage pipeline:
  Stage 1 - Research: extract key insight, angle, metric placeholder
  Stage 2 - Hook Writing: generate 3 hook options (question, bold, story)
  Stage 3 - Body Draft: full post using best hook + research
  Stage 4 - Anti-slop Review: validate against rules, auto-fix once if needed
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from .common import cfg, compact_text, skill_spec
from .validators import banned_hits, has_placeholder, validate_draft


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CONTENT_SYSTEM_PROMPT = (
    "You help Akshay Vashishtha draft LinkedIn posts. "
    "Only use facts provided. Unknown facts become [INSERT ...] placeholders. "
    "Never invent metrics."
)

_PILLARS_CFG = None


def _pillars() -> Dict[str, Any]:
    global _PILLARS_CFG
    if _PILLARS_CFG is None:
        _PILLARS_CFG = cfg("content-pillars.json", {})
    return _PILLARS_CFG


def _anti_slop_rules() -> List[str]:
    return _pillars().get("anti_slop_rules", [])


def _pillar_by_id(pillar_id: str) -> Dict[str, Any]:
    for p in _pillars().get("pillars", []):
        if p.get("id") == pillar_id:
            return p
    return {}


# ---------------------------------------------------------------------------
# Deterministic fallbacks
# ---------------------------------------------------------------------------

def _fallback_research(topic: str, inspiration_posts: list) -> Dict[str, Any]:
    summary = ""
    if inspiration_posts:
        first = inspiration_posts[0]
        author = (first.get("author") or {}).get("name", "a peer PM")
        summary = compact_text(first.get("content", ""), 200)
    else:
        author = "industry peers"
        summary = f"Discussion around {topic}"
    return {
        "key_insight": f"{author} discussed {topic} — extract the core takeaway",
        "akshay_angle": f"Akshay can relate from his Mumzworld/Myntra/ABFRL experience with {topic}",
        "metric_placeholder": "[INSERT SPECIFIC METRIC from your experience]",
        "source_summary": summary,
    }


def _fallback_hooks(topic: str) -> List[Dict[str, str]]:
    return [
        {"type": "question", "text": f"Why do most PMs get {topic} wrong?"},
        {"type": "bold_statement", "text": f"The real problem with {topic} is not what you think."},
        {"type": "story", "text": f"Last quarter I learned something about {topic} the hard way."},
    ]


def _fallback_body(topic: str, hook: str) -> str:
    return (
        f"{hook}\n\n"
        "Here is what I saw firsthand: [INSERT YOUR REAL EXAMPLE HERE].\n\n"
        "The numbers told a clear story: [INSERT SPECIFIC METRIC].\n\n"
        "At Mumzworld/Myntra/ABFRL, this changed how I approached the problem.\n\n"
        "What has your experience been with this?"
    )


# ---------------------------------------------------------------------------
# ContentLab
# ---------------------------------------------------------------------------

class ContentLab:
    """Multi-stage post drafting pipeline."""

    def __init__(self, llm: Any):
        """Accept an LLMClient instance (from skills.py)."""
        self.llm = llm

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def draft_from_inspiration(
        self,
        topic: str,
        inspiration_posts: list,
        pillar: str = "",
    ) -> Dict[str, Any]:
        """Run all 4 stages: Research -> Hooks -> Body -> Anti-slop."""
        stages_completed: List[str] = []
        stages_fallback: List[str] = []

        # Stage 1 — Research
        research = self._stage_research(topic, inspiration_posts, pillar)
        (stages_completed if research["_from_llm"] else stages_fallback).append("research")

        # Stage 2 — Hook Writing
        hooks_result = self._stage_hooks(topic, research, pillar)
        (stages_completed if hooks_result["_from_llm"] else stages_fallback).append("hooks")

        # Stage 3 — Body Draft
        body_result = self._stage_body(topic, research, hooks_result, pillar)
        (stages_completed if body_result["_from_llm"] else stages_fallback).append("body")

        # Stage 4 — Anti-slop Review
        reviewed = self._stage_anti_slop(body_result["body"])
        (stages_completed if reviewed["_from_llm"] else stages_fallback).append("anti_slop")

        return {
            "hooks": hooks_result["hooks"],
            "selected_hook": hooks_result["selected_hook"],
            "body": reviewed["body"],
            "char_count": len(reviewed["body"]),
            "validation": reviewed["validation"],
            "research": {
                "key_insight": research["key_insight"],
                "akshay_angle": research["akshay_angle"],
                "metric_placeholder": research["metric_placeholder"],
            },
            "stages_completed": stages_completed,
            "stages_fallback": stages_fallback,
        }

    def draft_from_topic(
        self,
        topic: str,
        angle: str,
        pillar: str = "",
    ) -> Dict[str, Any]:
        """Run stages 2-4 (for Topic Radar integration — research already done)."""
        stages_completed: List[str] = []
        stages_fallback: List[str] = []

        # Synthetic research from provided angle
        research = {
            "key_insight": f"Topic: {topic}",
            "akshay_angle": angle,
            "metric_placeholder": "[INSERT SPECIFIC METRIC]",
            "source_summary": topic,
            "_from_llm": False,
        }

        # Stage 2 — Hook Writing
        hooks_result = self._stage_hooks(topic, research, pillar)
        (stages_completed if hooks_result["_from_llm"] else stages_fallback).append("hooks")

        # Stage 3 — Body Draft
        body_result = self._stage_body(topic, research, hooks_result, pillar)
        (stages_completed if body_result["_from_llm"] else stages_fallback).append("body")

        # Stage 4 — Anti-slop Review
        reviewed = self._stage_anti_slop(body_result["body"])
        (stages_completed if reviewed["_from_llm"] else stages_fallback).append("anti_slop")

        return {
            "hooks": hooks_result["hooks"],
            "selected_hook": hooks_result["selected_hook"],
            "body": reviewed["body"],
            "char_count": len(reviewed["body"]),
            "validation": reviewed["validation"],
            "research": {
                "key_insight": research["key_insight"],
                "akshay_angle": research["akshay_angle"],
                "metric_placeholder": research["metric_placeholder"],
            },
            "stages_completed": stages_completed,
            "stages_fallback": stages_fallback,
        }

    # ------------------------------------------------------------------
    # Stage 1 — Research
    # ------------------------------------------------------------------

    def _stage_research(
        self, topic: str, inspiration_posts: list, pillar: str
    ) -> Dict[str, Any]:
        fallback = _fallback_research(topic, inspiration_posts)
        fallback["_from_llm"] = False

        if not self.llm.available:
            return fallback

        post_summaries = []
        for p in inspiration_posts[:3]:
            author = (p.get("author") or {}).get("name", "Unknown")
            content = compact_text(p.get("content", ""), 400)
            engagement = p.get("engagement") or p.get("stats") or {}
            post_summaries.append(
                {"author": author, "content": content, "engagement": engagement}
            )

        pillar_info = _pillar_by_id(pillar)
        prompt = json.dumps(
            {
                "task": (
                    "Analyze these LinkedIn posts and extract research for Akshay to write his own post. "
                    "Return JSON with exactly these keys: key_insight, akshay_angle, metric_placeholder. "
                    "key_insight = the core takeaway from the inspiration posts. "
                    "akshay_angle = what angle Akshay could take based on his Mumzworld/Myntra/ABFRL experience. "
                    "metric_placeholder = a specific [INSERT ...] placeholder for a metric Akshay should add."
                ),
                "topic": topic,
                "pillar": pillar_info.get("name", pillar),
                "inspiration_posts": post_summaries,
                "akshay_background": "PM at Mumzworld (mother & baby e-commerce, Middle East). Previously Myntra and ABFRL (fashion e-commerce, India).",
                "rules": [
                    "Use only the facts provided here.",
                    "Unknown facts become [INSERT ...] placeholders.",
                    "Never invent metrics or personal experiences.",
                    "Return valid JSON only, no markdown fences.",
                ],
            },
            indent=2,
            ensure_ascii=False,
            default=str,
        )

        raw = self.llm.call(prompt, max_tokens=350)
        if not raw:
            return fallback

        try:
            parsed = json.loads(raw.strip().removeprefix("```json").removesuffix("```").strip())
            if not isinstance(parsed, dict):
                return fallback
            result = {
                "key_insight": str(parsed.get("key_insight", fallback["key_insight"])),
                "akshay_angle": str(parsed.get("akshay_angle", fallback["akshay_angle"])),
                "metric_placeholder": str(parsed.get("metric_placeholder", fallback["metric_placeholder"])),
                "source_summary": fallback.get("source_summary", ""),
                "_from_llm": True,
            }
            return result
        except (json.JSONDecodeError, ValueError):
            return fallback

    # ------------------------------------------------------------------
    # Stage 2 — Hook Writing
    # ------------------------------------------------------------------

    def _stage_hooks(
        self, topic: str, research: Dict[str, Any], pillar: str
    ) -> Dict[str, Any]:
        fallback_hooks = _fallback_hooks(topic)
        fallback = {
            "hooks": fallback_hooks,
            "selected_hook": fallback_hooks[0]["text"],
            "_from_llm": False,
        }

        if not self.llm.available:
            return fallback

        pillar_info = _pillar_by_id(pillar)
        prompt = json.dumps(
            {
                "task": (
                    "Write 3 LinkedIn post hook options (the first line of a post). "
                    "Each hook must be max 100 characters. Return JSON array with objects: "
                    '{type, text}. Types: "question", "bold_statement", "story". '
                    "Return valid JSON only, no markdown fences."
                ),
                "topic": topic,
                "key_insight": research.get("key_insight", ""),
                "akshay_angle": research.get("akshay_angle", ""),
                "pillar": pillar_info.get("name", pillar),
                "example_angles": pillar_info.get("example_angles", []),
                "rules": [
                    "First-person voice. Short, punchy.",
                    "No corporate speak. No banned phrases.",
                    "Each hook max 100 characters.",
                    "Never invent metrics or facts.",
                ],
                "banned_phrases": [
                    "In today's fast-paced world", "Game-changer", "Leverage",
                    "Synergy", "Delighted to share", "Thrilled to announce",
                ],
            },
            indent=2,
            ensure_ascii=False,
            default=str,
        )

        raw = self.llm.call(prompt, max_tokens=300)
        if not raw:
            return fallback

        try:
            cleaned = raw.strip().removeprefix("```json").removesuffix("```").strip()
            parsed = json.loads(cleaned)
            if not isinstance(parsed, list) or len(parsed) < 2:
                return fallback

            hooks = []
            for item in parsed[:3]:
                if not isinstance(item, dict):
                    continue
                text = str(item.get("text", ""))[:100]
                hook_type = str(item.get("type", "unknown"))
                # Check for banned phrases in hooks
                if banned_hits(text):
                    continue
                hooks.append({"type": hook_type, "text": text})

            if len(hooks) < 2:
                return fallback

            return {
                "hooks": hooks,
                "selected_hook": hooks[0]["text"],
                "_from_llm": True,
            }
        except (json.JSONDecodeError, ValueError):
            return fallback

    # ------------------------------------------------------------------
    # Stage 3 — Body Draft
    # ------------------------------------------------------------------

    def _stage_body(
        self,
        topic: str,
        research: Dict[str, Any],
        hooks_result: Dict[str, Any],
        pillar: str,
    ) -> Dict[str, Any]:
        hook = hooks_result["selected_hook"]
        fallback_text = _fallback_body(topic, hook)
        fallback = {"body": fallback_text, "_from_llm": False}

        if not self.llm.available:
            return fallback

        pillar_info = _pillar_by_id(pillar)
        content_rules = _pillars().get("content_rules", {})
        prompt = json.dumps(
            {
                "task": (
                    "Draft a LinkedIn post body. Start with the provided hook line. "
                    "Write 3-5 short paragraphs separated by blank lines. "
                    "MUST include [INSERT YOUR REAL EXAMPLE HERE] and [INSERT SPECIFIC METRIC] as placeholders. "
                    "End with an engagement question. Max 1300 characters total. "
                    "Return only the post text, no JSON wrapper."
                ),
                "hook": hook,
                "research": {
                    "key_insight": research.get("key_insight", ""),
                    "akshay_angle": research.get("akshay_angle", ""),
                    "metric_placeholder": research.get("metric_placeholder", ""),
                },
                "pillar": pillar_info.get("name", pillar),
                "hashtags": (pillar_info.get("hashtags") or [])[:3],
                "format": content_rules.get("format_preferences", []),
                "rules": [
                    "First-person voice. Short sentences. No corporate speak.",
                    "MUST include [INSERT YOUR REAL EXAMPLE HERE] placeholder.",
                    "MUST include [INSERT SPECIFIC METRIC] placeholder.",
                    "Max 2 emojis, only if natural.",
                    "1-3 hashtags at the end, not inline.",
                    "Max 1300 characters total.",
                    "This is a DRAFT — Akshay will fill in real examples.",
                    "Never invent metrics, numbers, or personal stories.",
                ],
                "anti_slop_rules": _anti_slop_rules(),
            },
            indent=2,
            ensure_ascii=False,
            default=str,
        )

        raw = self.llm.call(prompt, max_tokens=500)
        if not raw:
            return fallback

        body = raw.strip()

        # Basic sanity: must have some content
        if len(body) < 50:
            return fallback

        return {"body": body, "_from_llm": True}

    # ------------------------------------------------------------------
    # Stage 4 — Anti-slop Review
    # ------------------------------------------------------------------

    def _stage_anti_slop(self, body: str) -> Dict[str, Any]:
        """Validate draft against all rules. Auto-fix once if needed."""
        validation = validate_draft(body, require_placeholder=True)

        if validation["ok"] and not validation.get("warnings"):
            return {
                "body": body,
                "validation": validation,
                "_from_llm": False,  # validation is deterministic
            }

        # Attempt auto-fix
        fixed = self._auto_fix(body, validation)
        fixed_validation = validate_draft(fixed, require_placeholder=True)

        if fixed_validation["ok"]:
            return {
                "body": fixed,
                "validation": fixed_validation,
                "_from_llm": False,
            }

        # If still failing after fix, try LLM rewrite as last resort
        if self.llm.available:
            rewritten = self._llm_fix(fixed, fixed_validation)
            if rewritten:
                rewrite_validation = validate_draft(rewritten, require_placeholder=True)
                if rewrite_validation["ok"]:
                    return {
                        "body": rewritten,
                        "validation": rewrite_validation,
                        "_from_llm": True,
                    }

        # Return best effort with validation info
        return {
            "body": fixed,
            "validation": fixed_validation,
            "_from_llm": False,
        }

    def _auto_fix(self, body: str, validation: Dict[str, Any]) -> str:
        """Deterministic fixes for common validation failures."""
        fixed = body

        # Fix: banned phrases — remove them
        for error in validation.get("errors", []):
            if error.startswith("banned_phrases:"):
                phrases = error.split(":", 1)[1].split(",")
                for phrase in phrases:
                    fixed = re.sub(re.escape(phrase.strip()), "", fixed, flags=re.IGNORECASE)

        # Fix: missing placeholder — inject one
        if not has_placeholder(fixed):
            # Insert before the last paragraph
            paragraphs = fixed.split("\n\n")
            if len(paragraphs) >= 2:
                paragraphs.insert(-1, "The real number: [INSERT SPECIFIC METRIC].")
            else:
                fixed += "\n\n[INSERT YOUR REAL EXAMPLE HERE]"
            fixed = "\n\n".join(paragraphs)

        # Fix: over 1300 chars — truncate at paragraph boundary
        if len(fixed) > 1300:
            paragraphs = fixed.split("\n\n")
            truncated = []
            char_count = 0
            for para in paragraphs:
                if char_count + len(para) + 2 > 1280:  # leave room for newlines
                    break
                truncated.append(para)
                char_count += len(para) + 2
            # Ensure we keep at least hook + placeholder
            if truncated:
                fixed = "\n\n".join(truncated)
                if not has_placeholder(fixed):
                    fixed += "\n\n[INSERT SPECIFIC METRIC]"

        # Fix: too many hashtags — keep only first 3
        hashtag_pattern = r"#\w+"
        hashtags = re.findall(hashtag_pattern, fixed)
        if len(hashtags) > 4:
            # Remove all hashtags, re-add first 3 at the end
            for tag in hashtags:
                fixed = fixed.replace(tag, "")
            fixed = fixed.rstrip() + "\n\n" + " ".join(hashtags[:3])

        # Clean up excess whitespace from removals
        fixed = re.sub(r"\n{3,}", "\n\n", fixed).strip()

        return fixed

    def _llm_fix(self, body: str, validation: Dict[str, Any]) -> Optional[str]:
        """Ask LLM to fix remaining validation issues."""
        issues = validation.get("errors", []) + validation.get("warnings", [])
        prompt = json.dumps(
            {
                "task": (
                    "Fix this LinkedIn post draft. The issues are listed below. "
                    "Return only the fixed post text. Keep the same structure and voice. "
                    "MUST include [INSERT YOUR REAL EXAMPLE HERE] and [INSERT SPECIFIC METRIC] placeholders."
                ),
                "draft": body,
                "issues": issues,
                "rules": [
                    "First-person voice. Short sentences.",
                    "Max 1300 characters.",
                    "Max 3 hashtags at the end.",
                    "MUST keep [INSERT ...] placeholders.",
                    "Never invent metrics.",
                ],
            },
            indent=2,
            ensure_ascii=False,
            default=str,
        )
        raw = self.llm.call(prompt, max_tokens=500)
        if raw and len(raw.strip()) > 50:
            return raw.strip()
        return None
