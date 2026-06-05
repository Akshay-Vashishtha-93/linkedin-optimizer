"""Topic Radar — discover trending topics from already-scraped LinkedIn posts.

Instead of web search (blocked by CAPTCHA), this extracts and clusters topics
from the posts already scraped by content/engage/watch skills. This is free
(no extra API calls) and produces topics actually proven to get engagement.

The LLM clusters and generates angles; deterministic fallback if LLM unavailable.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any, Dict, List, Optional

from .common import (
    CONFIG_DIR,
    PROCESSED_DIR,
    compact_text,
    iso_now,
    read_json,
)
from .validators import RELEVANCE_KEYWORDS, is_relevant_text


# Map keywords to content pillars
PILLAR_KEYWORDS: Dict[str, List[str]] = {
    "Marketplace & Operations": [
        "marketplace", "e-commerce", "ecommerce", "checkout", "cart",
        "fulfillment", "logistics", "delivery", "last mile", "qcommerce",
        "seller", "merchant", "catalog", "order", "payments",
    ],
    "PM Craft & Leadership": [
        "product", "pm", "prioritization", "roadmap", "sprint",
        "stakeholder", "metrics", "okr", "backlog", "agile", "leadership",
        "cross-functional", "discovery", "user research",
    ],
    "AI in Product": [
        "ai", "llm", "machine learning", "gpt", "agent", "automation",
        "artificial intelligence", "claude", "copilot", "generative",
    ],
    "GCC Tech Scene": [
        "dubai", "gcc", "mena", "saudi", "uae", "abu dhabi", "riyadh",
        "startup", "middle east",
    ],
}


def _classify_pillar(text: str) -> str:
    """Classify text into the best-matching content pillar."""
    lowered = text.lower()
    scores: Dict[str, int] = {}
    for pillar, keywords in PILLAR_KEYWORDS.items():
        hits = sum(1 for kw in keywords if kw in lowered)
        scores[pillar] = hits
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "PM Craft & Leadership"


def _extract_topics_from_posts(posts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extract topic signals from scraped post data."""
    topics: List[Dict[str, Any]] = []

    for post in posts:
        content = post.get("content") or post.get("summary") or post.get("post_summary") or ""
        if not content or len(content) < 30:
            continue

        if not is_relevant_text(content):
            continue

        author = post.get("author")
        if isinstance(author, dict):
            author = author.get("name", "Unknown")
        author = author or post.get("author_name", "Unknown")

        engagement = post.get("engagement") or {}
        likes = engagement.get("likes", 0) if isinstance(engagement, dict) else 0
        comments = engagement.get("comments", 0) if isinstance(engagement, dict) else 0
        eng_score = post.get("engagement_score") or post.get("_engagement_score") or (likes + comments * 5)

        first_line = content.split("\n")[0].strip()
        if len(first_line) > 120:
            first_line = first_line[:117] + "..."

        lowered = content.lower()
        matched_keywords = [kw for kw in RELEVANCE_KEYWORDS if kw in lowered]
        pillar = _classify_pillar(content)

        topics.append({
            "title": first_line,
            "snippet": compact_text(content, 300),
            "author": author,
            "pillar": pillar,
            "keywords": matched_keywords[:5],
            "engagement_score": eng_score,
            "source_url": post.get("url") or post.get("linkedinUrl") or post.get("shareLinkedinUrl"),
        })

    return topics


def _cluster_topics(topics: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Cluster similar topics and pick the best representative per cluster."""
    by_pillar: Dict[str, List[Dict[str, Any]]] = {}
    for topic in topics:
        pillar = topic["pillar"]
        by_pillar.setdefault(pillar, []).append(topic)

    clustered: List[Dict[str, Any]] = []
    for pillar, group in by_pillar.items():
        group.sort(key=lambda t: t.get("engagement_score", 0), reverse=True)

        seen_starts: set = set()
        for topic in group:
            start = topic["title"][:40].lower()
            if start in seen_starts:
                continue
            seen_starts.add(start)

            source_urls = [topic["source_url"]] if topic.get("source_url") else []

            all_keywords = Counter()
            for t in group:
                for kw in t.get("keywords", []):
                    all_keywords[kw] += 1

            relevance_score = min(10, round(
                (topic.get("engagement_score", 0) / 500) +
                len(topic.get("keywords", [])) * 0.5 +
                2.0
            , 1))

            clustered.append({
                "title": topic["title"],
                "snippet": topic["snippet"],
                "pillar": pillar,
                "cluster": pillar.split(" ")[0],
                "theme": pillar,
                "author": topic["author"],
                "keywords": topic.get("keywords", []),
                "engagement_score": topic.get("engagement_score", 0),
                "relevance_score": relevance_score,
                "source_urls": source_urls,
                "sources": source_urls,
                "trending_keywords": [kw for kw, _ in all_keywords.most_common(5)],
            })

            if len(clustered) >= 15:
                break

    clustered.sort(key=lambda t: t["relevance_score"], reverse=True)
    return clustered[:12]


class TopicRadar:
    def __init__(self, llm_client=None):
        self.llm = llm_client

    def scan(self) -> Dict[str, Any]:
        """Scan processed posts from content/engage/watch and extract trending topics."""
        all_posts: List[Dict[str, Any]] = []

        for skill in ["content", "engage", "watch"]:
            data = read_json(PROCESSED_DIR / f"{skill}.json", {})
            items = data.get("items", [])
            all_posts.extend(items)

        if not all_posts:
            return {
                "topics": [],
                "total_posts_analyzed": 0,
                "errors": ["no_scraped_posts_available"],
            }

        raw_topics = _extract_topics_from_posts(all_posts)
        clustered = _cluster_topics(raw_topics)

        if self.llm and self.llm.available:
            for topic in clustered:
                angle = self._generate_angle(topic)
                if angle:
                    topic["suggested_angle"] = angle
        else:
            for topic in clustered:
                topic["suggested_angle"] = self._fallback_angle(topic)

        return {
            "topics": clustered,
            "total_posts_analyzed": len(all_posts),
            "pillars_covered": list({t["pillar"] for t in clustered}),
            "errors": [],
        }

    def _generate_angle(self, topic: Dict[str, Any]) -> Optional[str]:
        prompt = json.dumps({
            "task": "Suggest ONE specific angle for Akshay to write a LinkedIn post about this topic. "
                    "Akshay is a Senior PM at Mumzworld (e-commerce, Dubai) with experience at Myntra, ABFRL, Moglix. "
                    "The angle must connect to his real experience. Include [INSERT YOUR REAL EXAMPLE] placeholder. "
                    "Max 2 sentences.",
            "topic": topic["title"],
            "pillar": topic["pillar"],
            "keywords": topic.get("keywords", []),
        }, ensure_ascii=False)

        try:
            result = self.llm.call(prompt, max_tokens=150)
            if result and "[INSERT" in result:
                return result.strip()
        except Exception:
            pass
        return None

    def _fallback_angle(self, topic: Dict[str, Any]) -> str:
        pillar = topic.get("pillar", "")
        keywords = topic.get("keywords", [])
        kw = keywords[0] if keywords else "product"

        angles = {
            "Marketplace & Operations": (
                f"Share your take on {kw} from Mumzworld's marketplace experience. "
                "What metric changed when you [INSERT YOUR REAL EXAMPLE]?"
            ),
            "PM Craft & Leadership": (
                f"Write about a {kw} lesson from your PM career across Myntra, ABFRL, and Mumzworld. "
                "What framework helped you [INSERT YOUR REAL EXAMPLE]?"
            ),
            "AI in Product": (
                f"Share how you're thinking about {kw} at Mumzworld. "
                "What was the result of [INSERT YOUR REAL EXAMPLE]?"
            ),
            "GCC Tech Scene": (
                f"Write about the GCC angle on {kw}. "
                "What's different about building this in Dubai vs India? [INSERT YOUR REAL EXAMPLE]"
            ),
        }
        return angles.get(pillar, f"Write about {kw} with a real example from [INSERT YOUR REAL EXAMPLE].")
