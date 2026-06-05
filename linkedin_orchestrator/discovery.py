"""Rotating anchor discovery from already-scraped posts + Apify enrichment.

Strategy: Extract authors from posts scraped by content/engage/watch skills,
then optionally enrich top candidates with Apify profile data. This finds
people who ACTIVELY POST (the best anchors for engagement).
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Set, Tuple

from .common import CONFIG_DIR, PROCESSED_DIR, iso_now, read_json, write_json


ROLE_SCORE: Dict[str, float] = {
    "director": 3.0, "head of product": 3.0, "vp": 2.5,
    "principal": 2.5, "lead": 2.0, "senior": 2.0,
    "group product": 2.5, "product manager": 1.0,
}

DOMAIN_KEYWORDS = [
    "marketplace", "e-commerce", "ecommerce", "fulfillment", "logistics",
    "checkout", "payments", "super app", "superapp", "ai", "growth",
    "mobile", "fintech", "delivery", "qcommerce",
]


class AnchorDiscovery:
    def __init__(self, apify_client):
        self.apify = apify_client

    def _existing_names(self) -> Set[str]:
        targets = read_json(CONFIG_DIR / "targets.json", {})
        names = {"akshay vashishtha"}
        for peer in targets.get("peers", {}).get("profiles", []):
            name = (peer.get("name") or "").lower().strip()
            if name:
                names.add(name)
        for anchor in targets.get("fixed_anchors", {}).get("profiles", []):
            name = (anchor.get("name") or "").lower().strip()
            if name:
                names.add(name)
        for anchor in targets.get("engagement_targets", {}).get("profiles", []):
            name = (anchor.get("name") or "").lower().strip()
            if name:
                names.add(name)
        return names

    def _existing_urls(self) -> Set[str]:
        targets = read_json(CONFIG_DIR / "targets.json", {})
        urls = set()
        for section in ("fixed_anchors", "rotating_anchors"):
            for anchor in targets.get(section, {}).get("profiles", []):
                url = (anchor.get("url") or "").lower().rstrip("/")
                if url:
                    urls.add(url)
        return urls

    def _extract_authors_from_posts(self) -> List[Dict[str, Any]]:
        """Extract unique authors from all scraped post data."""
        authors: Dict[str, Dict[str, Any]] = {}

        for skill in ["content", "engage", "watch"]:
            data = read_json(PROCESSED_DIR / f"{skill}.json", {})
            for item in data.get("items", []):
                author = item.get("author")
                if isinstance(author, dict):
                    name = author.get("name", "")
                elif isinstance(author, str):
                    name = author
                else:
                    continue

                if not name or name == "Unknown" or len(name) < 3:
                    continue

                url = item.get("url") or ""
                # Try to extract profile URL from post URL
                profile_url = ""
                if "/in/" in url:
                    m = re.search(r"(https?://[^/]*linkedin\.com/in/[^/?]+)", url)
                    if m:
                        profile_url = m.group(1) + "/"

                eng_score = item.get("engagement_score", 0)
                relevance = item.get("relevance_reasons", [])

                key = name.lower().strip()
                if key in authors:
                    # Update with higher engagement
                    existing = authors[key]
                    existing["total_engagement"] += eng_score
                    existing["post_count"] += 1
                    existing["all_keywords"].update(relevance)
                    if not existing.get("url") and profile_url:
                        existing["url"] = profile_url
                else:
                    authors[key] = {
                        "name": name,
                        "url": profile_url,
                        "total_engagement": eng_score,
                        "post_count": 1,
                        "all_keywords": set(relevance),
                        "source_skill": skill,
                    }

        return list(authors.values())

    def _score_author(self, author: Dict[str, Any]) -> Tuple[float, List[str]]:
        score = 0.0
        reasons: List[str] = []

        # Engagement volume
        eng = author.get("total_engagement", 0)
        if eng > 5000:
            score += 3.0
            reasons.append("very_high_engagement")
        elif eng > 1000:
            score += 2.0
            reasons.append("high_engagement")
        elif eng > 200:
            score += 1.0
            reasons.append("moderate_engagement")

        # Post frequency
        posts = author.get("post_count", 0)
        if posts >= 3:
            score += 2.0
            reasons.append("prolific_poster")
        elif posts >= 2:
            score += 1.0
            reasons.append("active_poster")

        # Domain keyword coverage
        keywords = author.get("all_keywords", set())
        domain_hits = [kw for kw in DOMAIN_KEYWORDS if kw in keywords]
        if domain_hits:
            score += min(2.0, len(domain_hits) * 0.5)
            reasons.append(f"domain:{','.join(domain_hits[:3])}")

        # Relevance keyword breadth
        if len(keywords) >= 3:
            score += 1.0
            reasons.append("broad_relevance")

        return round(score, 1), reasons

    def search(self, max_searches: int = 5) -> Dict[str, Any]:
        """Extract active authors from scraped posts, score and deduplicate."""
        existing_names = self._existing_names()
        existing_urls = self._existing_urls()

        raw_authors = self._extract_authors_from_posts()

        candidates: List[Dict[str, Any]] = []
        for author in raw_authors:
            name_key = author["name"].lower().strip()
            if name_key in existing_names:
                continue
            url = (author.get("url") or "").lower().rstrip("/")
            if url and url in existing_urls:
                continue

            score, reasons = self._score_author(author)
            if score < 1.0:
                continue  # Skip low-value candidates

            candidates.append({
                "name": author["name"],
                "url": author.get("url") or "",
                "role": "",
                "company": "",
                "location": "",
                "connections": None,
                "relevance_score": score,
                "reasons": reasons,
                "reason": f"Active poster ({author['post_count']} posts, {author['total_engagement']} engagement). {'; '.join(reasons[:3])}",
                "post_count": author["post_count"],
                "total_engagement": author["total_engagement"],
                "keywords": sorted(author.get("all_keywords", set()))[:8],
                "discovered_at": iso_now(),
            })

        # Optionally enrich top candidates with Apify profile scraper
        if self.apify and self.apify.token:
            candidates = self._enrich_top_candidates(candidates[:10])

        candidates.sort(key=lambda p: p["relevance_score"], reverse=True)
        return {
            "candidates": candidates[:15],
            "candidates_found": len(candidates),
            "searches_run": 0,  # No Apify searches for discovery, just post analysis
            "errors": [],
        }

    def _enrich_top_candidates(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Enrich candidates that have LinkedIn URLs with profile data."""
        urls_to_enrich = [c["url"] for c in candidates if c.get("url") and "linkedin.com/in/" in c["url"]]
        if not urls_to_enrich:
            return candidates

        try:
            profiles = self.apify.call(
                "harvestapi~linkedin-profile-scraper",
                {"queries": urls_to_enrich[:5], "profileScraperMode": "Profile details no email ($4 per 1k)"},
                wait=120,
            )
            # Map by URL
            profile_map: Dict[str, Dict] = {}
            for p in profiles:
                if p.get("error") or p.get("status") in (404, 403):
                    continue
                slug = p.get("publicIdentifier", "")
                if slug:
                    profile_map[slug.lower()] = p

            for candidate in candidates:
                url = candidate.get("url", "")
                m = re.search(r"linkedin\.com/in/([^/?]+)", url)
                if not m:
                    continue
                slug = m.group(1).lower()
                profile = profile_map.get(slug)
                if not profile:
                    continue
                candidate["role"] = profile.get("headline") or profile.get("title") or ""
                candidate["company"] = (profile.get("currentPosition") or [{}])[0].get("companyName", "") if isinstance(profile.get("currentPosition"), list) else ""
                candidate["location"] = (profile.get("location") or {}).get("default", "") if isinstance(profile.get("location"), dict) else str(profile.get("location", ""))
                candidate["connections"] = profile.get("connectionsCount")
                candidate["creator"] = profile.get("creator")
                # Boost score for enriched data
                if profile.get("creator"):
                    candidate["relevance_score"] += 1.0
                    candidate["reasons"].append("creator_mode")
                if (profile.get("connectionsCount") or 0) > 5000:
                    candidate["relevance_score"] += 0.5
                    candidate["reasons"].append("large_network")
        except Exception:
            pass  # Enrichment is optional
        return candidates

    def refresh(self) -> Dict[str, Any]:
        """Search, score, and write results to targets.json rotating_anchors."""
        result = self.search()
        targets_path = CONFIG_DIR / "targets.json"
        targets = read_json(targets_path, {})
        targets["rotating_anchors"] = {
            "description": "Discovered from scraped post authors. Refreshed weekly.",
            "last_refreshed": iso_now(),
            "profiles": result["candidates"],
        }
        write_json(targets_path, targets)
        return result
