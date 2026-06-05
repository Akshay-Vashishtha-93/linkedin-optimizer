from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List


BANNED_PHRASES = [
    "in today's fast-paced world",
    "game-changer",
    "leverage",
    "synergy",
    "delighted to share",
    "thrilled to announce",
    "it's not about",
    "great post",
    "love this",
    "so true",
]

RELEVANCE_KEYWORDS = [
    "product",
    "pm",
    "marketplace",
    "ecommerce",
    "e-commerce",
    "commerce",
    "checkout",
    "cart",
    "fulfillment",
    "logistics",
    "delivery",
    "qcommerce",
    "super app",
    "super-app",
    "mobile",
    "growth",
    "ai",
    "llm",
    "data",
    "experiments",
    "dubai",
    "gcc",
    "mena",
    "payments",
    "user",
    "customers",
    "startup",
    "saas",
    "fintech",
]

OFF_TOPIC_TERMS = [
    "post-production",
    "post production",
    "film",
    "netflix",
    "cinema",
    "tvindustry",
    "audiovisual",
    "vfx",
    "masterclass",
    "coordinacion",
    "posproduccion",
]


def lower_text(value: Any) -> str:
    return str(value or "").lower()


def has_placeholder(text: str) -> bool:
    return bool(re.search(r"\[[^\]]*(insert|placeholder|specific|real|metric|example)[^\]]*\]", text, re.I))


def banned_hits(text: str) -> List[str]:
    lowered = lower_text(text)
    return [phrase for phrase in BANNED_PHRASES if phrase in lowered]


def validate_draft(text: str, *, require_placeholder: bool = True) -> Dict[str, Any]:
    errors: List[str] = []
    warnings: List[str] = []
    if not text or not text.strip():
        errors.append("empty_draft")
    hits = banned_hits(text)
    if hits:
        errors.append("banned_phrases:" + ",".join(hits))
    if require_placeholder and not has_placeholder(text):
        errors.append("missing_required_placeholder")
    if len(text) > 1300:
        warnings.append("over_1300_chars")
    if text.count("#") > 4:
        warnings.append("too_many_hashtags")
    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
    }


def is_relevant_text(text: str) -> bool:
    lowered = lower_text(text)
    if any(term in lowered for term in OFF_TOPIC_TERMS):
        return False
    return any(keyword in lowered for keyword in RELEVANCE_KEYWORDS)


def relevance_reasons(text: str, limit: int = 5) -> List[str]:
    lowered = lower_text(text)
    return [kw for kw in RELEVANCE_KEYWORDS if kw in lowered][:limit]


def validate_evidence(evidence: Dict[str, Any]) -> Dict[str, Any]:
    errors: List[str] = []
    warnings: List[str] = []
    if not evidence.get("source_type"):
        errors.append("missing_source_type")
    if not evidence.get("raw_path"):
        warnings.append("missing_raw_path")
    if not evidence.get("scraped_at"):
        errors.append("missing_scraped_at")
    confidence = evidence.get("confidence")
    if confidence is None:
        warnings.append("missing_confidence")
    elif confidence < 0.5:
        warnings.append("low_confidence")
    if evidence.get("freshness_status") in {"stale", "expired"}:
        warnings.append("stale_evidence")
    return {"ok": not errors, "errors": errors, "warnings": warnings}


def summarize_validation(items: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    errors: List[str] = []
    warnings: List[str] = []
    for item in items:
        errors.extend(item.get("errors", []))
        warnings.extend(item.get("warnings", []))
    return {
        "ok": not errors,
        "errors": sorted(set(errors)),
        "warnings": sorted(set(warnings)),
    }


def validation_result(errors: Iterable[str] = (), warnings: Iterable[str] = ()) -> Dict[str, Any]:
    err = sorted(set(errors))
    warn = sorted(set(warnings))
    return {"ok": not err, "errors": err, "warnings": warn}
