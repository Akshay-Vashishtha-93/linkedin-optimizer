from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RUNS_DIR = DATA_DIR / "runs"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = ROOT / "reports"
CONFIG_DIR = ROOT / "config"

SKILL_PATHS = {
    "profile": Path("/Users/akshay/.agents/skills/linkedin-profile/SKILL.md"),
    "jobs": Path("/Users/akshay/.agents/skills/linkedin-jobs/SKILL.md"),
    "network": Path("/Users/akshay/.agents/skills/linkedin-network/SKILL.md"),
    "content": Path("/Users/akshay/.agents/skills/linkedin-content/SKILL.md"),
    "engage": Path("/Users/akshay/.agents/skills/linkedin-engage/SKILL.md"),
    "watch": Path("/Users/akshay/.agents/skills/linkedin-watch/SKILL.md"),
    "scorecard": Path("/Users/akshay/.agents/skills/linkedin-scorecard/SKILL.md"),
}

FRESHNESS_DAYS = {
    "jobs": 1,
    "engage": 3,
    "profile": 14,
    "network": 30,
    "watch": 7,
    "content": 14,
    "scorecard": 7,
    "topics": 3,
}


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def iso_now() -> str:
    return now_utc().isoformat()


def parse_dt(value: Any) -> Optional[datetime]:
    if not value:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, (int, float)):
        if value > 10_000_000_000:
            value = value / 1000
        return datetime.fromtimestamp(value, timezone.utc)
    text = str(value).strip()
    if not text:
        return None
    text = text.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def age_days(value: Any) -> Optional[float]:
    parsed = parse_dt(value)
    if not parsed:
        return None
    return max(0.0, (now_utc() - parsed).total_seconds() / 86400)


def freshness_status(skill: str, value: Any) -> str:
    days = age_days(value)
    if days is None:
        return "unknown"
    limit = FRESHNESS_DAYS.get(skill, 7)
    if days <= limit:
        return "fresh"
    if days <= limit * 2:
        return "stale"
    return "expired"


def read_json(path: Path, default: Any = None) -> Any:
    try:
        if path.exists():
            return json.loads(path.read_text())
    except json.JSONDecodeError:
        return default
    return default


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str) + "\n")


def cfg(name: str, default: Any = None) -> Any:
    return read_json(CONFIG_DIR / name, default)


def data_path(relative: str) -> Path:
    return DATA_DIR / relative


def processed_path(skill: str) -> Path:
    return PROCESSED_DIR / f"{skill}.json"


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "item"


def stable_id(*parts: Any) -> str:
    payload = "|".join("" if part is None else str(part) for part in parts)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def normalize_actor_id(actor_id: str) -> str:
    return actor_id.replace("/", "~")


def redact(text: str) -> str:
    text = re.sub(r"sk-[A-Za-z0-9_-]{20,}", "[REDACTED_KEY]", text)
    text = re.sub(r"apify_api_[A-Za-z0-9_-]+", "[REDACTED_APIFY_TOKEN]", text)
    return text


def compact_text(value: Any, limit: int = 260) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text[:limit] + ("..." if len(text) > limit else "")


def skill_spec(skill: str) -> Dict[str, Any]:
    path = SKILL_PATHS.get(skill)
    if not path or not path.exists():
        return {
            "path": str(path) if path else None,
            "available": False,
            "sha1": None,
            "summary": "Skill spec not found",
        }
    text = path.read_text()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    summary = next((line for line in lines if line.startswith("description:")), "")
    if not summary:
        summary = next((line for line in lines if line.startswith("# ")), "")
    return {
        "path": str(path),
        "available": True,
        "sha1": hashlib.sha1(text.encode("utf-8")).hexdigest()[:12],
        "summary": redact(summary.replace("description:", "").strip()),
    }


def evidence(
    *,
    source_type: str,
    source_url: Optional[str],
    actor: Optional[str],
    raw_path: Optional[Path],
    scraped_at: Optional[Any],
    confidence: float,
    verified_fields: Iterable[str],
    unknown_fields: Iterable[str] = (),
    notes: Iterable[str] = (),
) -> Dict[str, Any]:
    scraped = parse_dt(scraped_at) or now_utc()
    return {
        "source_type": source_type,
        "source_url": source_url,
        "actor": actor,
        "scraped_at": scraped.isoformat(),
        "raw_path": str(raw_path.relative_to(ROOT)) if raw_path and raw_path.is_absolute() else str(raw_path) if raw_path else None,
        "freshness_status": "unknown",
        "confidence": round(max(0.0, min(1.0, confidence)), 2),
        "verified_fields": sorted(set(verified_fields)),
        "unknown_fields": sorted(set(unknown_fields)),
        "notes": list(notes),
    }


def attach_freshness(skill: str, ev: Dict[str, Any]) -> Dict[str, Any]:
    ev = dict(ev)
    ev["freshness_status"] = freshness_status(skill, ev.get("scraped_at"))
    return ev


def latest_run_files(limit: int = 12) -> list[Dict[str, Any]]:
    if not RUNS_DIR.exists():
        return []
    items = []
    for path in RUNS_DIR.glob("*/run.json"):
        data = read_json(path, {})
        if data:
            items.append(data)
    items.sort(key=lambda x: x.get("started_at", ""), reverse=True)
    return items[:limit]


def safe_env(name: str) -> Optional[str]:
    value = os.environ.get(name)
    return value.strip() if value and value.strip() else None
