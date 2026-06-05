from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from .apify_client import ApifyClient, ApifyError
from .discovery import AnchorDiscovery
from .common import (
    CONFIG_DIR,
    DATA_DIR,
    PROCESSED_DIR,
    ROOT,
    RUNS_DIR,
    attach_freshness,
    cfg,
    compact_text,
    data_path,
    evidence,
    freshness_status,
    iso_now,
    latest_run_files,
    now_utc,
    processed_path,
    read_json,
    redact,
    skill_spec,
    stable_id,
    write_json,
)
from .scoring import (
    DIMENSION_LABELS,
    canonical_overall,
    dedupe_jobs,
    parse_benchmark_report,
    rank_posts,
    score_job,
    score_network_target,
    score_profile,
)
from .content_lab import ContentLab
from .topic_radar import TopicRadar
from .validators import (
    banned_hits,
    has_placeholder,
    is_relevant_text,
    relevance_reasons,
    validate_draft,
    validate_evidence,
    validation_result,
)


URLS = {
    "Sanket Purohit": "https://www.linkedin.com/in/sanketpurohit/",
    "Diego Granados": "https://www.linkedin.com/in/diegogranados/",
    "Sajjad Ahmad": "https://www.linkedin.com/in/sajjadahmadpm/",
    "Raeesa Omar": "https://www.linkedin.com/in/raeesaomar/",
    "Dharani Dharan G": "https://www.linkedin.com/in/dharanidharan/",
    "Shradha Mohanan": "https://www.linkedin.com/in/shradhamohanan/",
    "Akshay Vashishtha": "https://www.linkedin.com/in/akshay-vashishtha/",
    "Lenny Rachitsky": "https://www.linkedin.com/in/lennyrachitsky/",
    "Shreyas Doshi": "https://www.linkedin.com/in/shreyasdoshi/",
    "Aatir Abdul Rauf": "https://www.linkedin.com/in/aatirabdulrauf/",
    "Pawel Huryn": "https://www.linkedin.com/in/pawelhuryn/",
    "Ethan Mollick": "https://www.linkedin.com/in/emollick/",
    "Hussain Abbasi": "https://www.linkedin.com/in/hussainabbasi/",
    "Pragya Pande": "https://www.linkedin.com/in/pragyapande/",
}


def _anchor_urls() -> List[str]:
    """Get URLs from fixed_anchors + rotating_anchors in targets.json."""
    targets = cfg("targets.json", {})
    urls = []
    for anchor in targets.get("fixed_anchors", {}).get("profiles", []):
        url = anchor.get("url")
        if url:
            urls.append(url)
    for anchor in targets.get("rotating_anchors", {}).get("profiles", []):
        url = anchor.get("url")
        if url:
            urls.append(url)
    return urls


def _engagement_urls() -> List[str]:
    """Get URLs for engagement targets (GCC peers to comment on)."""
    targets = cfg("targets.json", {})
    urls = []
    for target in targets.get("engagement_targets", {}).get("profiles", []):
        name = target.get("name")
        url = URLS.get(name)
        if url:
            urls.append(url)
    return urls


class LLMClient:
    def __init__(self):
        self.api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
        self.model = os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def call(self, prompt: str, max_tokens: int = 400) -> Optional[str]:
        if not self.available:
            return None
        try:
            import anthropic

            client = anthropic.Anthropic(api_key=self.api_key)
            resp = client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=(
                    "You help Akshay Vashishtha with LinkedIn growth. "
                    "Only use facts in the prompt. Unknown facts must become [INSERT ...] placeholders. "
                    "Never invent metrics, job status, salaries, or people."
                ),
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.content[0].text if resp.content else None
        except Exception:
            return None


class RunContext:
    def __init__(self, skill: str, cost_estimate: Dict[str, Any]):
        self.skill = skill
        self.run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{skill}-{stable_id(skill, iso_now())[:8]}"
        self.run_dir = RUNS_DIR / self.run_id
        self.started_at = iso_now()
        self.cost_estimate = cost_estimate
        self.logs: List[str] = []
        self.warnings: List[str] = list(cost_estimate.get("warnings", []))
        self.errors: List[str] = []
        self.raw_path = self.run_dir / "raw.json"
        self.processed_path = self.run_dir / "processed.json"
        self.validation_path = self.run_dir / "validation.json"
        self.run_path = self.run_dir / "run.json"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.log("run_started")
        self.save_run("running")

    def log(self, message: str) -> None:
        self.logs.append(f"{iso_now()} {redact(message)}")

    def save_raw(self, data: Any) -> Path:
        write_json(self.raw_path, data)
        return self.raw_path

    def save_processed(self, data: Dict[str, Any]) -> Path:
        write_json(self.processed_path, data)
        write_json(processed_path(self.skill), data)
        return self.processed_path

    def save_validation(self, data: Dict[str, Any]) -> Path:
        write_json(self.validation_path, data)
        return self.validation_path

    def save_run(self, status: str) -> Dict[str, Any]:
        payload = {
            "run_id": self.run_id,
            "skill": self.skill,
            "status": status,
            "started_at": self.started_at,
            "finished_at": iso_now() if status != "running" else None,
            "cost_estimate": self.cost_estimate,
            "skill_spec": skill_spec(self.skill),
            "agent_steps": [
                "skill_gateway",
                "apify_or_cache_scraper",
                "deterministic_scorer",
                "llm_drafter_if_available",
                "evidence_validator",
            ],
            "raw_path": str(self.raw_path.relative_to(ROOT)) if self.raw_path.exists() else None,
            "processed_path": str(self.processed_path.relative_to(ROOT)) if self.processed_path.exists() else None,
            "validation_path": str(self.validation_path.relative_to(ROOT)) if self.validation_path.exists() else None,
            "logs": self.logs,
            "warnings": sorted(set(self.warnings)),
            "errors": sorted(set(self.errors)),
        }
        write_json(self.run_path, payload)
        return payload


class SkillGateway:
    def __init__(self, root: Path = ROOT):
        self.root = root
        self.apify = ApifyClient(root)
        self.llm = LLMClient()
        self.content_lab = ContentLab(self.llm)
        self.handlers: Dict[str, Tuple[Callable[[RunContext], Dict[str, Any]], Iterable[str]]] = {
            "profile": (self.run_profile, ["profile_scraper"]),
            "jobs": (self.run_jobs, ["job_scraper"]),
            "network": (self.run_network, []),
            "content": (self.run_content, ["profile_posts"]),
            "engage": (self.run_engage, ["profile_posts"]),
            "watch": (self.run_watch, ["profile_posts"]),
            "scorecard": (self.run_scorecard, []),
            "discover": (self.run_discover_anchors, []),
            "topics": (self.run_topics, []),
        }

    def run(self, skill: str) -> Dict[str, Any]:
        if skill not in self.handlers:
            raise KeyError(f"Unknown skill: {skill}")
        handler, cost_keys = self.handlers[skill]
        budget = self.apify.estimate(cost_keys).as_dict()
        ctx = RunContext(skill, budget)
        try:
            processed = handler(ctx)
            validation = self.validate_processed(skill, processed)
            processed["validation"] = validation
            processed["run_id"] = ctx.run_id
            processed["cost_estimate"] = budget
            ctx.save_processed(processed)
            ctx.save_validation(validation)
            status = "success" if validation.get("ok") else "partial"
            run = ctx.save_run(status)
            return {"run": run, "processed": processed}
        except Exception as exc:
            ctx.errors.append(str(exc))
            ctx.log(f"run_failed:{exc}")
            validation = validation_result(errors=[str(exc)], warnings=ctx.warnings)
            ctx.save_validation(validation)
            run = ctx.save_run("failed")
            return {"run": run, "processed": self.empty_processed(skill, errors=[str(exc)], warnings=ctx.warnings)}

    def empty_processed(self, skill: str, errors: Iterable[str] = (), warnings: Iterable[str] = ()) -> Dict[str, Any]:
        return {
            "skill": skill,
            "generated_at": iso_now(),
            "status": "failed" if list(errors) else "empty",
            "freshness_status": "unknown",
            "coverage": 0,
            "warnings": list(warnings),
            "errors": list(errors),
            "items": [],
            "summary": "No verified data available.",
        }

    def evidence_bound_draft(
        self,
        *,
        skill: str,
        task: str,
        evidence_payload: Dict[str, Any],
        fallback: str,
        require_placeholder: bool = True,
        max_tokens: int = 450,
    ) -> str:
        """Ask Claude for wording only after evidence is already verified.

        The validator is the hard stop: if Claude returns generic or unsupported
        copy, the deterministic fallback is used instead.
        """
        if not self.llm.available:
            return fallback
        prompt = json.dumps(
            {
                "task": task,
                "skill_spec": skill_spec(skill),
                "evidence": evidence_payload,
                "rules": [
                    "Use only the evidence provided here.",
                    "Do not invent metrics, salaries, hiring managers, job status, company claims, or personal experience.",
                    "Unknown facts must remain as [INSERT ...] placeholders.",
                    "Use Akshay's first-person voice. Short sentences. No corporate speak.",
                ],
            },
            indent=2,
            ensure_ascii=False,
            default=str,
        )
        draft = self.llm.call(prompt, max_tokens=max_tokens)
        if not draft:
            return fallback
        draft = draft.strip()
        result = validate_draft(draft, require_placeholder=require_placeholder)
        if result["errors"]:
            return fallback
        return draft

    def run_discover_anchors(self, ctx: RunContext) -> Dict[str, Any]:
        """Discover new rotating anchors via Apify LinkedIn profile search."""
        ctx.log("anchor_discovery_started")
        discovery = AnchorDiscovery(self.apify)
        result = discovery.refresh()
        ctx.log(f"anchor_discovery_complete:found={result['candidates_found']},searches={result['searches_run']}")
        ctx.save_raw(result)
        candidates = result.get("candidates", [])
        if result.get("errors"):
            ctx.warnings.extend(result["errors"])
        return {
            "skill": "discover",
            "generated_at": iso_now(),
            "status": "success" if candidates else "empty",
            "freshness_status": "fresh",
            "coverage": min(1.0, len(candidates) / 10),
            "warnings": [] if candidates else ["no_candidates_found"],
            "errors": result.get("errors", []),
            "items": candidates,
            "summary": (
                f"Discovered {len(candidates)} rotating anchor candidates "
                f"from {result['searches_run']} Apify searches."
            ),
            "evidence": [],
        }

    def dashboard_state(self) -> Dict[str, Any]:
        self.bootstrap_processed()
        skills = {}
        for skill in self.handlers:
            data = read_json(processed_path(skill), None)
            skills[skill] = data if data else self.empty_processed(skill, warnings=["not_run_yet"])
        apify_usage = self.apify.live_usage()
        return {
            "generated_at": iso_now(),
            "budget": {
                "plan": apify_usage.get("plan", "unknown"),
                "usage_usd": apify_usage.get("usage_usd", 0),
                "limit_usd": apify_usage.get("limit_usd", 29.0),
                "remaining_usd": apify_usage.get("remaining_usd", 29.0),
                "pct_used": apify_usage.get("pct_used", 0),
                "is_exhausted": apify_usage.get("is_exhausted", False),
                "usage_error": apify_usage.get("error"),
            },
            "llm_available": self.llm.available,
            "skills": skills,
            "recent_runs": latest_run_files(),
        }

    def bootstrap_processed(self) -> None:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        builders = {
            "profile": self.cached_profile_processed,
            "jobs": self.cached_jobs_processed,
            "network": self.cached_network_processed,
            "content": self.cached_content_processed,
            "engage": self.cached_engage_processed,
            "watch": self.cached_watch_processed,
            "scorecard": self.cached_scorecard_processed,
            "topics": self.cached_topics_processed,
        }
        for skill, builder in builders.items():
            if not processed_path(skill).exists():
                write_json(processed_path(skill), builder())

    def validate_processed(self, skill: str, processed: Dict[str, Any]) -> Dict[str, Any]:
        errors: List[str] = []
        warnings: List[str] = list(processed.get("warnings", []))
        evidences = processed.get("evidence", [])
        for ev in evidences:
            result = validate_evidence(ev)
            errors.extend(result["errors"])
            warnings.extend(result["warnings"])
        for item in processed.get("items", []):
            ev = item.get("evidence")
            if ev:
                result = validate_evidence(ev)
                errors.extend(result["errors"])
                warnings.extend(result["warnings"])
        if skill in {"content", "engage", "jobs", "network"}:
            for item in processed.get("items", []):
                text = item.get("draft") or item.get("comment_draft") or item.get("outreach_draft") or item.get("connection_note")
                if text:
                    draft_result = validate_draft(text, require_placeholder=True)
                    errors.extend(draft_result["errors"])
                    warnings.extend(draft_result["warnings"])
        if processed.get("coverage", 0) < 0.4:
            warnings.append("low_source_coverage")
        return {
            "ok": not errors,
            "errors": sorted(set(errors)),
            "warnings": sorted(set(warnings)),
        }

    def run_profile(self, ctx: RunContext) -> Dict[str, Any]:
        actor = self.apify.actor_id("profile_scraper")
        raw_source = "cache"
        items: List[Dict[str, Any]]
        # Optimize: only scrape Akshay ($0.004). Peers are static in config.
        input_data = {
            "queries": [URLS["Akshay Vashishtha"]],
            "profileScraperMode": "Profile details no email ($4 per 1k)",
        }
        try:
            ctx.log(f"apify_profile_scrape actor={actor}")
            items = self.apify.call(actor, input_data, wait=180)
            raw_source = "apify"
        except Exception as exc:
            ctx.warnings.append("profile_scrape_failed_using_cache")
            ctx.log(f"profile_scrape_failed:{exc}")
            cached = read_json(data_path("profile/akshay-latest.json"), {})
            items = [cached] if cached else []
        raw_path = ctx.save_raw({"source": raw_source, "items": items})
        # Filter out error items (404, 403) and find Akshay's profile
        valid = [item for item in items if item.get("status") not in (404, 403, "404", "403") and not item.get("error")]
        akshay = next(
            (item for item in valid if "akshay" in (item.get("firstName") or "").lower() or "akshay" in (item.get("publicIdentifier") or "").lower()),
            valid[0] if valid else {},
        )
        if raw_source == "apify" and akshay:
            write_json(data_path("profile/akshay-latest.json"), akshay)
        return self.build_profile_processed(akshay, raw_path, actor, raw_source)

    def build_profile_processed(self, profile: Dict[str, Any], raw_path: Path, actor: str, source: str) -> Dict[str, Any]:
        baseline = parse_benchmark_report()
        scored = score_profile(profile, baseline)
        ev = attach_freshness(
            "profile",
            evidence(
                source_type=source,
                source_url=profile.get("linkedinUrl") or URLS["Akshay Vashishtha"],
                actor=actor if source == "apify" else None,
                raw_path=raw_path,
                scraped_at=iso_now() if source == "apify" else "2026-05-15T00:00:00+00:00",
                confidence=0.92 if source == "apify" else 0.72,
                verified_fields=["headline", "about", "connectionsCount", "followerCount", "premium", "experience"],
                unknown_fields=scored["unknown_fields"],
                notes=["missing_fields_are_unknown_not_negative"],
            ),
        )
        peers = self.peer_rows()
        actions = [
            {"title": "Turn on creator mode", "impact": "+1.0", "status": "manual_action", "reason": "Content/Creator Presence is the largest verified gap."},
            {"title": "Add 2-3 featured items", "impact": "+0.5", "status": "manual_action", "reason": "Featured content is missing or unknown in the scrape."},
            {"title": "Post 2-3x/week", "impact": "+1.5 over 8 weeks", "status": "manual_action", "reason": "The benchmark report identifies content as the main 9+ lever."},
            {"title": "Daily comments on target PM posts", "impact": "+0.5", "status": "manual_action", "reason": "Engagement grows visibility before connection requests."},
        ]
        self.repair_benchmark_history(scored, profile)
        return {
            "skill": "profile",
            "generated_at": iso_now(),
            "status": "success" if profile else "partial",
            "freshness_status": ev["freshness_status"],
            "coverage": 0.9 if profile else 0.2,
            "summary": f"Canonical profile score: {scored['overall_score']}/10. Content and creator presence remain the main gap.",
            "profile": {
                "name": "Akshay Vashishtha",
                "headline": profile.get("headline"),
                "connections": profile.get("connectionsCount"),
                "followers": profile.get("followerCount"),
                "premium": profile.get("premium"),
                "creator": profile.get("creator"),
            },
            "score": scored,
            "peers": peers,
            "actions": actions,
            "evidence": [ev],
            "warnings": [] if profile else ["no_profile_data"],
            "errors": [],
        }

    def repair_benchmark_history(self, scored: Dict[str, Any], profile: Dict[str, Any]) -> None:
        hist_path = data_path("profile/benchmark-history.json")
        hist = read_json(hist_path, {"history": []})
        history = hist.setdefault("history", [])
        entry = {
            "date": "2026-05-15",
            "overall_score": scored["overall_score"],
            "dimensions": scored["dimensions"],
            "connections": profile.get("connectionsCount", 5701),
            "followers": profile.get("followerCount", 5674),
            "source": "benchmark_report_canonical",
            "notes": "Repaired bad dashboard_sync overwrite; missing scrape fields treated as unknown.",
        }
        replaced = False
        for index, item in enumerate(history):
            if item.get("date") == "2026-05-15":
                if (
                    item.get("source") == "dashboard_sync"
                    or item.get("overall_score") != scored["overall_score"]
                    or item.get("dimensions") != scored["dimensions"]
                ):
                    history[index] = entry
                replaced = True
        if not replaced:
            history.append(entry)
        write_json(hist_path, hist)

    def peer_rows(self) -> List[Dict[str, Any]]:
        targets = cfg("targets.json", {})
        rows = []
        for peer in targets.get("peers", {}).get("profiles", []):
            rows.append(
                {
                    "name": peer.get("name"),
                    "company": peer.get("company"),
                    "role": peer.get("role"),
                    "connections": peer.get("connections"),
                    "followers": peer.get("followers"),
                    "creator": peer.get("has_creator_mode"),
                    "premium": peer.get("has_premium"),
                    "priority": peer.get("priority"),
                    "evidence_type": "config_peer_snapshot",
                }
            )
        rows.sort(key=lambda item: item.get("followers") or 0, reverse=True)
        return rows

    def cached_profile_processed(self) -> Dict[str, Any]:
        profile = read_json(data_path("profile/akshay-latest.json"), {})
        raw_path = data_path("profile/akshay-latest.json")
        return self.build_profile_processed(profile, raw_path, "cached_profile", "cache")

    def run_jobs(self, ctx: RunContext) -> Dict[str, Any]:
        actor = self.apify.actor_id("job_scraper")
        fallback_actor = self.apify.actor_id("job_scraper_fallback")
        search_urls = self.job_search_urls()
        items: List[Dict[str, Any]] = []
        source = "cache"
        try:
            ctx.log(f"apify_jobs_scrape actor={actor}")
            items = self.apify.call(actor, {"urls": search_urls[:4], "count": 25, "scrapeCompany": False}, wait=180)
            source = "apify"
        except Exception as first_exc:
            ctx.log(f"primary_job_actor_failed:{first_exc}")
            try:
                ctx.log(f"apify_jobs_scrape fallback_actor={fallback_actor}")
                items = self.apify.call(
                    fallback_actor,
                    {"job_title": "Senior Product Manager", "location": "United Arab Emirates", "jobs_entries": 50},
                    wait=300,
                )
                source = "apify_fallback"
            except Exception as second_exc:
                ctx.warnings.append("job_scrape_failed_using_cache")
                ctx.log(f"fallback_job_actor_failed:{second_exc}")
                cached = read_json(data_path("jobs/latest.json"), [])
                items = cached if isinstance(cached, list) else []
        raw_path = ctx.save_raw({"source": source, "items": items, "search_urls": search_urls})
        if source.startswith("apify") and items:
            write_json(data_path("jobs/latest.json"), items)
        return self.build_jobs_processed(items, raw_path, actor if source == "apify" else fallback_actor, source)

    def job_search_urls(self) -> List[str]:
        filters = cfg("job-filters.json", {})
        roles = filters.get("target_roles", ["Senior Product Manager"])[:4]
        locations = ["Dubai%2C+United+Arab+Emirates", "United+Arab+Emirates", "Riyadh%2C+Saudi+Arabia"]
        urls = []
        for role in roles:
            for location in locations[:2]:
                urls.append(
                    "https://www.linkedin.com/jobs/search/?"
                    f"keywords={role.replace(' ', '+')}&location={location}&f_TPR=r604800"
                )
        for company in filters.get("target_companies", [])[:8]:
            urls.append(
                "https://www.linkedin.com/jobs/search/?"
                f"keywords=Product+Manager+{str(company).split(' / ')[0].replace(' ', '+')}&location=United+Arab+Emirates&f_TPR=r2592000"
            )
        return urls

    def build_jobs_processed(self, jobs: List[Dict[str, Any]], raw_path: Path, actor: str, source: str) -> Dict[str, Any]:
        filters = cfg("job-filters.json", {})
        target_companies = filters.get("target_companies", [])
        out = []
        for job in dedupe_jobs(jobs):
            scored = score_job(job, target_companies)
            if scored["fit_score"] < 35:
                continue
            posted = job.get("postedAt") or job.get("listedAt") or iso_now()
            ev = attach_freshness(
                "jobs",
                evidence(
                    source_type=source,
                    source_url=job.get("link") or job.get("applyUrl"),
                    actor=actor if source.startswith("apify") else None,
                    raw_path=raw_path,
                    scraped_at=posted if source == "cache" else iso_now(),
                    confidence=0.88 if source.startswith("apify") else 0.68,
                    verified_fields=["title", "companyName", "location", "descriptionText", "postedAt"],
                    unknown_fields=[field for field in ["salary", "jobPosterName"] if not job.get(field)],
                ),
            )
            out.append(
                {
                    "id": scored["job_id"],
                    "title": job.get("title"),
                    "company": job.get("companyName"),
                    "location": job.get("location"),
                    "posted_at": job.get("postedAt"),
                    "url": job.get("link") or job.get("applyUrl"),
                    "company_url": job.get("companyLinkedinUrl"),
                    "poster": job.get("jobPosterName"),
                    "poster_url": job.get("jobPosterProfileUrl"),
                    "description": compact_text(job.get("descriptionText"), 420),
                    "fit_score": scored["fit_score"],
                    "fit_level": scored["fit_level"],
                    "reasons": scored["reasons"],
                    "risk_flags": scored["risk_flags"],
                    "domain_matches": scored["domain_matches"],
                    "outreach_draft": self.job_outreach(job, scored),
                    "referral_targets": self.referral_targets(job.get("companyName")),
                    "evidence": ev,
                }
            )
        out.sort(key=lambda item: item["fit_score"], reverse=True)
        warnings = []
        if not out:
            warnings.append("no_verified_matching_jobs")
        if source == "cache":
            warnings.append("using_cached_jobs")
        return {
            "skill": "jobs",
            "generated_at": iso_now(),
            "status": "success" if out else "partial",
            "freshness_status": freshness_status("jobs", out[0]["evidence"]["scraped_at"] if out else None),
            "coverage": min(1.0, len(out) / 10) if out else 0,
            "summary": f"{len(out)} relevant PM jobs ranked from {len(jobs)} scraped/cached records.",
            "items": out[:30],
            "evidence": [],
            "warnings": warnings,
            "errors": [],
        }

    def job_outreach(self, job: Dict[str, Any], scored: Dict[str, Any]) -> str:
        company = job.get("companyName") or "[INSERT COMPANY]"
        title = job.get("title") or "[INSERT ROLE]"
        domains = ", ".join(scored.get("domain_matches", [])[:3]) or "[INSERT ROLE-SPECIFIC DOMAIN]"
        fallback = (
            f"Cover opener: I am interested in {title} at {company} because the role maps to my marketplace, "
            f"checkout, fulfillment, and GCC consumer-tech experience. At Mumzworld, I worked on [INSERT YOUR REAL EXAMPLE] "
            f"that improved [INSERT SPECIFIC METRIC]. The {domains} angle is where I can add the most context.\n\n"
            f"Connection note: Hi [Name], saw the {title} role at {company}. I have built marketplace/commerce products across Mumzworld, "
            f"Myntra and ABFRL, including [INSERT SPECIFIC ACHIEVEMENT]. Would value connecting."
        )
        return self.evidence_bound_draft(
            skill="jobs",
            task="Draft a short cover opener and a short connection note for this verified job. Keep placeholders where Akshay must add real proof.",
            evidence_payload={
                "title": job.get("title"),
                "company": job.get("companyName"),
                "location": job.get("location"),
                "description": compact_text(job.get("descriptionText"), 1200),
                "fit_reasons": scored.get("reasons", []),
                "risk_flags": scored.get("risk_flags", []),
                "domain_matches": scored.get("domain_matches", []),
            },
            fallback=fallback,
        )

    def referral_targets(self, company: Optional[str]) -> List[Dict[str, str]]:
        company = company or "[Company]"
        encoded = company.replace(" ", "%20")
        return [
            {"label": "Head of Product search", "query": f"Head of Product {company}", "url": f"https://www.linkedin.com/search/results/people/?keywords=Head%20of%20Product%20{encoded}"},
            {"label": "Product Manager first-degree search", "query": f"Product Manager {company}", "url": f"https://www.linkedin.com/search/results/people/?keywords=Product%20Manager%20{encoded}&network=%5B%22F%22%5D"},
        ]

    def cached_jobs_processed(self) -> Dict[str, Any]:
        jobs = read_json(data_path("jobs/latest.json"), [])
        jobs = jobs if isinstance(jobs, list) else []
        return self.build_jobs_processed(jobs, data_path("jobs/latest.json"), "cached_jobs", "cache")

    def run_network(self, ctx: RunContext) -> Dict[str, Any]:
        raw = self.network_universe()
        raw_path = ctx.save_raw(raw)
        return self.build_network_processed(raw, raw_path)

    def network_universe(self) -> Dict[str, Any]:
        targets = cfg("targets.json", {})
        jobs = read_json(processed_path("jobs"), None) or self.cached_jobs_processed()
        peers = targets.get("peers", {}).get("profiles", [])
        engagement = targets.get("engagement_targets", {}).get("profiles", [])
        posters = []
        for job in jobs.get("items", []):
            if job.get("poster"):
                posters.append(
                    {
                        "name": job.get("poster"),
                        "role": "Recruiter / Job poster",
                        "company": job.get("company"),
                        "reason": f"Posted relevant role: {job.get('title')}",
                        "url": job.get("poster_url"),
                    }
                )
        return {"peers": peers, "engagement_targets": engagement, "job_posters": posters}

    def build_network_processed(self, raw: Dict[str, Any], raw_path: Path) -> Dict[str, Any]:
        filters = cfg("job-filters.json", {})
        people = []
        for source_key in ["job_posters", "engagement_targets", "peers"]:
            for person in raw.get(source_key, []):
                scored = score_network_target(person, filters.get("target_companies", []))
                ev = attach_freshness(
                    "network",
                    evidence(
                        source_type=source_key,
                        source_url=person.get("url") or URLS.get(person.get("name", "")),
                        actor=None,
                        raw_path=raw_path,
                        scraped_at=iso_now(),
                        confidence=0.78 if source_key == "job_posters" else 0.7,
                        verified_fields=["name", "company", "role", "reason"],
                        unknown_fields=[field for field in ["url", "mutual_connections"] if not person.get(field)],
                    ),
                )
                people.append(
                    {
                        "id": scored["target_id"],
                        "name": person.get("name"),
                        "role": person.get("role") or person.get("title"),
                        "company": person.get("company"),
                        "url": person.get("url") or URLS.get(person.get("name", "")),
                        "fit_score": scored["fit_score"],
                        "reasons": scored["reasons"],
                        "why_this_person": person.get("reason") or person.get("notes") or "Relevant to your GCC PM target market.",
                        "connection_path": self.connection_path(person),
                        "connection_note": self.connection_note(person),
                        "evidence": ev,
                    }
                )
        seen = {}
        for person in people:
            key = stable_id(person.get("name"), person.get("company"))
            if key not in seen or person["fit_score"] > seen[key]["fit_score"]:
                seen[key] = person
        ranked = sorted(seen.values(), key=lambda item: item["fit_score"], reverse=True)
        return {
            "skill": "network",
            "generated_at": iso_now(),
            "status": "success" if ranked else "partial",
            "freshness_status": "fresh",
            "coverage": min(1.0, len(ranked) / 10),
            "summary": f"{len(ranked)} prioritized connection targets from peers, engagement targets, and job posters.",
            "items": ranked[:20],
            "evidence": [],
            "warnings": [] if ranked else ["no_network_targets"],
            "errors": [],
        }

    def connection_path(self, person: Dict[str, Any]) -> str:
        if person.get("reason", "").lower().startswith("posted relevant role"):
            return "Comment or connect referencing the specific open role. Do not imply application status."
        if person.get("has_creator_mode"):
            return "Engage on 2-3 recent posts first, then connect with a specific note."
        return "Use profile context and shared GCC/product domain before sending a request."

    def connection_note(self, person: Dict[str, Any]) -> str:
        name = person.get("name") or "[Name]"
        company = person.get("company") or "[Company]"
        fallback = (
            f"Hi {name}, I am building marketplace and commerce products in Dubai and saw your work around [INSERT SPECIFIC DETAIL] at {company}. "
            "Would value connecting and exchanging notes on GCC product problems."
        )
        return self.evidence_bound_draft(
            skill="network",
            task="Draft one LinkedIn connection note from verified person evidence. Keep one placeholder for the specific profile detail Akshay must confirm.",
            evidence_payload={
                "name": person.get("name"),
                "role": person.get("role") or person.get("title"),
                "company": person.get("company"),
                "reason": person.get("reason") or person.get("notes"),
            },
            fallback=fallback,
        )

    def cached_network_processed(self) -> Dict[str, Any]:
        raw = self.network_universe()
        return self.build_network_processed(raw, data_path("network/strategy.json"))

    def run_content(self, ctx: RunContext) -> Dict[str, Any]:
        actor = self.apify.actor_id("profile_posts")
        all_urls = _anchor_urls()
        if not all_urls:
            all_urls = [URLS[name] for name in ["Sanket Purohit", "Diego Granados", "Sajjad Ahmad"] if name in URLS]
        # Optimize: scrape max 4 anchors per run, 3 posts each = ~12 posts (~$0.024)
        urls = all_urls[:4]
        source = "cache"
        try:
            ctx.log(f"apify_content_posts actor={actor} targets={len(urls)} (of {len(all_urls)} total)")
            posts = self.apify.call(actor, {"targetUrls": urls, "maxPosts": 3}, wait=180)
            source = "apify"
        except Exception as exc:
            ctx.warnings.append("content_scrape_failed_using_cache")
            ctx.log(f"content_scrape_failed:{exc}")
            cached = read_json(data_path("content/inspiration/latest.json"), {})
            posts = cached.get("posts", []) if isinstance(cached, dict) else []
        raw_path = ctx.save_raw({"source": source, "posts": posts, "profile_urls": urls})
        if source == "apify":
            write_json(data_path("content/inspiration/latest.json"), {"date": iso_now(), "posts": posts, "sources": urls})
        return self.build_content_processed(posts, raw_path, actor, source)

    def build_content_processed(self, posts: List[Dict[str, Any]], raw_path: Path, actor: str, source: str) -> Dict[str, Any]:
        ranked = rank_posts(posts)
        items = []
        for post in ranked[:12]:
            content = post.get("content") or ""
            ev = attach_freshness(
                "content",
                evidence(
                    source_type=source,
                    source_url=post.get("linkedinUrl") or post.get("shareLinkedinUrl"),
                    actor=actor if source == "apify" else None,
                    raw_path=raw_path,
                    scraped_at=post.get("postedAt", {}).get("timestamp") or iso_now(),
                    confidence=0.86 if source == "apify" else 0.62,
                    verified_fields=["author", "content", "engagement", "postedAt"],
                    unknown_fields=[],
                ),
            )
            items.append(
                {
                    "id": stable_id(post.get("shareUrn"), content[:60]),
                    "author": (post.get("author") or {}).get("name"),
                    "url": post.get("linkedinUrl") or post.get("shareLinkedinUrl"),
                    "summary": compact_text(content, 260),
                    "engagement_score": post.get("_engagement_score", 0),
                    "relevance_reasons": post.get("_relevance_reasons", []),
                    "draft": self.content_draft_from_post(post),
                    "evidence": ev,
                }
            )
        warnings = []
        if len(items) < 3:
            warnings.append("insufficient_relevant_content_inspiration")
        if source == "cache":
            warnings.append("using_cached_content")
        return {
            "skill": "content",
            "generated_at": iso_now(),
            "status": "success" if items else "partial",
            "freshness_status": freshness_status("content", items[0]["evidence"]["scraped_at"] if items else None),
            "coverage": min(1.0, len(items) / 8),
            "summary": f"{len(items)} relevant inspiration posts passed topic filters. Off-topic media/post-production content is excluded.",
            "items": items,
            "calendar": self.content_calendar(),
            "warnings": warnings,
            "errors": [],
            "evidence": [],
        }

    def content_draft_from_post(self, post: Dict[str, Any]) -> str:
        """Draft a LinkedIn post inspired by an existing post using the Content Lab pipeline.

        Falls back to deterministic template if ContentLab fails entirely.
        """
        reasons = post.get("_relevance_reasons") or relevance_reasons(post.get("content", ""))
        topic = reasons[0] if reasons else "marketplace"
        fallback = (
            f"Hook: I keep seeing PMs talk about {topic}, but the hard part is not the framework.\n\n"
            "The hard part is [INSERT YOUR REAL EXAMPLE HERE].\n\n"
            "At Mumzworld/Myntra/ABFRL, the pattern I saw was: [INSERT SPECIFIC NUMBER/METRIC].\n\n"
            "My takeaway: [INSERT YOUR REAL TAKEAWAY].\n\n"
            "What has worked for you when this shows up in real product work?"
        )
        # Determine pillar from relevance reasons
        pillar = self._guess_pillar(reasons)
        try:
            result = self.content_lab.draft_from_inspiration(
                topic=topic,
                inspiration_posts=[post],
                pillar=pillar,
            )
            body = result.get("body", "")
            validation = result.get("validation", {})
            if body and validation.get("ok", False):
                return body
            # If validation failed but we have a body, still return it
            # (warnings are acceptable, only hard errors cause fallback)
            if body and not validation.get("errors"):
                return body
        except Exception:
            pass
        # Full fallback: use the old single-shot evidence_bound_draft
        return self.evidence_bound_draft(
            skill="content",
            task="Draft a LinkedIn post starter inspired by this verified post. It must not be finished; require Akshay to add his real example and metric.",
            evidence_payload={
                "author": (post.get("author") or {}).get("name"),
                "post_summary": compact_text(post.get("content"), 1200),
                "relevance_reasons": reasons,
                "engagement": post.get("engagement") or post.get("stats"),
            },
            fallback=fallback,
        )

    def _guess_pillar(self, reasons: List[str]) -> str:
        """Map relevance keywords to a content pillar id."""
        pillar_map = {
            "ecommerce": "marketplace-ops", "e-commerce": "marketplace-ops",
            "marketplace": "marketplace-ops", "checkout": "marketplace-ops",
            "cart": "marketplace-ops", "fulfillment": "marketplace-ops",
            "logistics": "marketplace-ops", "delivery": "marketplace-ops",
            "product": "pm-craft", "pm": "pm-craft", "startup": "pm-craft",
            "experiments": "pm-craft", "data": "pm-craft",
            "ai": "ai-product", "llm": "ai-product",
            "dubai": "gcc-tech", "gcc": "gcc-tech", "mena": "gcc-tech",
        }
        for reason in reasons:
            pillar = pillar_map.get(reason.lower())
            if pillar:
                return pillar
        return "pm-craft"

    def content_calendar(self) -> List[Dict[str, str]]:
        return [
            {"day": "Monday", "type": "Educational", "pillar": "Marketplace & Operations", "topic": "Checkout or fulfillment lesson with a real metric"},
            {"day": "Wednesday", "type": "Engagement", "pillar": "PM Craft", "topic": "Opinion on prioritization or execution in GCC startups"},
            {"day": "Friday", "type": "Personal", "pillar": "GCC Tech Scene", "topic": "India to Dubai PM learning with a concrete story"},
        ]

    def cached_content_processed(self) -> Dict[str, Any]:
        cached = read_json(data_path("content/inspiration/latest.json"), {})
        posts = cached.get("posts", []) if isinstance(cached, dict) else []
        return self.build_content_processed(posts, data_path("content/inspiration/latest.json"), "cached_content", "cache")

    def run_engage(self, ctx: RunContext) -> Dict[str, Any]:
        actor = self.apify.actor_id("profile_posts")
        targets = cfg("targets.json", {}).get("engagement_targets", {}).get("profiles", [])
        urls = _engagement_urls()
        if not urls:
            urls = [URLS[target["name"]] for target in targets if target.get("name") in URLS]
        source = "cache"
        try:
            # Optimize: max 4 targets, 3 posts each = ~12 posts (~$0.024)
            ctx.log(f"apify_engage_posts actor={actor} targets={len(urls[:4])}")
            posts = self.apify.call(actor, {"targetUrls": urls[:4], "maxPosts": 3}, wait=180)
            source = "apify"
        except Exception as exc:
            ctx.warnings.append("engage_scrape_failed_using_cache")
            ctx.log(f"engage_scrape_failed:{exc}")
            cached = read_json(data_path("engagement/targets-today.json"), {})
            posts = cached.get("posts", []) if isinstance(cached, dict) else []
        raw_path = ctx.save_raw({"source": source, "posts": posts, "profile_urls": urls})
        if source == "apify":
            write_json(data_path("engagement/targets-today.json"), {"date": iso_now(), "posts": posts, "targets_used": [t.get("name") for t in targets]})
        return self.build_engage_processed(posts, raw_path, actor, source)

    def build_engage_processed(self, posts: List[Dict[str, Any]], raw_path: Path, actor: str, source: str) -> Dict[str, Any]:
        ranked = rank_posts(posts)
        items = []
        for post in ranked[:7]:
            author = (post.get("author") or {}).get("name") or "Unknown"
            content = post.get("content") or ""
            ev = attach_freshness(
                "engage",
                evidence(
                    source_type=source,
                    source_url=post.get("linkedinUrl") or post.get("shareLinkedinUrl"),
                    actor=actor if source == "apify" else None,
                    raw_path=raw_path,
                    scraped_at=post.get("postedAt", {}).get("timestamp") or iso_now(),
                    confidence=0.84 if source == "apify" else 0.6,
                    verified_fields=["author", "content", "engagement", "postedAt"],
                    unknown_fields=[],
                ),
            )
            items.append(
                {
                    "id": stable_id(author, content[:80]),
                    "author": author,
                    "url": post.get("linkedinUrl") or post.get("shareLinkedinUrl"),
                    "post_summary": compact_text(content, 300),
                    "relevance_reasons": post.get("_relevance_reasons", []),
                    "comment_draft": self.comment_draft(author, post),
                    "evidence": ev,
                }
            )
        warnings = []
        if len(items) < 3:
            warnings.append("fewer_than_three_fresh_relevant_posts")
        if source == "cache":
            warnings.append("using_cached_engagement_posts")
        return {
            "skill": "engage",
            "generated_at": iso_now(),
            "status": "success" if items else "partial",
            "freshness_status": freshness_status("engage", items[0]["evidence"]["scraped_at"] if items else None),
            "coverage": min(1.0, len(items) / 5),
            "summary": f"{len(items)} engagement opportunities with evidence-backed comment drafts.",
            "items": items,
            "warnings": warnings,
            "errors": [],
            "evidence": [],
        }

    def comment_draft(self, author: str, post: Dict[str, Any]) -> str:
        reasons = post.get("_relevance_reasons") or relevance_reasons(post.get("content", ""))
        topic = reasons[0] if reasons else "[INSERT TOPIC]"
        fallback = (
            f"Interesting angle on {topic}. We saw a related pattern in commerce where [INSERT YOUR SPECIFIC EXAMPLE]. "
            f"The tradeoff I keep coming back to is [INSERT YOUR TAKE]. What made you prioritize this path, {author.split()[0] if author else '[Name]'}?"
        )
        return self.evidence_bound_draft(
            skill="engage",
            task="Draft one thoughtful LinkedIn comment from this verified post. No generic praise. Keep placeholders for Akshay's real example and take.",
            evidence_payload={
                "author": author,
                "post_summary": compact_text(post.get("content"), 1000),
                "relevance_reasons": reasons,
            },
            fallback=fallback,
        )

    def cached_engage_processed(self) -> Dict[str, Any]:
        cached = read_json(data_path("engagement/targets-today.json"), {})
        posts = cached.get("posts", []) if isinstance(cached, dict) else []
        return self.build_engage_processed(posts, data_path("engagement/targets-today.json"), "cached_engage", "cache")

    def run_watch(self, ctx: RunContext) -> Dict[str, Any]:
        actor = self.apify.actor_id("profile_posts")
        # Watch = GCC peers only (not global anchors — those go to content)
        watch_people = ["Raeesa Omar", "Sajjad Ahmad", "Shradha Mohanan", "Dharani Dharan G", "Sanket Purohit", "Hussain Abbasi"]
        urls = [URLS[name] for name in watch_people if name in URLS]
        source = "cache"
        try:
            ctx.log(f"apify_watch_posts actor={actor}")
            # Optimize: max 4 people, 4 posts each = ~16 posts (~$0.032)
            posts = self.apify.call(actor, {"targetUrls": urls[:4], "maxPosts": 4}, wait=180)
            source = "apify"
        except Exception as exc:
            ctx.warnings.append("watch_scrape_failed_using_cache")
            ctx.log(f"watch_scrape_failed:{exc}")
            cached = read_json(data_path("watch/latest.json"), {})
            posts = cached.get("posts", []) if isinstance(cached, dict) else []
        raw_path = ctx.save_raw({"source": source, "posts": posts, "profile_urls": urls})
        if source == "apify":
            write_json(data_path("watch/latest.json"), {"date": iso_now(), "posts": posts, "sources": watch_people})
        return self.build_watch_processed(posts, raw_path, actor, source)

    def build_watch_processed(self, posts: List[Dict[str, Any]], raw_path: Path, actor: str, source: str) -> Dict[str, Any]:
        ranked = rank_posts(posts)
        items = []
        for post in ranked[:10]:
            content = post.get("content") or ""
            ev = attach_freshness(
                "watch",
                evidence(
                    source_type=source,
                    source_url=post.get("linkedinUrl") or post.get("shareLinkedinUrl"),
                    actor=actor if source == "apify" else None,
                    raw_path=raw_path,
                    scraped_at=post.get("postedAt", {}).get("timestamp") or iso_now(),
                    confidence=0.84 if source == "apify" else 0.58,
                    verified_fields=["author", "content", "engagement", "postedAt"],
                ),
            )
            items.append(
                {
                    "id": stable_id((post.get("author") or {}).get("name"), content[:80]),
                    "author": (post.get("author") or {}).get("name"),
                    "url": post.get("linkedinUrl") or post.get("shareLinkedinUrl"),
                    "summary": compact_text(content, 260),
                    "signals": post.get("_relevance_reasons", []),
                    "evidence": ev,
                }
            )
        warnings = []
        if len(items) < 3:
            warnings.append("insufficient_evidence_for_confident_digest")
        if source == "cache":
            warnings.append("using_cached_watch_posts")
        digest = self.watch_digest(items)
        return {
            "skill": "watch",
            "generated_at": iso_now(),
            "status": "success" if len(items) >= 3 else "partial",
            "freshness_status": freshness_status("watch", items[0]["evidence"]["scraped_at"] if items else None),
            "coverage": min(1.0, len(items) / 6),
            "summary": digest,
            "items": items,
            "warnings": warnings,
            "errors": [],
            "evidence": [],
        }

    def watch_digest(self, items: List[Dict[str, Any]]) -> str:
        if len(items) < 3:
            return "Insufficient verified, relevant posts for a confident competitor digest. Run a fresh watch sync or expand target coverage."
        signals = sorted({signal for item in items for signal in item.get("signals", [])})
        return f"Verified signals from {len(items)} relevant posts: {', '.join(signals[:8])}. Use these as content opportunities, not confirmed market trends."

    def cached_watch_processed(self) -> Dict[str, Any]:
        cached = read_json(data_path("watch/latest.json"), {})
        posts = cached.get("posts", []) if isinstance(cached, dict) else []
        return self.build_watch_processed(posts, data_path("watch/latest.json"), "cached_watch", "cache")

    def run_scorecard(self, ctx: RunContext) -> Dict[str, Any]:
        raw = {
            "profile": read_json(processed_path("profile"), self.cached_profile_processed()),
            "jobs": read_json(processed_path("jobs"), self.cached_jobs_processed()),
            "network": read_json(processed_path("network"), self.cached_network_processed()),
            "content": read_json(processed_path("content"), self.cached_content_processed()),
            "engage": read_json(processed_path("engage"), self.cached_engage_processed()),
            "watch": read_json(processed_path("watch"), self.cached_watch_processed()),
            "manual_engagement_log": read_json(data_path("engagement/log.json"), {"entries": []}),
            "manual_content_log": read_json(data_path("content/posted.json"), {"posts": []}),
        }
        raw_path = ctx.save_raw(raw)
        return self.build_scorecard_processed(raw, raw_path)

    def build_scorecard_processed(self, raw: Dict[str, Any], raw_path: Path) -> Dict[str, Any]:
        profile = raw.get("profile", {})
        score = profile.get("score", {}).get("overall_score") or 0
        entries = raw.get("manual_engagement_log", {}).get("entries", [])
        posts = raw.get("manual_content_log", {}).get("posts", [])
        stale_sections = [
            name for name in ["jobs", "engage", "content", "watch"]
            if raw.get(name, {}).get("freshness_status") in {"stale", "expired", "unknown"}
        ]
        next_action = "Turn on Creator Mode and publish one evidence-backed post this week."
        if raw.get("engage", {}).get("items"):
            next_action = "Use Engage tab to comment on three verified relevant posts, then log the actions."
        ev = attach_freshness(
            "scorecard",
            evidence(
                source_type="processed_aggregate",
                source_url=None,
                actor=None,
                raw_path=raw_path,
                scraped_at=iso_now(),
                confidence=0.82,
                verified_fields=["profile_score", "jobs_count", "network_count", "manual_logs"],
                unknown_fields=["post_impressions", "profile_views", "search_appearances"],
                notes=["manual_linkedin_analytics_not_available"],
            ),
        )
        return {
            "skill": "scorecard",
            "generated_at": iso_now(),
            "status": "success",
            "freshness_status": "fresh",
            "coverage": 0.75 if score else 0.35,
            "summary": f"Score is {score}/10. Logged comments: {len(entries)}. Logged posts: {len(posts)}. Next action: {next_action}",
            "metrics": {
                "profile_score": score,
                "target_score": 9.0,
                "comments_logged": len(entries),
                "posts_logged": len(posts),
                "jobs_ranked": len(raw.get("jobs", {}).get("items", [])),
                "network_targets": len(raw.get("network", {}).get("items", [])),
                "stale_sections": stale_sections,
            },
            "next_action": next_action,
            "items": [],
            "evidence": [ev],
            "warnings": ["manual_linkedin_analytics_missing"] + [f"stale_section:{name}" for name in stale_sections],
            "errors": [],
        }

    def cached_scorecard_processed(self) -> Dict[str, Any]:
        raw = {
            "profile": read_json(processed_path("profile"), self.cached_profile_processed()),
            "jobs": read_json(processed_path("jobs"), self.cached_jobs_processed()),
            "network": read_json(processed_path("network"), self.cached_network_processed()),
            "content": read_json(processed_path("content"), self.cached_content_processed()),
            "engage": read_json(processed_path("engage"), self.cached_engage_processed()),
            "watch": read_json(processed_path("watch"), self.cached_watch_processed()),
            "manual_engagement_log": read_json(data_path("engagement/log.json"), {"entries": []}),
            "manual_content_log": read_json(data_path("content/posted.json"), {"posts": []}),
        }
        return self.build_scorecard_processed(raw, data_path("processed/scorecard-bootstrap.json"))

    # ------------------------------------------------------------------
    # Topics (Topic Radar — no Apify, uses DuckDuckGo HTML search)
    # ------------------------------------------------------------------

    def run_topics(self, ctx: RunContext) -> Dict[str, Any]:
        ctx.log("topic_radar_scan_start")
        radar = TopicRadar(llm_client=self.llm)
        result = radar.scan()
        topics = result.get("topics", [])
        ctx.log(f"topic_radar_scan_done topics={len(topics)} posts_analyzed={result.get('total_posts_analyzed', 0)}")
        ctx.save_raw(result)
        return {
            "skill": "topics",
            "generated_at": iso_now(),
            "status": "success" if topics else "empty",
            "freshness_status": "fresh",
            "coverage": min(1.0, len(topics) / 8),
            "summary": f"{len(topics)} trending topics extracted from {result.get('total_posts_analyzed', 0)} scraped posts across {len(result.get('pillars_covered', []))} pillars.",
            "items": topics,
            "evidence": [],
            "warnings": result.get("errors", []) if not topics else [],
            "errors": [],
        }

    def cached_topics_processed(self) -> Dict[str, Any]:
        cached = read_json(processed_path("topics"), None)
        if cached:
            return cached
        return {
            "skill": "topics",
            "generated_at": iso_now(),
            "status": "empty",
            "freshness_status": "unknown",
            "coverage": 0,
            "summary": "No topic scan has been run yet. Trigger a topics sync to discover trending content.",
            "queries_used": [],
            "total_raw_results": 0,
            "total_after_filter": 0,
            "theme_distribution": {},
            "items": [],
            "evidence": [],
            "warnings": ["not_run_yet"],
            "errors": [],
        }

    def log_manual_action(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        action_type = payload.get("type")
        if action_type not in {"comment", "post", "connection", "job"}:
            raise ValueError("Unsupported manual action type")
        if action_type == "post":
            path = data_path("content/posted.json")
            data = read_json(path, {"posts": []})
            data.setdefault("posts", []).append({"logged_at": iso_now(), **payload})
        else:
            path = data_path("engagement/log.json")
            data = read_json(path, {"entries": []})
            data.setdefault("entries", []).append({"logged_at": iso_now(), **payload})
        write_json(path, data)
        if processed_path("scorecard").exists():
            processed_path("scorecard").unlink()
        return {"ok": True, "path": str(path.relative_to(ROOT))}
