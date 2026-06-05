from __future__ import annotations

import html
import re
from typing import Any, Dict, Iterable, List, Tuple

from .common import REPORTS_DIR, compact_text, stable_id
from .validators import is_relevant_text, relevance_reasons


DIMENSION_KEYS = [
    "headline",
    "about",
    "experience",
    "metrics_outcomes",
    "network_size",
    "content_presence",
    "premium",
    "skills",
    "education",
    "industry_relevance",
]

DIMENSION_LABELS = {
    "headline": "Headline & Positioning",
    "about": "About / Summary",
    "experience": "Experience Detail",
    "metrics_outcomes": "Metrics & Outcomes",
    "network_size": "Network Size",
    "content_presence": "Content / Creator Presence",
    "premium": "Premium / Verification",
    "skills": "Skills & Endorsements",
    "education": "Education & Certifications",
    "industry_relevance": "Industry Relevance",
}

REPORT_NAME_TO_KEY = {
    "headline & positioning": "headline",
    "about / summary": "about",
    "experience detail": "experience",
    "metrics & outcomes": "metrics_outcomes",
    "network size": "network_size",
    "content / creator presence": "content_presence",
    "premium / verification": "premium",
    "skills & endorsements": "skills",
    "education & certifications": "education",
    "industry relevance": "industry_relevance",
}


def canonical_overall(dimensions: Dict[str, Any]) -> float:
    values = [float(dimensions[key]) for key in DIMENSION_KEYS if isinstance(dimensions.get(key), (int, float))]
    if not values:
        return 0.0
    return round(sum(values) / len(values), 1)


def parse_benchmark_report(path=None) -> Dict[str, Any]:
    report_path = path or REPORTS_DIR / "benchmark-2026-05-15.html"
    text = report_path.read_text()
    score_match = re.search(r'class="score-bump">([\d.]+)\s*/\s*10', text)
    dimensions: Dict[str, float] = {}
    row_re = re.compile(
        r'<div class="param-name">\s*\d+\.\s*([^<]+)</div>.*?'
        r'<span class="bar-score[^"]*">([\d.]+)</span>',
        re.S,
    )
    for raw_name, raw_score in row_re.findall(text):
        name = html.unescape(raw_name).strip().lower()
        key = REPORT_NAME_TO_KEY.get(name)
        if key:
            dimensions[key] = float(raw_score)
    return {
        "source": str(report_path),
        "overall_score": float(score_match.group(1)) if score_match else canonical_overall(dimensions),
        "dimensions": dimensions,
        "calculated_equal_average": canonical_overall(dimensions),
    }


def metric_count(text: str) -> int:
    return len(re.findall(r"\d+[%xX]|\d+\s?[Cc]r|₹\s?\d+|\$\s?\d+|\d+\s?[Kk]\b|\d+\s?days?", text or ""))


def score_headline(headline: str, baseline: float) -> Tuple[float, List[str]]:
    lowered = (headline or "").lower()
    keywords = [
        "product",
        "manager",
        "ai",
        "marketplace",
        "superapp",
        "super-app",
        "mobile",
        "fulfillment",
        "hyperlocal",
        "dubai",
        "e-commerce",
        "ecommerce",
    ]
    hits = [kw for kw in keywords if kw in lowered]
    if not headline:
        return baseline, ["headline_missing_using_baseline"]
    score = 6.5 + min(2.0, len(hits) * 0.25)
    if "|" in headline:
        score += 0.5
    if "senior product manager" in lowered:
        score += 0.5
    # The May 15 report is the canonical baseline. Do not award a silent
    # improvement unless a future scored report or explicit rule supports it.
    return min(baseline, min(9.0, round(score * 2) / 2)), [f"keyword_hits:{','.join(hits[:8])}"]


def score_about(about: str, baseline: float) -> Tuple[float, List[str]]:
    if not about:
        return baseline, ["about_missing_using_baseline"]
    metrics = metric_count(about)
    score = 6.5
    if len(about) > 700:
        score += 0.75
    if "selected outcomes" in about.lower():
        score += 0.5
    score += min(1.75, metrics * 0.35)
    return min(baseline, min(9.0, round(score * 2) / 2)), [f"metric_count:{metrics}"]


def score_network(connections: int, followers: int) -> Tuple[float, List[str]]:
    conn = connections or 0
    foll = followers or 0
    ratio = foll / conn if conn else 0
    score = 4.0
    if conn >= 1000:
        score += 1
    if conn >= 3000:
        score += 1
    if conn >= 5000:
        score += 1
    if conn >= 10000:
        score += 1
    if ratio >= 1.2:
        score += 0.5
    if ratio >= 1.5:
        score += 0.5
    return min(9.0, round(score * 2) / 2), [f"followers_to_connections:{ratio:.2f}"]


def score_content(profile: Dict[str, Any]) -> Tuple[float, List[str]]:
    creator = profile.get("creator")
    featured = profile.get("featured")
    if creator is True and featured:
        return 6.0, ["creator_mode_on", "featured_present"]
    if creator is True:
        return 5.0, ["creator_mode_on"]
    return 3.0, ["creator_mode_off", "featured_missing_or_unknown"]


def score_premium(profile: Dict[str, Any], baseline: float) -> Tuple[float, List[str], List[str]]:
    premium = profile.get("premium")
    verified = profile.get("verified")
    unknown: List[str] = []
    if premium is None:
        unknown.append("premium")
    if verified is None:
        unknown.append("verified")
    if premium is True and verified is True:
        return 8.0, ["premium_true", "verified_true"], unknown
    if premium is True and verified is None:
        return max(8.0, baseline), ["premium_true", "verified_unknown_using_report_context"], unknown
    if premium is True:
        return 6.0, ["premium_true", "verified_false_or_missing"], unknown
    if premium is False:
        return 5.0, ["premium_false"], unknown
    return baseline, ["premium_unknown_using_baseline"], unknown


def score_skills(skills: Any, baseline: float) -> Tuple[float, List[str], List[str]]:
    if isinstance(skills, list) and len(skills) > 0:
        count = len(skills)
        score = 5.0 + min(2.5, count / 16)
        return round(score * 2) / 2, [f"skill_count:{count}"], []
    return baseline, ["skills_missing_from_scrape_using_baseline"], ["skills"]


def score_profile(profile: Dict[str, Any], baseline: Dict[str, Any] | None = None) -> Dict[str, Any]:
    baseline = baseline or parse_benchmark_report()
    base_dims = {key: float(value) for key, value in baseline.get("dimensions", {}).items()}
    dimensions = {key: base_dims.get(key) for key in DIMENSION_KEYS}
    reasons: Dict[str, List[str]] = {}
    unknown_fields: List[str] = []

    dimensions["headline"], reasons["headline"] = score_headline(profile.get("headline", ""), base_dims.get("headline", 7.0))
    dimensions["about"], reasons["about"] = score_about(profile.get("about", ""), base_dims.get("about", 7.0))

    experience = profile.get("experience")
    if isinstance(experience, list) and experience:
        described = sum(1 for item in experience if item.get("description"))
        dimensions["experience"] = base_dims.get("experience", 7.0)
        reasons["experience"] = [f"experience_entries:{len(experience)}", f"described_roles:{described}", "qualitative_report_baseline_used"]
    else:
        unknown_fields.append("experience")
        reasons["experience"] = ["experience_missing_using_baseline"]

    about_plus_exp = " ".join(
        [profile.get("about", "")]
        + [item.get("description", "") for item in (experience if isinstance(experience, list) else [])]
    )
    metrics = metric_count(about_plus_exp)
    dimensions["metrics_outcomes"] = base_dims.get("metrics_outcomes", 9.0) if metrics >= 4 else max(6.0, 5 + metrics * 0.75)
    reasons["metrics_outcomes"] = [f"metric_count:{metrics}"]

    dimensions["network_size"], reasons["network_size"] = score_network(
        int(profile.get("connectionsCount") or 0),
        int(profile.get("followerCount") or 0),
    )
    dimensions["content_presence"], reasons["content_presence"] = score_content(profile)
    premium_score, premium_reasons, premium_unknown = score_premium(profile, base_dims.get("premium", 8.0))
    dimensions["premium"] = premium_score
    reasons["premium"] = premium_reasons
    unknown_fields.extend(premium_unknown)

    skill_score, skill_reasons, skill_unknown = score_skills(profile.get("skills"), base_dims.get("skills", 7.5))
    dimensions["skills"] = skill_score
    reasons["skills"] = skill_reasons
    unknown_fields.extend(skill_unknown)

    dimensions["education"] = base_dims.get("education", 6.5)
    dimensions["industry_relevance"] = base_dims.get("industry_relevance", 9.5)
    reasons["education"] = ["qualitative_report_baseline_used"]
    reasons["industry_relevance"] = ["qualitative_report_baseline_used"]

    clean_dims = {key: float(dimensions[key]) for key in DIMENSION_KEYS if dimensions.get(key) is not None}
    return {
        "overall_score": canonical_overall(clean_dims),
        "dimensions": clean_dims,
        "dimension_labels": DIMENSION_LABELS,
        "reasons": reasons,
        "unknown_fields": sorted(set(unknown_fields)),
        "method": "equal_average_canonical_report_baseline",
        "baseline_report_score": baseline.get("overall_score"),
    }


TARGET_COMPANIES_PRIORITY = {
    "talabat": 20,
    "noon": 20,
    "careem": 20,
    "tabby": 20,
    "property finder": 18,
    "amazon": 16,
    "majid al futtaim": 16,
    "maf": 16,
    "tamara": 15,
    "deliveroo": 15,
    "kitopi": 14,
}

STRONG_DOMAIN = [
    "marketplace",
    "e-commerce",
    "ecommerce",
    "super app",
    "superapp",
    "fulfillment",
    "last mile",
    "last-mile",
    "checkout",
    "payments",
    "qcommerce",
]

GOOD_DOMAIN = [
    "mobile",
    "growth",
    "platform",
    "consumer",
    "logistics",
    "delivery",
    "cart",
    "order",
    "seller",
    "merchant",
    "catalog",
    "search",
    "discovery",
    "ai",
    "automation",
]

HARD_EXCLUDE = [
    "intern",
    "junior",
    "associate product",
    "entry level",
    "marketing manager",
    "sales manager",
    "brand manager",
    "product design",
    "product development",
    "engineer",
    "data analyst",
    "business analyst",
    "customer service",
    "store manager",
    "warehouse",
    "medical device",
    "pharmaceutical",
    "clinical",
    "uae national only",
    "emirati only",
]


def canonical_job_key(job: Dict[str, Any]) -> str:
    return stable_id(job.get("id"), job.get("link") or job.get("applyUrl"), job.get("companyName"), job.get("title"))


def dedupe_jobs(jobs: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for job in jobs:
        key = canonical_job_key(job)
        if key in seen:
            continue
        seen.add(key)
        out.append(job)
    return out


def score_job(job: Dict[str, Any], target_companies: Iterable[str]) -> Dict[str, Any]:
    title = (job.get("title") or "").lower()
    std_title = (job.get("standardizedTitle") or "").lower()
    desc = (job.get("descriptionText") or "").lower()
    company = (job.get("companyName") or "").lower()
    full_text = f"{title} {std_title} {desc[:2500]} {company}"
    reasons: List[str] = []
    risk_flags: List[str] = []
    score = 0

    if any(term in full_text for term in HARD_EXCLUDE):
        risk_flags.append("hard_exclusion_match")

    senior_terms = ["senior product manager", "sr product manager", "lead product", "principal product", "head of product", "director of product", "group product", "product lead"]
    ok_terms = ["product manager", "product owner"]
    if any(term in title or term in std_title for term in senior_terms):
        score += 25
        reasons.append("senior_or_lead_level")
    elif any(term in title or term in std_title for term in ok_terms):
        score += 15
        reasons.append("pm_level")
    else:
        risk_flags.append("weak_title_match")

    target_set = [str(item).lower() for item in target_companies]
    target_match = next((tc for tc in target_set if tc and tc.split(" / ")[0] in company), None)
    if target_match:
        score += 20
        reasons.append("target_company")
    else:
        for company_key, points in TARGET_COMPANIES_PRIORITY.items():
            if company_key in company:
                score += points
                reasons.append("priority_company")
                break

    strong_hits = [term for term in STRONG_DOMAIN if term in full_text]
    good_hits = [term for term in GOOD_DOMAIN if term in full_text]
    if strong_hits:
        score += min(24, len(strong_hits) * 6)
        reasons.append("core_domain:" + ",".join(strong_hits[:4]))
    if good_hits:
        score += min(12, len(good_hits) * 2)
        reasons.append("adjacent_domain:" + ",".join(good_hits[:4]))

    loc = (job.get("location") or "").lower()
    if "dubai" in loc:
        score += 8
        reasons.append("dubai")
    elif "abu dhabi" in loc or "riyadh" in loc or "united arab emirates" in loc:
        score += 5
        reasons.append("target_region")
    else:
        risk_flags.append("location_not_primary")

    industries = (job.get("industries") or "").lower()
    if any(term in industries for term in ["technology", "software", "internet", "e-commerce", "financial", "logistics", "food", "transportation"]):
        score += 8
        reasons.append("tech_or_target_industry")

    if "easy apply" in lower_values(job):
        reasons.append("easy_apply")

    score = max(0, min(100, score))
    if risk_flags:
        score = min(score, 59)
    fit = "Strong fit" if score >= 70 else "Good fit" if score >= 50 else "Worth reviewing" if score >= 35 else "Low fit"
    return {
        "job_id": canonical_job_key(job),
        "fit_score": score,
        "fit_level": fit,
        "reasons": reasons,
        "risk_flags": risk_flags,
        "domain_matches": strong_hits + good_hits[:5],
    }


def lower_values(job: Dict[str, Any]) -> str:
    return " ".join(str(v).lower() for v in job.values() if isinstance(v, (str, int, float)))


def score_network_target(person: Dict[str, Any], target_companies: Iterable[str]) -> Dict[str, Any]:
    company = (person.get("company") or person.get("companyName") or "").lower()
    role = (person.get("role") or person.get("title") or "").lower()
    reason_text = (person.get("reason") or person.get("notes") or "").lower()
    score = 30
    reasons: List[str] = []
    if any(term in role for term in ["head", "director", "vp", "principal"]):
        score += 20
        reasons.append("senior_product_leader")
    elif "product" in role or "pm" in role:
        score += 12
        reasons.append("product_role")
    for target in target_companies:
        if str(target).lower().split(" / ")[0] in company:
            score += 18
            reasons.append("target_company")
            break
    if person.get("has_creator_mode") or person.get("creator"):
        score += 10
        reasons.append("active_creator_signal")
    if any(term in reason_text for term in ["hiring", "high engagement", "creator", "same market", "ai", "fulfillment"]):
        score += 8
        reasons.append("strategic_reason")
    followers = person.get("followers") or 0
    connections = person.get("connections") or 0
    if followers and connections and followers > connections * 1.2:
        score += 7
        reasons.append("audience_multiplier")
    return {
        "target_id": stable_id(person.get("name"), company, role),
        "fit_score": min(100, score),
        "reasons": reasons or ["config_target"],
    }


def rank_posts(posts: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    ranked = []
    for post in posts:
        content = post.get("content") or post.get("text") or ""
        if not is_relevant_text(content):
            continue
        engagement = post.get("engagement") or {}
        likes = engagement.get("likes") or 0
        comments = engagement.get("comments") or 0
        ranked.append(
            {
                **post,
                "_relevance_reasons": relevance_reasons(content),
                "_engagement_score": likes + comments * 5,
                "_summary": compact_text(content, 220),
            }
        )
    ranked.sort(key=lambda item: (item.get("_engagement_score", 0), item.get("postedAt", {}).get("timestamp", 0)), reverse=True)
    return ranked
