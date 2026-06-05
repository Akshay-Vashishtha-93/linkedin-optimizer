from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .common import cfg, normalize_actor_id, read_json, safe_env, write_json, DATA_DIR


class ApifyError(RuntimeError):
    pass


class BudgetDecision:
    def __init__(self, allowed: bool, estimated_cost: float, warnings: Optional[List[str]] = None):
        self.allowed = allowed
        self.estimated_cost = estimated_cost
        self.warnings = warnings or []

    def as_dict(self) -> Dict[str, Any]:
        return {
            "allowed": self.allowed,
            "estimated_cost": self.estimated_cost,
            "warnings": self.warnings,
        }


_USAGE_CACHE_PATH = DATA_DIR / "apify_usage_cache.json"
_USAGE_CACHE_TTL = 300  # 5 minutes


class ApifyClient:
    def __init__(self, root: Path):
        self.root = root
        self.settings = cfg("settings.json", {})
        self.token = self._load_token()

    def _load_token(self) -> Optional[str]:
        env_token = safe_env("APIFY_TOKEN")
        if env_token:
            return env_token
        auth_path = Path.home() / ".apify" / "auth.json"
        data = read_json(auth_path, {})
        token = data.get("token") if isinstance(data, dict) else None
        return token.strip() if token else None

    def actor_id(self, key_or_actor: str) -> str:
        actors = self.settings.get("apify", {}).get("actors", {})
        actor = actors.get(key_or_actor, key_or_actor)
        return normalize_actor_id(actor)

    def live_usage(self) -> Dict[str, Any]:
        """Fetch live Apify account usage. Returns cached result if <5min old."""
        cached = read_json(_USAGE_CACHE_PATH, {})
        if cached and (time.time() - cached.get("fetched_at", 0)) < _USAGE_CACHE_TTL:
            return cached

        if not self.token:
            return {"error": "no_token", "plan": "unknown", "usage_usd": 0, "limit_usd": 0}

        try:
            # Get plan info
            user_url = f"https://api.apify.com/v2/users/me?token={self.token}"
            with urllib.request.urlopen(user_url, timeout=15) as resp:
                user_data = json.loads(resp.read()).get("data", {})
            plan = user_data.get("plan", {})
            plan_name = plan.get("id", "unknown")
            plan_limit_usd = float(plan.get("maxMonthlyUsageUsd", 29))

            # Get actual monthly usage
            usage_url = f"https://api.apify.com/v2/users/me/usage/monthly?token={self.token}"
            with urllib.request.urlopen(usage_url, timeout=15) as resp:
                usage_data = json.loads(resp.read()).get("data", {})

            # Sum all service usage costs
            services = usage_data.get("monthlyServiceUsage", {})
            total_usd = sum(
                float(svc.get("amountAfterVolumeDiscountUsd", 0))
                for svc in services.values()
            )

            cycle = usage_data.get("usageCycle", {})
            result = {
                "plan": plan_name,
                "usage_usd": round(total_usd, 2),
                "limit_usd": round(plan_limit_usd, 2),
                "remaining_usd": round(max(0, plan_limit_usd - total_usd), 2),
                "pct_used": round((total_usd / plan_limit_usd * 100) if plan_limit_usd else 0, 1),
                "is_exhausted": total_usd >= plan_limit_usd if plan_limit_usd else False,
                "cycle_start": cycle.get("startAt"),
                "cycle_end": cycle.get("endAt"),
                "fetched_at": time.time(),
                "error": None,
            }
        except Exception as exc:
            budget = self.settings.get("budget", {})
            result = {
                "plan": "unknown",
                "usage_usd": 0,
                "limit_usd": float(budget.get("monthly_cap_usd", 29.0)),
                "remaining_usd": float(budget.get("monthly_cap_usd", 29.0)),
                "pct_used": 0,
                "is_exhausted": False,
                "fetched_at": time.time(),
                "error": f"api_fetch_failed:{exc}",
            }

        write_json(_USAGE_CACHE_PATH, result)
        return result

    def estimate(self, cost_keys: Iterable[str], multiplier: int = 1) -> BudgetDecision:
        usage = self.live_usage()
        cap = usage.get("limit_usd", 29.0)
        spent = usage.get("usage_usd", 0)
        alert_pct = float(self.settings.get("budget", {}).get("alert_at_pct", 80))
        costs = self.settings.get("apify", {}).get("cost_per_run", {})
        estimated = sum(float(costs.get(key, 0.0)) for key in cost_keys) * multiplier
        warnings: List[str] = []
        if not self.token:
            warnings.append("missing_apify_token")
        if usage.get("is_exhausted"):
            warnings.append("apify_budget_exhausted")
            return BudgetDecision(False, round(estimated, 4), warnings)
        if usage.get("error"):
            warnings.append(f"apify_usage_api:{usage['error']}")
        if cap and ((spent + estimated) / cap * 100) >= alert_pct:
            warnings.append("budget_alert_threshold_reached")
        return BudgetDecision(True, round(estimated, 4), warnings)

    def call(self, actor_id: str, input_data: Dict[str, Any], wait: int = 180) -> List[Dict[str, Any]]:
        if not self.token:
            raise ApifyError("APIFY_TOKEN not configured and ~/.apify/auth.json not found")

        # Pre-flight budget check
        usage = self.live_usage()
        if usage.get("is_exhausted"):
            raise ApifyError(
                f"Apify monthly budget exhausted: ${usage.get('usage_usd', '?')}"
                f" / ${usage.get('limit_usd', '?')}. Resets next billing cycle."
            )

        actor = normalize_actor_id(actor_id)
        url = f"https://api.apify.com/v2/acts/{actor}/runs?token={self.token}&waitForFinish={wait}"
        req = urllib.request.Request(
            url,
            data=json.dumps(input_data).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=wait + 30) as resp:
                run = json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            raise ApifyError(f"Apify actor {actor} failed: HTTP {exc.code}") from exc
        except Exception as exc:
            raise ApifyError(f"Apify actor {actor} failed: {exc}") from exc

        dataset_id = run.get("data", {}).get("defaultDatasetId")
        status = run.get("data", {}).get("status")
        if not dataset_id:
            raise ApifyError(f"Apify actor {actor} returned no dataset; status={status}")

        items_url = f"https://api.apify.com/v2/datasets/{dataset_id}/items?token={self.token}"
        try:
            with urllib.request.urlopen(items_url, timeout=60) as resp:
                items = json.loads(resp.read())
        except Exception as exc:
            raise ApifyError(f"Could not fetch Apify dataset {dataset_id}: {exc}") from exc

        # Invalidate usage cache after a run (costs will have changed)
        if _USAGE_CACHE_PATH.exists():
            _USAGE_CACHE_PATH.unlink()

        return items if isinstance(items, list) else []
