import unittest

from linkedin_orchestrator import SkillGateway
from linkedin_orchestrator.common import ROOT, data_path, read_json
from linkedin_orchestrator.scoring import parse_benchmark_report, score_profile
from linkedin_orchestrator.validators import validate_draft


class LinkedInOrchestratorTests(unittest.TestCase):
    def test_benchmark_report_is_canonical_profile_baseline(self):
        baseline = parse_benchmark_report()
        self.assertEqual(baseline["overall_score"], 7.5)
        self.assertEqual(baseline["calculated_equal_average"], 7.5)
        self.assertEqual(baseline["dimensions"]["content_presence"], 3.0)

    def test_missing_scraped_fields_are_unknown_not_negative(self):
        baseline = parse_benchmark_report()
        profile = read_json(data_path("profile/akshay-latest.json"), {})
        scored = score_profile(profile, baseline)
        self.assertEqual(scored["overall_score"], 7.5)
        # If skills are present from scrape, they won't be unknown
        self.assertIn("verified", scored["unknown_fields"])
        self.assertEqual(scored["dimensions"]["premium"], 8.0)
        # Skills score: if list has 50 items → 5.0 + min(2.5, 50/16) = 7.5
        # If empty → baseline 7.5. Either way, ≥ 7.0
        self.assertGreaterEqual(scored["dimensions"]["skills"], 7.0)

    def test_drafts_need_placeholders_and_avoid_generic_praise(self):
        bad = validate_draft("Great post. This is a game-changer.", require_placeholder=True)
        self.assertFalse(bad["ok"])
        self.assertTrue(any("banned_phrases" in error for error in bad["errors"]))
        self.assertIn("missing_required_placeholder", bad["errors"])

        good = validate_draft(
            "Interesting angle. We saw this in commerce when [INSERT YOUR SPECIFIC EXAMPLE]. "
            "The metric I would check is [INSERT SPECIFIC NUMBER/METRIC].",
            require_placeholder=True,
        )
        self.assertTrue(good["ok"])

    def test_jobs_are_processed_cards_not_links_only(self):
        gateway = SkillGateway(ROOT)
        jobs = gateway.cached_jobs_processed()
        self.assertGreater(len(jobs["items"]), 0)
        first = jobs["items"][0]
        self.assertIn("fit_score", first)
        self.assertIn("reasons", first)
        self.assertIn("outreach_draft", first)
        self.assertIn("[INSERT", first["outreach_draft"])
        self.assertIn("evidence", first)

    def test_watch_cached_has_evidence_and_validation(self):
        gateway = SkillGateway(ROOT)
        watch = gateway.cached_watch_processed()
        # Watch must have items with evidence, or insufficient_evidence warning
        if watch["coverage"] < 0.5:
            self.assertTrue(
                "insufficient_evidence_for_confident_digest" in watch["warnings"]
                or "using_cached_watch_posts" in watch["warnings"]
            )
        for item in watch.get("items", []):
            self.assertIn("evidence", item)
            self.assertIn("source_type", item["evidence"])

    def test_skill_run_creates_artifacts(self):
        gateway = SkillGateway(ROOT)
        result = gateway.run("scorecard")
        run = result["run"]
        self.assertIn("run_id", run)
        self.assertEqual(run["skill"], "scorecard")
        self.assertTrue((ROOT / run["raw_path"]).exists())
        self.assertTrue((ROOT / run["processed_path"]).exists())
        self.assertTrue((ROOT / run["validation_path"]).exists())

    def test_llm_draft_falls_back_when_validation_fails(self):
        class FakeLLM:
            available = True

            def call(self, prompt, max_tokens=400):
                return "Great post. This is a game-changer."

        gateway = SkillGateway(ROOT)
        gateway.llm = FakeLLM()
        fallback = "Interesting angle. Akshay should add [INSERT YOUR REAL EXAMPLE]."
        draft = gateway.evidence_bound_draft(
            skill="engage",
            task="Draft a comment.",
            evidence_payload={"post_summary": "Verified PM post text."},
            fallback=fallback,
        )
        self.assertEqual(draft, fallback)


if __name__ == "__main__":
    unittest.main()
