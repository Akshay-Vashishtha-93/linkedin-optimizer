# LinkedIn Optimizer

AI-powered LinkedIn growth system built by a non-developer Product Manager using Claude Code + Apify.

9 integrated skills that scrape LinkedIn data, benchmark your profile, generate content drafts, discover networking targets, and track your progress — all running locally with evidence-backed outputs.

## What It Does

| Skill | What It Does |
|-------|-------------|
| **Profile** | Scores your profile on 10 dimensions against peer benchmarks |
| **Content** | Scrapes top PM/AI creators, generates 4-stage post drafts (Research → Hook → Body → Anti-slop) |
| **Topics** | Extracts trending topics from scraped posts, clusters by your content pillars |
| **Jobs** | Ranks PM jobs in your target region with fit scores, outreach drafts, referral targets |
| **Network** | Prioritizes connection targets by strategic value, generates personalized notes |
| **Discover** | Finds new people to follow from active post authors, enriches via Apify |
| **Engage** | Queues fresh posts with evidence-backed comment drafts |
| **Watch** | Competitor/peer signal digest from recent posts |
| **Scorecard** | Weekly operating view — score, stale sections, next action |

## Architecture

```
Dashboard (HTML) → POST /api/skills/{skill}/run → server.py
                                                      ↓
                                              SkillGateway
                                       ┌──────────┼──────────┐
                                   ApifyClient  LLMClient  ContentLab
                                       ↓           ↓          ↓
                                  LinkedIn      Claude     4-stage
                                  scraping      Haiku      pipeline
                                       ↓           ↓          ↓
                                   Scoring → Validators → Evidence Store
                                       ↓
                                  data/processed/{skill}.json
                                       ↓
                              Dashboard reads & renders
```

**Key design decisions:**
- **Evidence-backed everything** — every item has source, confidence, freshness, verified/unknown fields
- **LLM can only draft from evidence** — Claude generates text but cannot invent facts. Bad drafts are rejected and replaced with deterministic fallbacks
- **Anti-slop validation** — banned phrases ("game-changer", "leverage", "delighted to share"), placeholder enforcement, char limits
- **Budget guard** — live Apify usage from API, pre-flight exhaustion check, cost estimation per run

## Tech Stack

- **Python 3** (stdlib only — no pip dependencies except `anthropic`)
- **Apify** for LinkedIn scraping (profile, posts, jobs actors)
- **Claude Haiku** for evidence-bound drafting
- **HTML/JS** dashboard (no framework, vanilla)

## Setup

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/linkedin-optimizer.git
cd linkedin-optimizer

# 2. Create .env (see .env.example)
cp .env.example .env
# Add your APIFY_TOKEN and ANTHROPIC_API_KEY

# 3. Create targets config
cp config/targets.example.json config/targets.json
# Edit with your LinkedIn URL, peers, and engagement targets

# 4. Install anthropic SDK
pip install anthropic

# 5. Run
./start.sh
# Opens http://localhost:8080/dashboard.html
```

## Content Lab (4-Stage Draft Pipeline)

The content system doesn't just ask an LLM to "write a post." It runs a structured pipeline:

1. **Research** — Extracts key insight from inspiration posts, identifies your angle
2. **Hook Writing** — Generates 3 options (question, bold statement, story) — max 100 chars each
3. **Body Draft** — Full post with mandatory `[INSERT YOUR REAL EXAMPLE]` placeholders
4. **Anti-slop Review** — Validates against banned phrases, char limits, hashtag count. Auto-fixes and retries once.

If any stage fails, it falls back to deterministic output. The LLM is never trusted blindly.

## Cost

Optimized for Apify Starter ($29/mo):

| Skill | Posts/Profiles | Estimated Cost |
|-------|---------------|----------------|
| Content | 4 anchors × 3 posts | ~$0.024 |
| Engage | 4 targets × 3 posts | ~$0.024 |
| Profile | 1 profile | ~$0.004 |
| Jobs | 25 results | ~$0.025 |
| Topics | Free (uses cached posts) | $0 |
| Discover | Free (uses cached posts) | $0 |

Monthly estimate (3x content/week + 1x jobs/week + 1x profile/month): **~$3-4/month**

## Project Structure

```
linkedin-optimizer/
├── server.py                    # Local HTTP server, loads .env, routes to SkillGateway
├── start.sh                     # Launch script
├── dashboard.html               # 11-tab visual dashboard
├── config/
│   ├── settings.json            # Budget, actors, scoring weights
│   ├── content-pillars.json     # 4 content pillars + anti-slop rules
│   ├── job-filters.json         # PM job search filters
│   └── targets.example.json     # Template for your LinkedIn targets
├── linkedin_orchestrator/
│   ├── skills.py                # SkillGateway — 9 handlers, evidence-bound drafts
│   ├── apify_client.py          # Live budget sync, pre-flight checks, actor calls
│   ├── scoring.py               # Deterministic profile (10 dims), job (0-100), network scoring
│   ├── validators.py            # Anti-slop, evidence validation, relevance filtering
│   ├── content_lab.py           # 4-stage content pipeline
│   ├── topic_radar.py           # Topic extraction + pillar clustering
│   ├── discovery.py             # Rotating anchor discovery from post authors
│   └── common.py                # Evidence model, freshness, utilities
└── tests/
    └── test_linkedin_orchestrator.py  # 7 tests (scoring, validation, artifacts)
```

## Built With

This entire system was built by a Product Manager with no prior coding experience, using [Claude Code](https://claude.ai/claude-code) as the development partner. The orchestrator, dashboard, scoring engine, content pipeline, and all 9 skills were created through AI-assisted development — proof that PMs can build production tools, not just spec them.

## License

MIT
