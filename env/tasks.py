"""
Task set generators for Easy / Medium / Hard difficulty levels.

Hard tasks include:
- Dependency chains
- Tight deadlines
- Noisy/ambiguous transcripts
- Developer skill mismatches
"""

from __future__ import annotations

from typing import List, Tuple

from .models import Developer, ExtractedItem, Difficulty


# ---------------------------------------------------------------------------
# Developer pools
# ---------------------------------------------------------------------------

def get_developers(difficulty: Difficulty) -> List[Developer]:
    if difficulty == Difficulty.EASY:
        return [
            Developer(id="D1", name="Alice", capacity=10, skill=1.0,
                      specializations=["backend", "auth"]),
            Developer(id="D2", name="Bob", capacity=10, skill=1.0,
                      specializations=["frontend", "testing"]),
            Developer(id="D3", name="Carol", capacity=10, skill=1.0,
                      specializations=["devops", "infra"]),
        ]
    elif difficulty == Difficulty.MEDIUM:
        return [
            Developer(id="D1", name="Alice", capacity=8, skill=0.9,
                      specializations=["backend", "auth", "database"]),
            Developer(id="D2", name="Bob", capacity=6, skill=0.8,
                      specializations=["frontend", "analytics"]),
            Developer(id="D3", name="Carol", capacity=7, skill=0.85,
                      specializations=["devops", "infra", "testing"]),
        ]
    else:  # HARD
        return [
            Developer(id="D1", name="Alice", capacity=5, skill=0.7,
                      specializations=["backend"]),
            Developer(id="D2", name="Bob", capacity=4, skill=0.6,
                      specializations=["frontend"]),
            Developer(id="D3", name="Carol", capacity=6, skill=0.8,
                      specializations=["devops"]),
            Developer(id="D4", name="Dave", capacity=3, skill=0.5,
                      specializations=["testing"]),  # bottleneck dev
        ]


# ---------------------------------------------------------------------------
# Transcript generators
# ---------------------------------------------------------------------------

def get_transcript(difficulty: Difficulty) -> str:
    if difficulty == Difficulty.EASY:
        return (
            "Team, clear priorities this sprint. "
            "Implement the user login page — medium priority, due by day 5. "
            "Add unit tests for the auth module — low priority, day 7. "
            "Fix the CSS bug on the homepage — quick fix, day 2. "
            "Set up GitHub Actions CI pipeline — medium priority, day 6."
        )

    elif difficulty == Difficulty.MEDIUM:
        return (
            "Alright everyone, here's our sprint plan. "
            "First, implement OAuth2 login with Google SSO — high priority, blocks the client demo on day 5. "
            "Second, the payment bug is intermittently failing transactions — P0, must fix within 2 days. "
            "Third, build the analytics dashboard with charts — medium priority, day 7. "
            "Fourth, migrate the user table to PostgreSQL — high priority, 8 story points, due day 6. "
            "Fifth, write integration tests for the auth module — low priority, day 8. "
            "Sixth, configure the CI/CD pipeline — medium priority, day 5. "
            "Also fix avatar loading bug on profile page — quick, day 3."
        )

    else:  # HARD: noisy, ambiguous, tight deadlines, dependencies
        return (
            "Ok so uh, we really need to get moving. The payment thing is broken again — "
            "yesterday it was throwing 500 errors on checkout, someone needs to jump on that today or tomorrow max. "
            "Also the entire auth system needs to be redone because apparently the token refresh is leaking memory, "
            "and the client is breathing down our necks about the SSO feature — "
            "that was supposed to be done last sprint. That's blocking the onboarding flow. "
            "On the frontend side — the dashboard is loading for like 8 seconds, "
            "charts aren't rendering on Safari, and the mobile layout is completely broken on iOS 17. "
            "The database migration from MySQL to Postgres is halfway done, someone left it mid-sprint, "
            "we need that done by day 4 otherwise the API endpoints won't work. "
            "DevOps needs to finish the Kubernetes setup — we can't deploy anything right now without it. "
            "Also QA keeps complaining there are zero tests, we need at least 80% coverage on auth. "
            "And someone needs to write the API documentation, the integration partners are waiting. "
            "Oh and the file upload feature — we promised that to the client for end of sprint, day 10."
        )


# ---------------------------------------------------------------------------
# Pre-baked extracted items (for determinism without LLM)
# ---------------------------------------------------------------------------

def get_extracted_items(difficulty: Difficulty) -> List[ExtractedItem]:
    if difficulty == Difficulty.EASY:
        return [
            ExtractedItem(task="Implement user login page", deadline=5, priority=2, tags=["frontend", "auth"]),
            ExtractedItem(task="Add unit tests for auth module", deadline=7, priority=1, tags=["testing", "auth"]),
            ExtractedItem(task="Fix CSS bug on homepage", deadline=2, priority=2, tags=["bug", "frontend"]),
            ExtractedItem(task="Setup GitHub Actions CI pipeline", deadline=6, priority=2, tags=["infra", "ci"]),
        ]

    elif difficulty == Difficulty.MEDIUM:
        return [
            ExtractedItem(task="Implement OAuth2 login with Google SSO", deadline=5, priority=3, tags=["auth", "backend", "feature"]),
            ExtractedItem(task="Fix intermittent payment processing bug", deadline=2, priority=4, tags=["bug", "payments", "backend"]),
            ExtractedItem(task="Build analytics dashboard with charts", deadline=7, priority=2, tags=["frontend", "analytics"]),
            ExtractedItem(task="Migrate user table to PostgreSQL", deadline=6, priority=3, tags=["backend", "database", "infra"]),
            ExtractedItem(task="Write integration tests for auth module", deadline=8, priority=1, tags=["testing", "auth"]),
            ExtractedItem(task="Configure CI/CD pipeline", deadline=5, priority=2, tags=["infra", "ci"]),
            ExtractedItem(task="Fix avatar loading bug on profile page", deadline=3, priority=2, tags=["bug", "frontend"]),
        ]

    else:  # HARD
        return [
            ExtractedItem(task="Fix payment 500 errors on checkout", deadline=2, priority=4, tags=["bug", "payments", "backend"]),
            ExtractedItem(task="Fix token refresh memory leak in auth system", deadline=3, priority=4, tags=["bug", "auth", "backend"]),
            ExtractedItem(task="Implement SSO feature for onboarding flow", deadline=4, priority=4, tags=["auth", "feature", "backend"]),
            ExtractedItem(task="Fix dashboard 8s load time performance", deadline=5, priority=3, tags=["frontend", "performance"]),
            ExtractedItem(task="Fix chart rendering on Safari", deadline=5, priority=3, tags=["bug", "frontend"]),
            ExtractedItem(task="Fix mobile layout on iOS 17", deadline=5, priority=3, tags=["bug", "frontend"]),
            ExtractedItem(task="Complete MySQL to PostgreSQL migration", deadline=4, priority=4, tags=["backend", "database", "infra"]),
            ExtractedItem(task="Finish Kubernetes cluster setup", deadline=4, priority=4, tags=["devops", "infra"]),
            ExtractedItem(task="Achieve 80% test coverage on auth module", deadline=6, priority=3, tags=["testing", "auth"]),
            ExtractedItem(task="Write API documentation for integration partners", deadline=8, priority=2, tags=["documentation"]),
            ExtractedItem(task="Implement file upload feature", deadline=10, priority=3, tags=["feature", "backend", "frontend"]),
        ]