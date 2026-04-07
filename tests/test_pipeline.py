from planner import generate_plan
from env.models import Difficulty


def test_generate_plan_from_transcript_returns_assignments():
    transcript = (
        "We need to fix the checkout bug today because it blocks the release. "
        "Also build the analytics dashboard by day 5 after auth is done. "
        "Please write regression tests for billing this week."
    )

    result = generate_plan(
        difficulty=Difficulty.MEDIUM,
        transcript=transcript,
        strategy="heuristic",
    )

    assert result.transcript
    assert len(result.extracted_items) >= 2
    assert len(result.jira_tickets) >= 2
    assert len(result.assignments) >= 1
    assert 0.0 <= result.score <= 1.0
