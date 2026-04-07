from env.extraction import extract_tasks


def test_rule_based_extraction_captures_rich_fields():
    transcript = (
        "We need to fix the checkout bug today because it is blocking the release. "
        "Also build the analytics dashboard by day 5 after the auth work lands. "
        "Please write regression tests for billing this week."
    )

    items = extract_tasks(transcript)

    assert len(items) >= 3
    assert any(item.description for item in items)
    assert any(item.acceptance_criteria for item in items)
    assert any(item.category != "general" for item in items)
    assert any(item.urgency_reason for item in items)
    assert any(item.dependency_hints for item in items)
