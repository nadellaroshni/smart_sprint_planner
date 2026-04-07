"""
JIRA ticket generator.

Converts extracted meeting items into enriched task objects with:
  - story point estimation
  - tag expansion
  - dependency inference
  - richer acceptance and context descriptions
"""

from __future__ import annotations

import logging
import re
from typing import List, Tuple

from .models import ExtractedItem, Priority, Task, TaskStatus

logger = logging.getLogger(__name__)

_SP_RULES: List[Tuple[str, int]] = [
    (r"\b(typo|style|css|colour|color|avatar|icon|tooltip)\b", 1),
    (r"\b(bug|fix|hotfix|patch|minor)\b", 2),
    (r"\b(unit test|write test|test coverage|regression)\b", 2),
    (r"\b(ci.?cd|pipeline|github action|workflow)\b", 3),
    (r"\b(dashboard|chart|analytics|ui|page|form|component)\b", 3),
    (r"\b(api|endpoint|route|controller)\b", 3),
    (r"\b(oauth|sso|authentication|login|auth)\b", 5),
    (r"\b(feature|module|service|integration)\b", 5),
    (r"\b(migrate|migration|refactor|database|postgres|mongo)\b", 8),
    (r"\b(architecture|redesign|overhaul|system)\b", 13),
]

_BLOCKER_PAIRS = [
    ("login", "dashboard"),
    ("auth", "payments"),
    ("database", "api"),
    ("api", "frontend"),
    ("ci", "deploy"),
]


def estimate_story_points(text: str) -> int:
    lower = text.lower()
    for pattern, points in _SP_RULES:
        if re.search(pattern, lower):
            return points
    return 3


def infer_tags(item: ExtractedItem) -> List[str]:
    combined = set(item.tags)
    lower = f"{item.task} {item.description} {item.raw_text}".lower()
    for keyword in [
        "bug",
        "auth",
        "payments",
        "frontend",
        "backend",
        "infra",
        "testing",
        "database",
        "analytics",
        "ci",
        "deploy",
        "security",
        "documentation",
        "performance",
    ]:
        if keyword in lower:
            combined.add(keyword)
    return sorted(combined)


def _infer_dependencies(tickets: List[Task]) -> None:
    for blocker_kw, dependent_kw in _BLOCKER_PAIRS:
        blockers = [
            t for t in tickets if blocker_kw in t.title.lower() or blocker_kw in " ".join(t.tags)
        ]
        dependents = [
            t for t in tickets if dependent_kw in t.title.lower() or dependent_kw in " ".join(t.tags)
        ]
        for dep in dependents:
            for blk in blockers:
                if blk.id != dep.id and blk.id not in dep.dependencies:
                    dep.dependencies.append(blk.id)


def _apply_dependency_hints(tickets: List[Task], items: List[ExtractedItem]) -> None:
    titles = {ticket.id: ticket.title.lower() for ticket in tickets}
    for ticket, item in zip(tickets, items):
        for hint in item.dependency_hints:
            hint_lower = hint.lower()
            for other in tickets:
                if other.id == ticket.id:
                    continue
                if any(word in titles[other.id] for word in hint_lower.split() if len(word) > 3):
                    if other.id not in ticket.dependencies:
                        ticket.dependencies.append(other.id)


def generate_description(item: ExtractedItem, tags: List[str]) -> str:
    parts = [f"Task: {item.task}."]
    if item.description:
        parts.append(f"Context: {item.description}")
    if item.urgency_reason:
        parts.append(f"Urgency: {item.urgency_reason}.")

    if "bug" in tags:
        parts.append("Reproduce the issue, ship the fix, and add regression coverage.")
    elif "testing" in tags:
        parts.append("Increase test confidence for the target flow and keep CI green.")
    elif "frontend" in tags:
        parts.append("Verify behavior on desktop and mobile, including cross-browser checks.")
    elif "auth" in tags:
        parts.append("Validate auth flow, token handling, and security-sensitive edge cases.")
    elif "infra" in tags:
        parts.append("Keep the deployment path stable and document rollback or recovery steps.")
    else:
        parts.append("Deliver reviewed, production-ready work.")

    if item.acceptance_criteria:
        parts.append("Acceptance: " + "; ".join(item.acceptance_criteria) + ".")
    if item.owner_hint:
        parts.append(f"Suggested owner: {item.owner_hint}.")
    return " ".join(parts)


def create_tickets(items: List[ExtractedItem]) -> List[Task]:
    tickets: List[Task] = []

    for i, item in enumerate(items):
        tags = infer_tags(item)
        sp = estimate_story_points(f"{item.task} {item.description} {item.raw_text}")
        description = generate_description(item, tags)

        ticket = Task(
            id=f"T{i + 1:03d}",
            title=item.task,
            description=description,
            story_points=sp,
            deadline=item.deadline,
            priority=Priority(item.priority),
            status=TaskStatus.BACKLOG,
            tags=tags,
            dependencies=[],
        )
        tickets.append(ticket)

    _infer_dependencies(tickets)
    _apply_dependency_hints(tickets, items)
    logger.info(f"Generated {len(tickets)} JIRA tickets.")
    return tickets
