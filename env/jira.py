"""
JIRA Ticket Generator.

Converts raw ExtractedItems into enriched Task objects with:
- Fibonacci story point estimation (1,2,3,5,8,13)
- Skill/tag inference
- Dependency detection (keyword-based)
- Acceptance criteria generation
"""

from __future__ import annotations

import re
import logging
from typing import List, Tuple

from .models import ExtractedItem, Task, Priority, TaskStatus

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Story point estimation table
# ---------------------------------------------------------------------------

# (regex pattern, story points)
_SP_RULES: List[Tuple[str, int]] = [
    # Quick fixes
    (r"\b(typo|style|css|colour|color|avatar|icon|tooltip)\b", 1),
    # Small bugs
    (r"\b(bug|fix|hotfix|patch|minor)\b", 2),
    # Standard tasks
    (r"\b(unit test|write test|test coverage)\b", 2),
    (r"\b(ci.?cd|pipeline|github action|workflow)\b", 3),
    # Medium features
    (r"\b(dashboard|chart|analytics|ui|page|form|component)\b", 3),
    (r"\b(api|endpoint|route|controller)\b", 3),
    # Complex features
    (r"\b(oauth|sso|authentication|login|auth)\b", 5),
    (r"\b(feature|module|service|integration)\b", 5),
    # Heavy work
    (r"\b(migrate|migration|refactor|database|postgres|mongo)\b", 8),
    (r"\b(architecture|redesign|overhaul|system)\b", 13),
]

# ---------------------------------------------------------------------------
# Tag → skill/specialization mapping
# ---------------------------------------------------------------------------

_TAG_SKILLS = {
    "bug": ["debugging"],
    "auth": ["backend", "security"],
    "payments": ["backend", "payments"],
    "frontend": ["frontend"],
    "backend": ["backend"],
    "infra": ["devops", "infra"],
    "testing": ["testing", "qa"],
    "database": ["backend", "database"],
    "analytics": ["frontend", "data"],
}

# ---------------------------------------------------------------------------
# Dependency inference (keyword proximity)
# ---------------------------------------------------------------------------

_BLOCKER_PAIRS = [
    ("login", "dashboard"),
    ("auth", "payments"),
    ("database", "api"),
    ("api", "frontend"),
    ("ci", "deploy"),
]


def estimate_story_points(text: str) -> int:
    """
    Fibonacci story point estimation using keyword rules.
    Falls back to 3 (median) if no rule matches.
    """
    lower = text.lower()
    for pattern, points in _SP_RULES:
        if re.search(pattern, lower):
            return points
    return 3


def infer_tags(item: ExtractedItem) -> List[str]:
    """Merge existing tags with additional inferred tags from text."""
    combined = set(item.tags)
    lower = (item.task + " " + item.raw_text).lower()
    for keyword in ["bug", "auth", "payments", "frontend", "backend",
                    "infra", "testing", "database", "analytics", "ci", "deploy"]:
        if keyword in lower:
            combined.add(keyword)
    return list(combined)


def _infer_dependencies(tickets: List[Task]) -> None:
    """
    Mutate tickets in-place to add dependency edges.
    Uses heuristic blocker pairs.
    """
    id_map = {t.id: t for t in tickets}

    for blocker_kw, dependent_kw in _BLOCKER_PAIRS:
        blockers = [
            t for t in tickets
            if blocker_kw in t.title.lower() or blocker_kw in " ".join(t.tags)
        ]
        dependents = [
            t for t in tickets
            if dependent_kw in t.title.lower() or dependent_kw in " ".join(t.tags)
        ]
        for dep in dependents:
            for blk in blockers:
                if blk.id != dep.id and blk.id not in dep.dependencies:
                    dep.dependencies.append(blk.id)
                    logger.debug(f"Dependency inferred: {dep.id} depends on {blk.id}")


def generate_acceptance_criteria(task_title: str, tags: List[str]) -> str:
    """Minimal acceptance criteria string (for description field)."""
    base = f"Task: {task_title}."
    if "bug" in tags:
        base += " Reproduce → Fix → Write regression test."
    elif "testing" in tags:
        base += " Coverage ≥ 80% for target module."
    elif "frontend" in tags:
        base += " Responsive on mobile + desktop. Cross-browser tested."
    elif "auth" in tags:
        base += " OAuth flow verified. Token refresh tested. Security review passed."
    elif "infra" in tags:
        base += " Pipeline green on main branch. Rollback plan documented."
    else:
        base += " Feature complete, reviewed, and merged to main."
    return base


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def create_tickets(items: List[ExtractedItem]) -> List[Task]:
    """
    Convert extracted items to enriched JIRA-style Task objects.

    Assigns IDs, estimates story points, infers tags,
    and resolves inter-task dependencies.
    """
    tickets: List[Task] = []

    for i, item in enumerate(items):
        tags = infer_tags(item)
        sp = estimate_story_points(item.task + " " + item.raw_text)
        description = generate_acceptance_criteria(item.task, tags)

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
        logger.debug(f"Ticket {ticket.id}: {ticket.title} | SP={sp} | P={ticket.priority}")

    _infer_dependencies(tickets)
    logger.info(f"Generated {len(tickets)} JIRA tickets.")
    return tickets