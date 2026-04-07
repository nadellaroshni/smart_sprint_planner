"""
Explicit task catalog for validator and API task discovery.
"""

from __future__ import annotations

from .models import Difficulty, TaskDescriptor


TASK_CATALOG: list[TaskDescriptor] = [
    TaskDescriptor(
        id="easy",
        name="easy",
        difficulty=Difficulty.EASY,
        objective=(
            "Create the best sprint plan from a static backlog with fixed team capacity "
            "and fixed deadlines."
        ),
        grader="env.graders.grade",
    ),
    TaskDescriptor(
        id="medium",
        name="medium",
        difficulty=Difficulty.MEDIUM,
        objective=(
            "Revise the sprint plan after one mid-sprint disruption such as urgent work "
            "or a developer capacity loss."
        ),
        grader="env.graders.grade",
    ),
    TaskDescriptor(
        id="hard",
        name="hard",
        difficulty=Difficulty.HARD,
        objective=(
            "Repeatedly re-plan under multiple disruptions including added work, "
            "capacity changes, and dependency shifts while preserving feasibility."
        ),
        grader="env.graders.grade",
    ),
]


def get_task_catalog() -> list[TaskDescriptor]:
    return TASK_CATALOG


def resolve_task_name(name: str | None) -> Difficulty | None:
    if not name:
        return None
    normalized = name.strip().lower()
    for task in TASK_CATALOG:
        if task.id == normalized or task.name == normalized:
            return task.difficulty
    return None
