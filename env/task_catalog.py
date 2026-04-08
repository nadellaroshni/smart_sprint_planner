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
            "Create the best sprint plan from a static backlog with fixed capacity and deadlines."
        ),
        grader="env.graders:grade_easy",
    ),
    TaskDescriptor(
        id="medium",
        name="medium",
        difficulty=Difficulty.MEDIUM,
        objective=(
            "Revise the sprint plan once after a single mid-sprint disruption (e.g., new urgent work or capacity loss)."
        ),
        grader="env.graders:grade_medium",
    ),
    TaskDescriptor(
        id="hard",
        name="hard",
        difficulty=Difficulty.HARD,
        objective=(
            "Re-plan repeatedly under multiple disruptions while preserving feasibility and value delivery."
        ),
        grader="env.graders:grade_hard",
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
