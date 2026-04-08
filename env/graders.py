"""
Grading and reward engine.

This version adds explicit signal for dynamic re-planning quality, not just
static assignment quality.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Tuple

logger = logging.getLogger(__name__)


def compute_step_reward(
    task_found: bool,
    dev_found: bool,
    has_capacity: bool,
    is_blocked: bool,
    on_time: bool,
    skill_match: bool,
    is_high_priority: bool,
) -> Tuple[float, Dict[str, float]]:
    breakdown: Dict[str, float] = {}

    if not task_found or not dev_found:
        breakdown["invalid_ids"] = -0.1
        return sum(breakdown.values()), breakdown

    if is_blocked:
        breakdown["blocked_task"] = -0.4
        return sum(breakdown.values()), breakdown

    if not has_capacity:
        breakdown["over_capacity"] = -0.2
        return sum(breakdown.values()), breakdown

    breakdown["on_time" if on_time else "late_penalty"] = 0.5 if on_time else -0.3
    if skill_match:
        breakdown["skill_match"] = 0.2
    if is_high_priority:
        breakdown["high_priority"] = 0.1
    return sum(breakdown.values()), breakdown


def compute_adaptation_reward(
    task: dict,
    recent_events: list[dict],
    pending_events: list[dict],
    on_time: bool,
    unblocked_dependents: int = 0,
    outstanding_event_tasks: int = 0,
) -> Tuple[float, Dict[str, float]]:
    breakdown: Dict[str, float] = {}
    is_disruption_task = bool(task.get("source_event"))

    if is_disruption_task:
        breakdown["disruption_task_completed"] = 0.25
        if on_time:
            breakdown["disruption_task_on_time"] = 0.1

    if recent_events and pending_events:
        breakdown["maintained_progress_during_volatility"] = 0.05

    if recent_events and not pending_events:
        breakdown["stabilized_final_disruption"] = 0.05

    if unblocked_dependents > 0:
        breakdown["dependency_unblock"] = min(0.15, 0.05 * unblocked_dependents)

    if outstanding_event_tasks > 0 and is_disruption_task:
        breakdown["event_backlog_relief"] = min(0.1, 0.03 * outstanding_event_tasks)

    return sum(breakdown.values()), breakdown


def compute_episode_bonus(env_state: dict, max_steps: int, steps_used: int) -> Tuple[float, Dict[str, float]]:
    completed = env_state.get("completed", [])
    tickets = env_state.get("tickets", [])
    developers = env_state.get("developers", [])
    metrics = env_state.get("metrics", {})
    breakdown: Dict[str, float] = {}

    total = len(completed) + len(tickets)
    if total > 0 and len(tickets) == 0:
        breakdown["all_completed"] = 1.0

    initial_caps = env_state.get("initial_capacities", {})
    consumed = []
    for dev in developers:
        initial = initial_caps.get(dev["id"], dev["capacity"])
        consumed.append(max(initial - dev["capacity"], 0))
    gini = _gini(consumed)
    if gini < 0.2:
        breakdown["balanced_workload"] = 0.5
    elif gini < 0.4:
        breakdown["moderate_balance"] = 0.2

    if env_state.get("deadline_violations", 0) == 0 and len(completed) > 0:
        breakdown["no_violations"] = 0.3

    if steps_used <= max_steps * 0.7 and len(tickets) == 0:
        breakdown["efficiency"] = 0.2

    disruptions_applied = metrics.get("disruptions_applied", 0)
    disruption_tasks_added = metrics.get("disruption_tasks_added", 0)
    disruption_tasks_completed = metrics.get("disruption_tasks_completed", 0)
    if disruptions_applied > 0:
        handling = disruption_tasks_completed / max(disruption_tasks_added, 1)
        if handling >= 1.0:
            breakdown["fully_absorbed_disruptions"] = 0.4
        elif handling >= 0.5:
            breakdown["partially_absorbed_disruptions"] = 0.2

    return sum(breakdown.values()), breakdown


def grade(env) -> Dict[str, Any]:
    state = env.state()
    completed = state.get("completed", [])
    tickets = state.get("tickets", [])
    developers = state.get("developers", [])
    extracted = state.get("extracted_items", [])
    deadline_violations = state.get("deadline_violations", 0)
    initial_caps = state.get("initial_capacities", {})
    metrics = state.get("metrics", {})
    current_step = getattr(env, "current_step", 0)

    total_tasks = len(completed) + len(tickets)

    completion_rate = len(completed) / max(total_tasks, 1)
    on_time = len(completed) - deadline_violations
    on_time_rate = on_time / max(len(completed), 1)

    detailed_items = [
        item for item in extracted
        if item.get("description") and item.get("acceptance_criteria") and item.get("category")
    ]
    extraction_quality = len(detailed_items) / max(len(extracted), 1) if extracted else 0.0

    consumed = []
    for dev in developers:
        initial = initial_caps.get(dev["id"], dev["capacity"])
        consumed.append(max(initial - dev["capacity"], 0))
    gini = _gini(consumed)
    workload_balance = max(0.0, 1.0 - (gini * 0.7))

    max_steps = max(getattr(env, "max_steps", max(total_tasks, 1)), 1)
    step_pressure = min(current_step / max_steps, 1.0) if current_step > 0 else 0.0
    completion_efficiency = len(completed) / max(total_tasks, 1)
    efficiency = max(0.0, min(1.0, 0.6 * completion_efficiency + 0.4 * (1.0 - step_pressure)))

    disruptions_applied = metrics.get("disruptions_applied", 0)
    disruption_tasks_added = metrics.get("disruption_tasks_added", 0)
    disruption_tasks_completed = metrics.get("disruption_tasks_completed", 0)
    recovery_actions = metrics.get("recovery_actions", 0)
    if disruptions_applied == 0:
        adaptability = 1.0
    else:
        disruption_completion = disruption_tasks_completed / max(disruption_tasks_added, 1)
        reaction_quality = min(recovery_actions / max(disruptions_applied, 1), 1.0)
        adaptability = min(1.0, 0.7 * disruption_completion + 0.3 * reaction_quality)

    score = (
        0.25 * completion_rate
        + 0.20 * on_time_rate
        + 0.15 * extraction_quality
        + 0.10 * workload_balance
        + 0.10 * efficiency
        + 0.20 * adaptability
    )

    breakdown = {
        "completion_rate": round(completion_rate, 3),
        "on_time_rate": round(on_time_rate, 3),
        "extraction_quality": round(extraction_quality, 3),
        "workload_balance": round(workload_balance, 3),
        "efficiency": round(efficiency, 3),
        "adaptability": round(adaptability, 3),
        "final_score": round(score, 3),
    }

    summary = (
        f"Score: {score:.2f} | Completed {len(completed)}/{total_tasks} tasks | "
        f"On-time: {on_time_rate:.0%} | Adaptability: {adaptability:.2f}"
    )
    logger.info(summary)
    return {"score": round(score, 3), "breakdown": breakdown, "summary": summary}


def _gini(values: list[int]) -> float:
    if not values or sum(values) == 0:
        return 0.0
    values = sorted(values)
    n = len(values)
    cumsum = 0.0
    for i, value in enumerate(values, 1):
        cumsum += value * (2 * i - n - 1)
    return cumsum / (n * sum(values))


# Task-specific grader wrappers for explicit task grading
def grade_easy(env) -> Dict[str, Any]:
    """Grader for easy task: static sprint planning with fixed backlog."""
    return grade(env)


def grade_medium(env) -> Dict[str, Any]:
    """Grader for medium task: single disruption replanning."""
    return grade(env)


def grade_hard(env) -> Dict[str, Any]:
    """Grader for hard task: multi-disruption dynamic replanning."""
    return grade(env)
