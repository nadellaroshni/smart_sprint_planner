"""
Grading & Reward Engine.

Implements a DENSE, multi-signal reward function to guide RL agents:

Per-step rewards:
  +0.5   Task completed on time
  +0.2   Developer specialisation matches task tags
  +0.1   High-priority task completed
  -0.3   Task completed late (past deadline)
  -0.2   Developer over-capacity (rejected action)
  -0.1   Invalid task/developer ID
  -0.4   Blocked task assigned before dependency resolved

End-of-episode bonuses:
  +1.0   All tasks completed
  +0.5   Workload balanced across devs (Gini < 0.2)
  +0.3   No deadline violations

Final hackathon grade (0.0 – 1.0):
  30% completion rate
  25% on-time delivery
  20% extraction quality
  15% workload balance
  10% efficiency (steps used)
"""

from __future__ import annotations

import logging
from typing import Dict, Any, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-step reward
# ---------------------------------------------------------------------------

def compute_step_reward(
    task_found: bool,
    dev_found: bool,
    has_capacity: bool,
    is_blocked: bool,
    on_time: bool,
    skill_match: bool,
    is_high_priority: bool,
) -> Tuple[float, Dict[str, float]]:
    """
    Returns (total_reward, breakdown_dict).
    """
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

    # Base completion reward
    if on_time:
        breakdown["on_time"] = 0.5
    else:
        breakdown["late_penalty"] = -0.3

    if skill_match:
        breakdown["skill_match"] = 0.2

    if is_high_priority:
        breakdown["high_priority"] = 0.1

    return sum(breakdown.values()), breakdown


# ---------------------------------------------------------------------------
# End-of-episode bonuses
# ---------------------------------------------------------------------------

def compute_episode_bonus(env_state: dict, max_steps: int, steps_used: int) -> Tuple[float, Dict[str, float]]:
    completed = env_state.get("completed", [])
    tickets = env_state.get("tickets", [])
    developers = env_state.get("developers", [])

    breakdown: Dict[str, float] = {}

    # Completion bonus
    total = len(completed) + len(tickets)
    if total > 0 and len(tickets) == 0:
        breakdown["all_completed"] = 1.0

    # Workload balance (Gini coefficient on consumed capacity)
    initial_caps = env_state.get("initial_capacities", {})
    consumed = []
    for dev in developers:
        initial = initial_caps.get(dev["id"], dev["capacity"])
        consumed.append(initial - dev["capacity"])
    gini = _gini(consumed)
    if gini < 0.2:
        breakdown["balanced_workload"] = 0.5
    elif gini < 0.4:
        breakdown["moderate_balance"] = 0.2

    # No deadline violations
    if env_state.get("deadline_violations", 0) == 0 and len(completed) > 0:
        breakdown["no_violations"] = 0.3

    # Efficiency bonus (fewer steps = better)
    if steps_used <= max_steps * 0.7 and len(tickets) == 0:
        breakdown["efficiency"] = 0.2

    return sum(breakdown.values()), breakdown


# ---------------------------------------------------------------------------
# Final hackathon grade
# ---------------------------------------------------------------------------

def grade(env) -> Dict[str, Any]:
    """
    Compute final score in [0, 1] with full breakdown.

    Weights:
      completion_rate     30%
      on_time_rate        25%
      extraction_quality  20%
      workload_balance    15%
      efficiency          10%
    """
    state = env.state()
    completed = state.get("completed", [])
    tickets = state.get("tickets", [])
    developers = state.get("developers", [])
    extracted = state.get("extracted_items", [])
    deadline_violations = state.get("deadline_violations", 0)
    initial_caps = state.get("initial_capacities", {})

    total_tasks = len(completed) + len(tickets)

    # 1. Completion rate
    completion_rate = len(completed) / max(total_tasks, 1)

    # 2. On-time delivery rate
    on_time = len(completed) - deadline_violations
    on_time_rate = on_time / max(len(completed), 1)

    # 3. Extraction quality (FIXED)
    extraction_quality = min(len(extracted) / max(len(completed) + 1, 1), 1.0)
    if len(extracted) == 0:
        extraction_quality = 0.0

    # 4. Workload balance (SMOOTHED)
    gini = _gini(consumed)
    workload_balance = max(0.0, 1.0 - (gini * 0.7))

    # 5. Efficiency (FIXED)
    efficiency = len(completed) / max(total_tasks, 1)

    # Weighted sum
    score = (
        0.30 * completion_rate
        + 0.25 * on_time_rate
        + 0.20 * extraction_quality
        + 0.15 * workload_balance
        + 0.10 * efficiency
    )

    breakdown = {
        "completion_rate": round(completion_rate, 3),
        "on_time_rate": round(on_time_rate, 3),
        "extraction_quality": round(extraction_quality, 3),
        "workload_balance": round(workload_balance, 3),
        "efficiency": round(efficiency, 3),
        "final_score": round(score, 3),
    }

    summary = (
        f"Score: {score:.2f} | "
        f"Completed {len(completed)}/{total_tasks} tasks | "
        f"On-time: {on_time_rate:.0%} | "
        f"Balance: {workload_balance:.2f} | "
        f"Efficiency: {efficiency:.2f}"
    )

    logger.info(summary)
    return {"score": round(score, 3), "breakdown": breakdown, "summary": summary}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _gini(values: list) -> float:
    """Gini coefficient – 0 = perfect equality, 1 = max inequality."""
    if not values or sum(values) == 0:
        return 0.0
    n = len(values)
    values = sorted(values)
    cumsum = 0.0
    for i, v in enumerate(values, 1):
        cumsum += v * (2 * i - n - 1)
    return cumsum / (n * sum(values))