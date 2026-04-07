"""
Feature Engineering — Observation → Feature Vector.

Converts the structured Observation (tickets, developers, sprint day)
into a flat numerical vector the RL agent can consume.

Feature groups:
  1. Sprint progress features  (3)
  2. Task features             (num_tasks × 7)
  3. Developer features        (num_devs × 5)
  4. Task-dev affinity matrix  (num_tasks × num_devs)

Total size depends on the scenario; we pad to a fixed max size
so the agent's weight matrix never changes dimensions.
"""

from __future__ import annotations

import numpy as np
from typing import List, Tuple

from env.models import Observation, Task, Developer

# ── Fixed caps (pad/truncate to these) ──────────────────────────────────────
MAX_TASKS = 12
MAX_DEVS  = 4

FEATURE_DIM = (
    5                            # sprint progress + event context
    + MAX_TASKS * 7              # task features
    + MAX_DEVS  * 5              # dev features
    + MAX_TASKS * MAX_DEVS       # affinity
)
# = 5 + 84 + 20 + 48 = 157


def encode(obs: Observation) -> np.ndarray:
    """
    Encode an Observation into a float32 feature vector of shape (FEATURE_DIM,).
    All values are normalised to [0, 1] where possible.
    """
    feats: List[float] = []

    # 1. Sprint progress ──────────────────────────────────────────────────────
    max_steps = 20
    feats.append(obs.sprint_day / max_steps)                  # how far into sprint
    feats.append(len(obs.jira_tickets) / max(MAX_TASKS, 1))   # fraction backlog remaining
    total_sp = sum(t.story_points for t in obs.jira_tickets)
    feats.append(min(total_sp / 50.0, 1.0))                   # story point pressure
    recent_events = obs.recent_events[-3:]
    feats.append(min(len(recent_events) / 3.0, 1.0))          # recent disruption count
    feats.append(
        float(any(event.type.value in {"add_task", "capacity_change"} for event in recent_events))
    )

    # 2. Task features ────────────────────────────────────────────────────────
    tasks = obs.jira_tickets[:MAX_TASKS]
    for t in tasks:
        days_left = max(t.deadline - obs.sprint_day, 0)
        feats.extend([
            t.story_points / 13.0,                  # normalised SP
            (t.priority - 1) / 3.0,                 # normalised priority 0..1
            min(days_left / 10.0, 1.0),             # urgency (0=already late)
            1.0 - min(days_left / 10.0, 1.0),       # inverse urgency (deadline pressure)
            float("bug" in t.tags),                 # is bug?
            float(bool(t.dependencies)),            # has dependencies?
            float(t.status.value == "blocked"),     # is currently blocked?
        ])
    # Pad missing tasks with zeros
    for _ in range(MAX_TASKS - len(tasks)):
        feats.extend([0.0] * 7)

    # 3. Developer features ───────────────────────────────────────────────────
    devs = obs.developers[:MAX_DEVS]
    for d in devs:
        initial_cap = 13  # conservative upper bound
        feats.extend([
            d.capacity / initial_cap,               # remaining capacity fraction
            d.skill,                                # skill level
            float(len(d.active_tasks) > 0),        # is busy?
            len(d.active_tasks) / max(MAX_TASKS, 1),  # workload fraction
            len(d.specializations) / 5.0,          # breadth of specialisations
        ])
    for _ in range(MAX_DEVS - len(devs)):
        feats.extend([0.0] * 5)

    # 4. Task-developer affinity ──────────────────────────────────────────────
    # affinity[i][j] = 1 if dev j specialises in at least one of task i's tags
    for t in obs.jira_tickets[:MAX_TASKS]:
        task_tags = set(t.tags)
        for d in obs.developers[:MAX_DEVS]:
            match = float(bool(task_tags & set(d.specializations)))
            feats.append(match)
        for _ in range(MAX_DEVS - len(obs.developers[:MAX_DEVS])):
            feats.append(0.0)
    for _ in range(MAX_TASKS - len(obs.jira_tickets[:MAX_TASKS])):
        feats.extend([0.0] * MAX_DEVS)

    assert len(feats) == FEATURE_DIM, f"Feature dim mismatch: {len(feats)} != {FEATURE_DIM}"
    return np.array(feats, dtype=np.float32)


def action_space(obs: Observation) -> List[Tuple[str, str]]:
    """
    Enumerate all valid (task_id, dev_id) pairs for the current observation.

    Filters out:
      - tasks that have no dev with sufficient capacity
    The env still applies its own guard, but this prunes the action space.
    """
    actions = []
    for task in obs.jira_tickets:
        for dev in obs.developers:
            if dev.capacity >= task.story_points:
                actions.append((task.id, dev.id))
    return actions


def action_index(actions: List[Tuple[str, str]], task_id: str, dev_id: str) -> int:
    """Return the index of (task_id, dev_id) in a pre-built action list."""
    return actions.index((task_id, dev_id))
