"""
Feature engineering for SprintEnv observations and actions.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from env.models import EventType, Observation

MAX_TASKS = 12
MAX_DEVS = 4
EVENT_TYPES = [
    EventType.ADD_TASK.value,
    EventType.CAPACITY_CHANGE.value,
    EventType.ADD_DEPENDENCY.value,
    EventType.REMOVE_DEPENDENCY.value,
]

GLOBAL_FEATURES = 18
TASK_FEATURES = 16
DEV_FEATURES = 7
ACTION_FEATURE_DIM = 16

FEATURE_DIM = (
    GLOBAL_FEATURES
    + MAX_TASKS * TASK_FEATURES
    + MAX_DEVS * DEV_FEATURES
    + MAX_TASKS * MAX_DEVS
)


def _event_counts(events) -> List[float]:
    counts = []
    for event_type in EVENT_TYPES:
        counts.append(sum(1 for event in events if event.type.value == event_type))
    return counts


def _event_type_flags(event_type: str | None) -> List[float]:
    return [float(event_type == known) for known in EVENT_TYPES]


def encode(obs: Observation) -> np.ndarray:
    feats: List[float] = []
    max_steps = 20

    pending_tasks = obs.jira_tickets
    total_sp = sum(task.story_points for task in pending_tasks)
    completed_ids = set(obs.completed_task_ids)
    blocked_count = sum(1 for task in pending_tasks if any(dep not in completed_ids for dep in task.dependencies))
    urgent_count = sum(1 for task in pending_tasks if int(task.priority) >= 3)
    urgent_blocked_count = sum(
        1 for task in pending_tasks if int(task.priority) >= 3 and any(dep not in completed_ids for dep in task.dependencies)
    )
    event_added_count = sum(1 for task in pending_tasks if bool(task.source_event))
    avg_deadline_pressure = np.mean(
        [1.0 - min(max(task.deadline - obs.sprint_day, 0) / 10.0, 1.0) for task in pending_tasks]
    ) if pending_tasks else 0.0
    recent_event_counts = _event_counts(obs.recent_events)
    pending_event_counts = _event_counts(obs.pending_events)

    feats.extend([
        obs.sprint_day / max_steps,
        len(pending_tasks) / max(MAX_TASKS, 1),
        min(total_sp / 80.0, 1.0),
        blocked_count / max(len(pending_tasks), 1),
        urgent_count / max(len(pending_tasks), 1),
        urgent_blocked_count / max(len(pending_tasks), 1),
        event_added_count / max(len(pending_tasks), 1),
        len(obs.completed_task_ids) / max(MAX_TASKS, 1),
        sum(dev.capacity for dev in obs.developers) / 45.0,
        avg_deadline_pressure,
        *[min(count / 3.0, 1.0) for count in recent_event_counts],
        *[min(count / 4.0, 1.0) for count in pending_event_counts],
    ])

    tasks = pending_tasks[:MAX_TASKS]
    for task in tasks:
        days_left = max(task.deadline - obs.sprint_day, 0)
        dependency_open = any(dep not in completed_ids for dep in task.dependencies)
        feats.extend([
            task.story_points / 13.0,
            (int(task.priority) - 1) / 3.0,
            min(days_left / 10.0, 1.0),
            1.0 - min(days_left / 10.0, 1.0),
            float("bug" in task.tags),
            float("security" in task.tags),
            float("payments" in task.tags),
            float("backend" in task.tags),
            float("frontend" in task.tags),
            float(bool(task.dependencies)),
            float(dependency_open),
            float(bool(task.source_event)),
            *_event_type_flags(task.source_event_type),
        ])
    for _ in range(MAX_TASKS - len(tasks)):
        feats.extend([0.0] * TASK_FEATURES)

    devs = obs.developers[:MAX_DEVS]
    impacted_devs = {
        event.payload.get("developer_id")
        for event in obs.pending_events
        if event.type.value == EventType.CAPACITY_CHANGE.value
    }
    for dev in devs:
        feats.extend([
            dev.capacity / 13.0,
            dev.skill,
            float(len(dev.active_tasks) > 0),
            len(dev.active_tasks) / max(MAX_TASKS, 1),
            len(dev.specializations) / 6.0,
            float(dev.capacity <= 2),
            float(dev.id in impacted_devs),
        ])
    for _ in range(MAX_DEVS - len(devs)):
        feats.extend([0.0] * DEV_FEATURES)

    for task in tasks:
        task_tags = set(task.tags)
        dependency_open = any(dep not in completed_ids for dep in task.dependencies)
        for dev in devs:
            affinity = float(bool(task_tags & set(dev.specializations)))
            feasible = float(dev.capacity >= task.story_points and not dependency_open)
            impacted = float(dev.id in impacted_devs)
            feats.append(0.6 * affinity + 0.3 * feasible + 0.1 * (1.0 - impacted))
        for _ in range(MAX_DEVS - len(devs)):
            feats.append(0.0)
    for _ in range(MAX_TASKS - len(tasks)):
        feats.extend([0.0] * MAX_DEVS)

    assert len(feats) == FEATURE_DIM, f"Feature dim mismatch: {len(feats)} != {FEATURE_DIM}"
    return np.asarray(feats, dtype=np.float32)


def action_space(obs: Observation) -> List[Tuple[str, str]]:
    actions: List[Tuple[str, str]] = []
    completed_ids = set(obs.completed_task_ids)
    for task in obs.jira_tickets:
        if any(dep not in completed_ids for dep in task.dependencies):
            continue
        for dev in obs.developers:
            if dev.capacity >= task.story_points:
                actions.append((task.id, dev.id))
    return actions


def action_embedding(obs: Observation, task_id: str, dev_id: str) -> np.ndarray:
    task = next((task for task in obs.jira_tickets if task.id == task_id), None)
    dev = next((dev for dev in obs.developers if dev.id == dev_id), None)
    if task is None or dev is None:
        return np.zeros(ACTION_FEATURE_DIM, dtype=np.float32)

    task_idx = next((i for i, t in enumerate(obs.jira_tickets[:MAX_TASKS]) if t.id == task_id), 0)
    dev_idx = next((i for i, d in enumerate(obs.developers[:MAX_DEVS]) if d.id == dev_id), 0)
    days_left = max(task.deadline - obs.sprint_day, 0)
    skill_match = float(bool(set(task.tags) & set(dev.specializations)))
    impacted = any(
        event.type.value == EventType.CAPACITY_CHANGE.value and event.payload.get("developer_id") == dev.id
        for event in obs.pending_events
    )
    future_dependency_relevant = any(
        event.type.value == EventType.ADD_DEPENDENCY.value
        and (event.payload.get("depends_on") == task.id or str(event.payload.get("depends_on", "")).strip().lower() == task.title.strip().lower())
        for event in obs.pending_events
    )

    return np.asarray([
        task_idx / max(min(len(obs.jira_tickets), MAX_TASKS) - 1, 1),
        dev_idx / max(min(len(obs.developers), MAX_DEVS) - 1, 1),
        task.story_points / 13.0,
        (int(task.priority) - 1) / 3.0,
        1.0 - min(days_left / 10.0, 1.0),
        float(bool(task.dependencies)),
        float(bool(task.source_event)),
        *_event_type_flags(task.source_event_type),
        dev.capacity / 13.0,
        dev.skill,
        skill_match,
        float(impacted),
        float(future_dependency_relevant),
    ], dtype=np.float32)


def valid_action_embeddings(obs: Observation) -> List[np.ndarray]:
    return [action_embedding(obs, task_id, dev_id) for task_id, dev_id in action_space(obs)]
