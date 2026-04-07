"""
Heuristic Agent: rule-based baseline for sprint planning.

The policy is intentionally simple:
  - prefer urgent, high-priority, smaller tasks
  - prefer developers with matching specializations
  - skip tasks whose dependencies are not yet completed
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

from env.models import Action, Observation

logger = logging.getLogger(__name__)


class HeuristicAgent:
    """Greedy sprint planning baseline."""

    def act(self, obs: Observation, **kwargs) -> Optional[Action]:
        best_score = -999.0
        best_pair: Optional[Tuple[str, str]] = None

        for task in obs.jira_tickets:
            if self._is_blocked(task, obs):
                continue

            for dev in obs.developers:
                if dev.capacity < task.story_points:
                    continue

                score = self._score(task, dev, obs.sprint_day)
                if score > best_score:
                    best_score = score
                    best_pair = (task.id, dev.id)

        if best_pair is None:
            return None
        return Action(task_id=best_pair[0], developer_id=best_pair[1])

    def _score(self, task, dev, sprint_day: int) -> float:
        days_left = max(task.deadline - sprint_day, 0)
        urgency = 10.0 / (days_left + 1)
        priority = task.priority * 1.5
        size = 1.0 / task.story_points
        skill = 2.0 if set(dev.specializations) & set(task.tags) else 0.0
        capacity = dev.capacity * dev.skill
        return urgency + priority + size + skill + capacity * 0.1

    @staticmethod
    def _is_blocked(task, obs: Observation) -> bool:
        done_ids = set(obs.completed_task_ids)
        return any(dep not in done_ids for dep in task.dependencies)
