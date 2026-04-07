"""
Heuristic Agent — Rule-based baseline for comparison.

Strategy:
  1. Score every (task, dev) pair:
       urgency  = 10 / (days_remaining + 1)
       priority = task.priority × 1.5
       size     = 1 / story_points          ← prefer smaller tasks
       skill    = +2.0 if dev specialises in task tags
       capacity = dev.capacity × dev.skill
  2. Pick the pair with the highest combined score that passes
     the dependency guard.
  3. If all remaining tasks are blocked, skip one step.

This is a greedy one-step lookahead — no learning, no memory.
It serves as a performance ceiling baseline that the RL agent should surpass.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

from env.models import Observation, Action

logger = logging.getLogger(__name__)


class HeuristicAgent:
    """Greedy, rule-based sprint planning agent."""

    def act(self, obs: Observation, **kwargs) -> Optional[Action]:
        best_score = -999.0
        best_pair:  Optional[Tuple[str, str]] = None

        for task in obs.jira_tickets:
            # Respect dependency order
            if self._is_blocked(task, obs):
                continue

            for dev in obs.developers:
                if dev.capacity < task.story_points:
                    continue

                score = self._score(task, dev, obs.sprint_day)
                if score > best_score:
                    best_score = score
                    best_pair  = (task.id, dev.id)

        if best_pair is None:
            return None
        return Action(task_id=best_pair[0], developer_id=best_pair[1])

    def _score(self, task, dev, sprint_day: int) -> float:
        days_left  = max(task.deadline - sprint_day, 0)
        urgency    = 10.0 / (days_left + 1)
        priority   = task.priority * 1.5
        size       = 1.0 / task.story_points
        skill      = 2.0 if set(dev.specializations) & set(task.tags) else 0.0
        capacity   = dev.capacity * dev.skill
        return urgency + priority + size + skill + capacity * 0.1

    @staticmethod
    def _is_blocked(task, obs: Observation) -> bool:
        done_ids = set()  # env tracks this — we can only infer from obs
        # We can't see completed tasks in obs, so we rely on env rejections
        # but still check if tags indicate strong ordering
        return False   # env will penalise if blocked; heuristic tries anyway