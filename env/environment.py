"""
SprintEnv — Core RL Environment.

Implements the OpenEnv/Gym-compatible interface:
  reset() → Observation
  step(Action) → (Observation, reward, done, info)
  state() → dict
  render() → str  (human-readable sprint board)

Reward design: DENSE (signal on every step + episode bonuses)
"""

from __future__ import annotations

import copy
import logging
from typing import Any, Dict, Tuple, Optional

from .models import (
    Observation, Action, StepResult, Difficulty,
    TaskStatus, Priority, SprintMetrics
)
from .transcription import transcribe, transcribe_from_text, _fallback_transcript
from .extraction import extract_tasks
from .jira import create_tickets
from .tasks import get_developers, get_transcript, get_extracted_items
from .graders import compute_step_reward, compute_episode_bonus

logger = logging.getLogger(__name__)


class SprintEnv:
    """
    Smart Sprint Planner RL Environment.

    Simulates an Agile sprint:
    1. Meeting transcript → transcription
    2. Transcript → action item extraction (LLM or rule-based)
    3. Action items → JIRA tickets
    4. RL agent assigns tickets to developers across sprint days
    5. Dense reward feedback drives learning
    """

    def __init__(
        self,
        difficulty: Difficulty = Difficulty.MEDIUM,
        max_steps: int = 20,
        use_llm: bool = True,
    ):
        self.difficulty = difficulty
        self.max_steps = max_steps
        self.use_llm = use_llm
        self.current_step: int = 0
        self._state: Dict[str, Any] = {}
        self._metrics = SprintMetrics()

    # ------------------------------------------------------------------
    # Public RL interface
    # ------------------------------------------------------------------

    def reset(
        self,
        difficulty: Optional[Difficulty] = None,
        audio_path: Optional[str] = None,
        transcript_override: Optional[str] = None,
    ) -> Observation:
        """
        Reset the environment to the start of a new sprint.

        Priority for transcript source:
          1. transcript_override (direct text)
          2. audio_path (Whisper transcription)
          3. Deterministic transcript for difficulty level
        """
        if difficulty:
            self.difficulty = difficulty

        self.current_step = 0
        self._metrics = SprintMetrics()

        # --- Transcription ---
        if transcript_override:
            transcript = transcript_override
        elif audio_path:
            transcript = transcribe(audio_path)
        else:
            transcript = get_transcript(self.difficulty)

        # --- Extraction ---
        if self.use_llm:
            try:
                extracted = extract_tasks(transcript)
            except Exception:
                logger.warning("LLM extraction failed, using pre-baked items.")
                extracted = get_extracted_items(self.difficulty)
        else:
            extracted = get_extracted_items(self.difficulty)

        # --- JIRA ticket generation ---
        tickets = create_tickets(extracted)

        # --- Developers ---
        developers = get_developers(self.difficulty)
        initial_caps = {d.id: d.capacity for d in developers}

        self._state = {
            "meeting_text": transcript,
            "extracted_items": extracted,
            "tickets": [t.model_dump() for t in tickets],
            "developers": [d.model_dump() for d in developers],
            "completed": [],
            "deadline_violations": 0,
            "initial_capacities": initial_caps,
        }

        logger.info(
            f"SprintEnv reset | difficulty={self.difficulty} | "
            f"tickets={len(tickets)} | developers={len(developers)}"
        )
        return self._get_obs()

    def step(self, action: Action) -> Tuple[Observation, float, bool, Dict]:
        """
        Execute one agent action: assign task_id → developer_id.

        Returns (observation, reward, done, info).
        """
        self.current_step += 1
        info: Dict[str, Any] = {}

        tickets = self._state["tickets"]
        developers = self._state["developers"]

        # --- Locate task and developer ---
        task = next((t for t in tickets if t["id"] == action.task_id), None)
        dev = next((d for d in developers if d["id"] == action.developer_id), None)

        task_found = task is not None
        dev_found = dev is not None

        # Early exit: invalid IDs
        if not task_found or not dev_found:
            reward, breakdown = compute_step_reward(
                task_found, dev_found, False, False, False, False, False
            )
            info = {"error": "invalid ids", "breakdown": breakdown}
            return self._get_obs(), reward, self._is_done(), info

        # --- Dependency check ---
        is_blocked = self._has_unresolved_deps(task)
        if is_blocked:
            reward, breakdown = compute_step_reward(
                True, True, True, True, False, False, False
            )
            task["status"] = TaskStatus.BLOCKED.value
            info = {"error": f"task {task['id']} is blocked", "breakdown": breakdown}
            self._metrics.tasks_blocked += 1
            return self._get_obs(), reward, self._is_done(), info

        # --- Capacity check ---
        has_capacity = dev["capacity"] >= task["story_points"]
        if not has_capacity:
            reward, breakdown = compute_step_reward(
                True, True, False, False, False, False, False
            )
            info = {"error": "insufficient capacity", "breakdown": breakdown}
            return self._get_obs(), reward, self._is_done(), info

        # --- Execute assignment ---
        on_time = task["deadline"] >= self.current_step
        skill_match = self._skill_matches(dev, task)
        is_high_priority = task["priority"] >= Priority.HIGH.value

        dev["capacity"] -= task["story_points"]
        dev["active_tasks"].append(task["id"])
        task["status"] = TaskStatus.DONE.value
        task["assigned_to"] = dev["id"]

        self._state["completed"].append(copy.deepcopy(task))
        tickets.remove(task)

        if not on_time:
            self._state["deadline_violations"] += 1
            self._metrics.tasks_failed_deadline += 1

        self._metrics.tasks_completed += 1

        reward, breakdown = compute_step_reward(
            True, True, True, False, on_time, skill_match, is_high_priority
        )
        self._metrics.total_reward += reward

        info = {
            "task": task["id"],
            "developer": dev["id"],
            "on_time": on_time,
            "skill_match": skill_match,
            "reward_breakdown": breakdown,
        }
        logger.debug(
            f"Step {self.current_step}: {task['id']} → {dev['id']} | "
            f"reward={reward:.2f} | on_time={on_time}"
        )

        done = self._is_done()

        # Episode end bonus
        if done:
            bonus, bonus_breakdown = compute_episode_bonus(
                self._state, self.max_steps, self.current_step
            )
            reward += bonus
            self._metrics.total_reward += bonus
            info["episode_bonus"] = bonus
            info["bonus_breakdown"] = bonus_breakdown
            self._log_episode_summary()

        return self._get_obs(), reward, done, info

    def state(self) -> Dict[str, Any]:
        return self._state

    def render(self) -> str:
        """Human-readable sprint board (for logging / debugging)."""
        lines = [
            f"\n{'=' * 60}",
            f"  SPRINT BOARD — Day {self.current_step} / {self.max_steps}",
            f"  Difficulty: {self.difficulty}",
            f"{'=' * 60}",
            f"  BACKLOG ({len(self._state['tickets'])} tasks):",
        ]
        for t in self._state["tickets"]:
            deps = f" [blocked by: {t['dependencies']}]" if t.get("dependencies") else ""
            lines.append(
                f"    [{t['id']}] {t['title'][:40]} | "
                f"P{t['priority']} | {t['story_points']}sp | day {t['deadline']}{deps}"
            )

        lines.append(f"\n  COMPLETED ({len(self._state['completed'])} tasks):")
        for t in self._state["completed"]:
            lines.append(f"    ✓ [{t['id']}] {t['title'][:40]} → {t.get('assigned_to', '?')}")

        lines.append(f"\n  DEVELOPERS:")
        for d in self._state["developers"]:
            initial = self._state["initial_capacities"].get(d["id"], d["capacity"])
            used = initial - d["capacity"]
            bar = "█" * used + "░" * d["capacity"]
            lines.append(f"    {d['id']} {d['name']}: [{bar}] {used}/{initial}sp")

        lines.append(
            f"\n  Metrics: completed={self._metrics.tasks_completed} | "
            f"violations={self._state.get('deadline_violations', 0)} | "
            f"total_reward={self._metrics.total_reward:.2f}"
        )
        lines.append("=" * 60)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_obs(self) -> Observation:
        from .models import Task, Developer, ExtractedItem
        return Observation(
            meeting_text=self._state["meeting_text"],
            extracted_items=[
                ExtractedItem(**i) if isinstance(i, dict) else i
                for i in self._state["extracted_items"]
            ],
            jira_tickets=[
                Task(**t) if isinstance(t, dict) else t
                for t in self._state["tickets"]
            ],
            developers=[
                Developer(**d) if isinstance(d, dict) else d
                for d in self._state["developers"]
            ],
            sprint_day=self.current_step,
            metrics=self._metrics,
            difficulty=self.difficulty,
        )

    def _is_done(self) -> bool:
        return (
            len(self._state["tickets"]) == 0
            or self.current_step >= self.max_steps
        )

    def _has_unresolved_deps(self, task: dict) -> bool:
        completed_ids = {t["id"] for t in self._state["completed"]}
        return any(dep not in completed_ids for dep in task.get("dependencies", []))

    def _skill_matches(self, dev: dict, task: dict) -> bool:
        dev_skills = set(dev.get("specializations", []))
        task_tags = set(task.get("tags", []))
        return bool(dev_skills & task_tags)

    def _log_episode_summary(self) -> None:
        completed = len(self._state["completed"])
        total = completed + len(self._state["tickets"])
        violations = self._state.get("deadline_violations", 0)
        logger.info(
            f"Episode complete | {completed}/{total} tasks | "
            f"{violations} deadline violations | "
            f"total_reward={self._metrics.total_reward:.2f}"
        )