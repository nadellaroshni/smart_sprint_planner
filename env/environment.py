"""
SprintEnv: core RL environment for sprint planning and re-planning.
"""

from __future__ import annotations

import copy
import logging
import random
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

from .extraction import extract_tasks
from .graders import compute_adaptation_reward, compute_episode_bonus, compute_step_reward
from .jira import create_tickets
from .models import (
    Action,
    Developer,
    Difficulty,
    EventType,
    ExtractedItem,
    Observation,
    Priority,
    SprintEvent,
    SprintMetrics,
    Task,
    TaskStatus,
)
from .tasks import get_scenario
from .transcription import transcribe

logger = logging.getLogger(__name__)


class SprintEnv:
    def __init__(
        self,
        difficulty: Difficulty = Difficulty.MEDIUM,
        max_steps: int = 20,
        use_llm: bool = True,
        sample_scenarios: bool = False,
        scenario_split: Optional[str] = None,
        seed: int = 42,
    ):
        self.difficulty = difficulty
        self.max_steps = max_steps
        self.use_llm = use_llm
        self.sample_scenarios = sample_scenarios
        self.scenario_split = scenario_split
        self.current_step = 0
        self._state: Dict[str, Any] = {}
        self._metrics = SprintMetrics()
        self._next_ticket_num = 1
        self._rng = random.Random(seed)

    def reset(
        self,
        difficulty: Optional[Difficulty] = None,
        audio_path: Optional[str] = None,
        transcript_override: Optional[str] = None,
        scenario_index: Optional[int] = None,
    ) -> Observation:
        if difficulty is not None:
            self.difficulty = difficulty

        self.current_step = 0
        self._metrics = SprintMetrics()
        self._next_ticket_num = 1

        scenario = get_scenario(
            self.difficulty,
            sample=self.sample_scenarios,
            rng=self._rng,
            scenario_index=scenario_index,
            split=self.scenario_split,
        )

        if transcript_override:
            transcript = transcript_override
        elif audio_path:
            transcript = transcribe(audio_path)
        else:
            transcript = scenario["transcript"]  # type: ignore[assignment]

        if self.use_llm:
            try:
                extracted = extract_tasks(transcript)
            except Exception:
                logger.warning("LLM extraction failed, using pre-baked items.")
                extracted = deepcopy(scenario["items"])  # type: ignore[assignment]
        else:
            extracted = deepcopy(scenario["items"])  # type: ignore[assignment]

        tickets = create_tickets(extracted)
        developers = deepcopy(scenario["developers"])  # type: ignore[assignment]
        initial_caps = {d.id: d.capacity for d in developers}
        event_schedule = sorted(deepcopy(scenario["events"]), key=lambda e: e.day)  # type: ignore[arg-type]

        self._state = {
            "scenario_id": scenario.get("scenario_id", ""),
            "meeting_text": transcript,
            "extracted_items": [item.model_dump() for item in extracted],
            "tickets": [ticket.model_dump() for ticket in tickets],
            "developers": [dev.model_dump() for dev in developers],
            "completed": [],
            "deadline_violations": 0,
            "initial_capacities": initial_caps,
            "pending_events": [event.model_dump(mode="json") for event in event_schedule],
            "recent_events": [],
            "event_history": [],
            "metrics": self._metrics.model_dump(),
        }
        self._next_ticket_num = len(tickets) + 1
        return self._get_obs()

    def step(self, action: Action) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        self.current_step += 1
        self._state["recent_events"] = []
        info: Dict[str, Any] = {}

        task = self._find_ticket(action.task_id)
        dev = self._find_developer(action.developer_id)

        if task is None or dev is None:
            reward, breakdown = compute_step_reward(task is not None, dev is not None, False, False, False, False, False)
            info = {"error": "invalid ids", "breakdown": breakdown}
            return self._finish_step(reward, info)

        if self._has_unresolved_deps(task):
            reward, breakdown = compute_step_reward(True, True, True, True, False, False, False)
            task["status"] = TaskStatus.BLOCKED.value
            self._metrics.tasks_blocked += 1
            info = {"error": f"task {task['id']} is blocked", "breakdown": breakdown}
            return self._finish_step(reward, info)

        if dev["capacity"] < task["story_points"]:
            reward, breakdown = compute_step_reward(True, True, False, False, False, False, False)
            info = {"error": "insufficient capacity", "breakdown": breakdown}
            return self._finish_step(reward, info)

        on_time = task["deadline"] >= self.current_step
        skill_match = self._skill_matches(dev, task)
        is_high_priority = task["priority"] >= Priority.HIGH.value
        future_dependency_value = self._future_dependency_unlock_value(task)
        future_capacity_risk = self._future_capacity_loss_risk(dev["id"])
        preserve_capacity_penalty = self._preserve_specialist_capacity_penalty(task, dev)
        unblocked_dependents = sum(
            1 for other in self._state["tickets"]
            if other["id"] != task["id"] and task["id"] in other.get("dependencies", [])
        )
        outstanding_event_tasks = sum(
            1 for ticket in self._state["tickets"]
            if ticket.get("source_event") and ticket["id"] != task["id"]
        )
        feasible_before = self._count_feasible_tasks()

        dev["capacity"] -= task["story_points"]
        dev["active_tasks"].append(task["id"])
        task["status"] = TaskStatus.DONE.value
        task["assigned_to"] = dev["id"]

        self._state["completed"].append(copy.deepcopy(task))
        self._state["tickets"].remove(task)
        self._metrics.tasks_completed += 1

        if not on_time:
            self._state["deadline_violations"] += 1
            self._metrics.tasks_failed_deadline += 1

        feasible_after = self._count_feasible_tasks()

        reward, breakdown = compute_step_reward(True, True, True, False, on_time, skill_match, is_high_priority)
        adaptation_reward, adaptation_breakdown = compute_adaptation_reward(
            task=task,
            recent_events=self._state.get("recent_events", []),
            pending_events=self._state.get("pending_events", []),
            on_time=on_time,
            unblocked_dependents=unblocked_dependents,
            outstanding_event_tasks=outstanding_event_tasks,
        )
        reward += adaptation_reward
        breakdown.update(adaptation_breakdown)

        if future_dependency_value > 0:
            unlock_bonus = min(0.15, 0.05 * future_dependency_value)
            reward += unlock_bonus
            breakdown["future_dependency_preparation"] = unlock_bonus

        if preserve_capacity_penalty > 0 and future_capacity_risk > 0:
            penalty = min(0.15, preserve_capacity_penalty * future_capacity_risk)
            reward -= penalty
            breakdown["future_capacity_preservation_penalty"] = -penalty

        if len(self._state["tickets"]) > 0 and feasible_after == 0:
            reward -= 0.2
            breakdown["future_feasibility_dead_end"] = -0.2
        elif feasible_after >= max(feasible_before - 1, 1):
            reward += 0.05
            breakdown["future_feasibility_preserved"] = 0.05

        if task.get("source_event"):
            self._metrics.disruption_tasks_completed += 1
        if self._metrics.disruptions_applied > 0:
            self._metrics.recovery_actions += 1

        info = {
            "task": task["id"],
            "developer": dev["id"],
            "on_time": on_time,
            "skill_match": skill_match,
            "reward_breakdown": breakdown,
        }
        return self._finish_step(reward, info)

    def state(self) -> Dict[str, Any]:
        self._state["metrics"] = self._metrics.model_dump()
        return self._state

    def render(self) -> str:
        lines = [
            f"\n{'=' * 60}",
            f"  SPRINT BOARD - Day {self.current_step} / {self.max_steps}",
            f"  Difficulty: {self.difficulty}",
            f"{'=' * 60}",
            f"  BACKLOG ({len(self._state['tickets'])} tasks):",
        ]

        for task in self._state["tickets"]:
            deps = f" [blocked by: {task['dependencies']}]" if task.get("dependencies") else ""
            source = " [event]" if task.get("source_event") else ""
            lines.append(
                f"    [{task['id']}] {task['title'][:40]} | P{task['priority']} | "
                f"{task['story_points']}sp | day {task['deadline']}{deps}{source}"
            )

        lines.append(f"\n  COMPLETED ({len(self._state['completed'])} tasks):")
        for task in self._state["completed"]:
            source = " [event]" if task.get("source_event") else ""
            lines.append(f"    [done] [{task['id']}] {task['title'][:40]} -> {task.get('assigned_to', '?')}{source}")

        lines.append("\n  DEVELOPERS:")
        for dev in self._state["developers"]:
            initial = self._state["initial_capacities"].get(dev["id"], dev["capacity"])
            used = initial - dev["capacity"]
            bar = "#" * used + "." * dev["capacity"]
            lines.append(f"    {dev['id']} {dev['name']}: [{bar}] {used}/{initial}sp")

        if self._state.get("recent_events"):
            lines.append("\n  RECENT EVENTS:")
            for event in self._state["recent_events"]:
                lines.append(f"    ! Day {event['day']}: {event['title']} ({event['type']})")

        lines.append(
            f"\n  Metrics: completed={self._metrics.tasks_completed} | "
            f"violations={self._state.get('deadline_violations', 0)} | "
            f"disruptions={self._metrics.disruptions_applied} | "
            f"reward={self._metrics.total_reward:.2f}"
        )
        lines.append("=" * 60)
        return "\n".join(lines)

    def _finish_step(self, reward: float, info: Dict[str, Any]) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        applied_events = self._apply_scheduled_events()
        if applied_events:
            info["events"] = applied_events
        self._metrics.total_reward += reward
        self._state["metrics"] = self._metrics.model_dump()

        done = self._is_done()
        if done:
            bonus, bonus_breakdown = compute_episode_bonus(self._state, self.max_steps, self.current_step)
            reward += bonus
            self._metrics.total_reward += bonus
            self._state["metrics"] = self._metrics.model_dump()
            info["episode_bonus"] = bonus
            info["bonus_breakdown"] = bonus_breakdown
            self._log_episode_summary()

        return self._get_obs(), reward, done, info

    def _get_obs(self) -> Observation:
        return Observation(
            meeting_text=self._state["meeting_text"],
            extracted_items=[ExtractedItem(**item) for item in self._state["extracted_items"]],
            jira_tickets=[Task(**ticket) for ticket in self._state["tickets"]],
            developers=[Developer(**dev) for dev in self._state["developers"]],
            completed_task_ids=[task["id"] for task in self._state["completed"]],
            sprint_day=self.current_step,
            metrics=self._metrics,
            difficulty=self.difficulty,
            recent_events=[SprintEvent(**event) for event in self._state.get("recent_events", [])],
            pending_events=[SprintEvent(**event) for event in self._state.get("pending_events", [])],
        )

    def _is_done(self) -> bool:
        if len(self._state["tickets"]) == 0 or self.current_step >= self.max_steps:
            return True
        if not self._has_feasible_action() and not self._state.get("pending_events"):
            return True
        return False

    def _has_unresolved_deps(self, task: dict) -> bool:
        completed_ids = {item["id"] for item in self._state["completed"]}
        return any(dep not in completed_ids for dep in task.get("dependencies", []))

    def _has_feasible_action(self) -> bool:
        for task in self._state.get("tickets", []):
            if self._has_unresolved_deps(task):
                continue
            for dev in self._state.get("developers", []):
                if dev["capacity"] >= task["story_points"]:
                    return True
        return False

    def _skill_matches(self, dev: dict, task: dict) -> bool:
        return bool(set(dev.get("specializations", [])) & set(task.get("tags", [])))

    def _apply_scheduled_events(self) -> List[Dict[str, Any]]:
        pending = self._state.get("pending_events", [])
        today = [event for event in pending if event["day"] == self.current_step]
        if not today:
            return []

        self._state["pending_events"] = [event for event in pending if event["day"] != self.current_step]
        applied: List[Dict[str, Any]] = []
        for raw_event in today:
            event = SprintEvent(**raw_event)
            applied.append(self._apply_event(event))

        self._state["recent_events"] = applied
        self._state["event_history"].extend(applied)
        self._metrics.disruptions_applied += len(applied)
        self._state["metrics"] = self._metrics.model_dump()
        return applied

    def _apply_event(self, event: SprintEvent) -> Dict[str, Any]:
        result = event.model_dump(mode="json")
        result["applied"] = True

        if event.type == EventType.ADD_TASK:
            task_payload = event.payload.get("task")
            if isinstance(task_payload, dict):
                extracted_data = task_payload
            else:
                extracted_data = event.payload
            extracted = ExtractedItem(**extracted_data)
            new_ticket = create_tickets([extracted])[0]
            ticket_payload = new_ticket.model_dump()
            ticket_payload["id"] = f"T{self._next_ticket_num:03d}"
            ticket_payload["source_event"] = event.title
            ticket_payload["source_event_type"] = event.type.value
            self._next_ticket_num += 1
            self._state["tickets"].append(ticket_payload)
            self._state["extracted_items"].append(extracted.model_dump())
            self._metrics.disruption_tasks_added += 1
            result["task_id"] = ticket_payload["id"]

        elif event.type == EventType.CAPACITY_CHANGE:
            dev = self._find_developer(event.payload.get("developer_id"))
            if dev is None:
                result["applied"] = False
                result["reason"] = "developer_not_found"
            else:
                initial = self._state["initial_capacities"].get(dev["id"], dev["capacity"])
                if "new_capacity" in event.payload:
                    dev["capacity"] = max(0, min(initial, int(event.payload.get("new_capacity", dev["capacity"]))))
                else:
                    delta = int(event.payload.get("capacity_delta", 0))
                    dev["capacity"] = max(0, min(initial, dev["capacity"] + delta))
                result["new_capacity"] = dev["capacity"]

        elif event.type == EventType.ADD_DEPENDENCY:
            task = self._find_ticket(event.payload.get("task_id")) or self._find_ticket_by_title(event.payload.get("task"))
            depends_on_raw = event.payload.get("depends_on")
            depends_on = self._resolve_task_reference(depends_on_raw)
            if task is None or not depends_on:
                result["applied"] = False
                result["reason"] = "task_or_dependency_missing"
            elif depends_on not in task["dependencies"]:
                task["dependencies"].append(depends_on)

        elif event.type == EventType.REMOVE_DEPENDENCY:
            task = self._find_ticket(event.payload.get("task_id")) or self._find_ticket_by_title(event.payload.get("task"))
            depends_on = self._resolve_task_reference(event.payload.get("depends_on"))
            if task is None or not depends_on:
                result["applied"] = False
                result["reason"] = "task_or_dependency_missing"
            else:
                task["dependencies"] = [dep for dep in task["dependencies"] if dep != depends_on]

        logger.info(
            f"Applied event on day {event.day}: {event.title} | type={event.type.value} | applied={result['applied']}"
        )
        return result

    def _find_ticket(self, task_id: Optional[str]) -> Optional[Dict[str, Any]]:
        if task_id is None:
            return None
        return next((ticket for ticket in self._state.get("tickets", []) if ticket["id"] == task_id), None)

    def _find_ticket_by_title(self, title: Optional[str]) -> Optional[Dict[str, Any]]:
        if title is None:
            return None
        title_lower = str(title).strip().lower()
        return next((ticket for ticket in self._state.get("tickets", []) if ticket["title"].strip().lower() == title_lower), None)

    def _resolve_task_reference(self, ref: Optional[str]) -> Optional[str]:
        if ref is None:
            return None
        if isinstance(ref, str) and ref.startswith("T"):
            return ref
        ticket = self._find_ticket_by_title(ref)
        return ticket["id"] if ticket is not None else None

    def _find_developer(self, developer_id: Optional[str]) -> Optional[Dict[str, Any]]:
        if developer_id is None:
            return None
        return next((dev for dev in self._state.get("developers", []) if dev["id"] == developer_id), None)

    def _count_feasible_tasks(self) -> int:
        count = 0
        for task in self._state.get("tickets", []):
            if self._has_unresolved_deps(task):
                continue
            if any(dev["capacity"] >= task["story_points"] for dev in self._state.get("developers", [])):
                count += 1
        return count

    def _future_dependency_unlock_value(self, task: dict) -> int:
        score = 0
        title_lower = task["title"].strip().lower()
        for raw_event in self._state.get("pending_events", []):
            if raw_event.get("type") != EventType.ADD_DEPENDENCY.value:
                continue
            depends_on = raw_event.get("payload", {}).get("depends_on")
            if depends_on == task["id"] or str(depends_on or "").strip().lower() == title_lower:
                score += 1
        return score

    def _future_capacity_loss_risk(self, developer_id: str) -> float:
        risk = 0.0
        for raw_event in self._state.get("pending_events", []):
            if raw_event.get("type") != EventType.CAPACITY_CHANGE.value:
                continue
            payload = raw_event.get("payload", {})
            if payload.get("developer_id") != developer_id:
                continue
            day_distance = max(int(raw_event.get("day", self.current_step + 1)) - self.current_step, 1)
            if "new_capacity" in payload:
                dev = self._find_developer(developer_id)
                if dev is None:
                    continue
                reduction = max(dev["capacity"] - int(payload.get("new_capacity", dev["capacity"])), 0)
            else:
                reduction = max(-int(payload.get("capacity_delta", 0)), 0)
            if reduction > 0:
                risk = max(risk, min(1.0, reduction / 5.0) * (1.0 / day_distance))
        return risk

    def _preserve_specialist_capacity_penalty(self, task: dict, chosen_dev: dict) -> float:
        chosen_tags = set(chosen_dev.get("specializations", []))
        task_tags = set(task.get("tags", []))
        if chosen_tags & task_tags:
            return 0.0

        for dev in self._state.get("developers", []):
            if dev["id"] == chosen_dev["id"]:
                continue
            if dev["capacity"] < task["story_points"]:
                continue
            if set(dev.get("specializations", [])) & task_tags:
                return 1.0
        return 0.0

    def _log_episode_summary(self) -> None:
        completed = len(self._state["completed"])
        total = completed + len(self._state["tickets"])
        logger.info(
            f"Episode complete | {completed}/{total} tasks | "
            f"violations={self._state.get('deadline_violations', 0)} | "
            f"disruptions={self._metrics.disruptions_applied} | "
            f"reward={self._metrics.total_reward:.2f}"
        )
