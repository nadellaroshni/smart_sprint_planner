"""
Scenario loading utilities.

Priority order:
  1. dataset.json scenarios, when present
  2. built-in fallback scenarios

For stability, deterministic calls use the first scenario for each difficulty.
Training can request sampled scenarios to use the full dataset.
"""

from __future__ import annotations

import json
import random
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

from .models import Developer, Difficulty, EventType, ExtractedItem, SprintEvent

DATASET_GLOB = "dataset*.json"


def _fallback_easy() -> Dict[str, object]:
    return {
        "transcript": (
            "This sprint is intentionally stable. We have a fixed backlog, fixed team "
            "capacity, and no expected mid-sprint surprises."
        ),
        "developers": [
            Developer(id="D1", name="Alice", capacity=9, skill=0.95, specializations=["backend", "auth", "payments"]),
            Developer(id="D2", name="Bob", capacity=8, skill=0.90, specializations=["frontend", "analytics", "ui"]),
            Developer(id="D3", name="Carol", capacity=8, skill=0.88, specializations=["testing", "infra", "ci"]),
        ],
        "items": [
            ExtractedItem(task="Fix checkout form validation bug", deadline=2, priority=3, tags=["bug", "frontend"]),
            ExtractedItem(task="Implement weekly analytics summary export", deadline=5, priority=3, tags=["backend", "analytics"]),
            ExtractedItem(task="Refresh profile page layout", deadline=6, priority=2, tags=["frontend", "ui"]),
            ExtractedItem(task="Add regression tests for auth login flow", deadline=7, priority=2, tags=["testing", "auth"]),
            ExtractedItem(task="Stabilize CI pipeline for pull requests", deadline=6, priority=2, tags=["infra", "ci"]),
        ],
        "events": [],
    }


def _fallback_medium() -> Dict[str, object]:
    return {
        "transcript": (
            "Start with a normal sprint plan using the current backlog and team capacity. "
            "One disruption is expected mid-sprint."
        ),
        "developers": [
            Developer(id="D1", name="Alice", capacity=8, skill=0.92, specializations=["backend", "auth", "database"]),
            Developer(id="D2", name="Bob", capacity=7, skill=0.86, specializations=["frontend", "analytics", "ui"]),
            Developer(id="D3", name="Carol", capacity=7, skill=0.87, specializations=["testing", "infra", "ci"]),
        ],
        "items": [
            ExtractedItem(task="Fix payment retry bug on checkout", deadline=2, priority=4, tags=["bug", "payments", "backend"]),
            ExtractedItem(task="Implement release health dashboard", deadline=5, priority=3, tags=["frontend", "analytics"]),
            ExtractedItem(task="Migrate audit logs to PostgreSQL", deadline=6, priority=3, tags=["backend", "database", "infra"]),
            ExtractedItem(task="Add integration tests for billing flow", deadline=7, priority=2, tags=["testing", "backend"]),
            ExtractedItem(task="Harden GitHub Actions deployment checks", deadline=6, priority=2, tags=["infra", "ci"]),
        ],
        "events": [
            SprintEvent(
                day=3,
                type=EventType.ADD_TASK,
                title="Urgent production bug arrives",
                description="A new P0 issue is reported mid-sprint and must be absorbed into the plan.",
                payload={
                    "task": {
                        "task": "Hotfix invoice generation failure for enterprise accounts",
                        "deadline": 4,
                        "priority": 4,
                        "tags": ["bug", "backend", "payments"],
                        "raw_text": "Urgent bug added during sprint after invoice generation started failing in production.",
                    }
                },
            )
        ],
    }


def _fallback_hard() -> Dict[str, object]:
    return {
        "transcript": (
            "This sprint will not stay stable. Expect repeated replanning because capacity, "
            "dependencies, and incoming work will all change."
        ),
        "developers": [
            Developer(id="D1", name="Alice", capacity=7, skill=0.88, specializations=["backend", "auth", "payments"]),
            Developer(id="D2", name="Bob", capacity=6, skill=0.80, specializations=["frontend", "analytics", "ui"]),
            Developer(id="D3", name="Carol", capacity=6, skill=0.83, specializations=["infra", "database", "devops"]),
            Developer(id="D4", name="Dave", capacity=5, skill=0.78, specializations=["testing", "qa", "ci"]),
        ],
        "items": [
            ExtractedItem(task="Fix token refresh failures in auth gateway", deadline=2, priority=4, tags=["bug", "auth", "backend"]),
            ExtractedItem(task="Implement partner onboarding SSO flow", deadline=4, priority=4, tags=["auth", "feature", "backend"]),
            ExtractedItem(task="Build operations command dashboard", deadline=5, priority=3, tags=["frontend", "analytics"]),
            ExtractedItem(task="Complete PostgreSQL migration for tenant data", deadline=4, priority=4, tags=["backend", "database", "infra"]),
            ExtractedItem(task="Improve mobile layout for customer portal", deadline=6, priority=2, tags=["frontend", "ui"]),
            ExtractedItem(task="Raise auth regression coverage to 80 percent", deadline=6, priority=3, tags=["testing", "auth"]),
            ExtractedItem(task="Stabilize deployment pipeline rollback checks", deadline=5, priority=3, tags=["infra", "ci"]),
        ],
        "events": [
            SprintEvent(day=2, type=EventType.CAPACITY_CHANGE, title="Alice becomes partially unavailable", description="Backend capacity drops.", payload={"developer_id": "D1", "capacity_delta": -3}),
            SprintEvent(day=3, type=EventType.ADD_TASK, title="Urgent security hotfix added", description="A new exploit forces immediate work.", payload={"task": {"task": "Patch privilege escalation in admin API", "deadline": 4, "priority": 4, "tags": ["bug", "backend", "auth"], "raw_text": "Security escalation bug added mid-sprint."}}),
            SprintEvent(day=4, type=EventType.ADD_DEPENDENCY, title="Dashboard now depends on SSO completion", description="Product clarifies a new blocker.", payload={"task_id": "T003", "depends_on": "T002"}),
            SprintEvent(day=5, type=EventType.CAPACITY_CHANGE, title="QA bandwidth shrinks", description="Shared testing support is reduced.", payload={"developer_id": "D4", "capacity_delta": -2}),
        ],
    }


_FALLBACKS = {
    Difficulty.EASY: _fallback_easy,
    Difficulty.MEDIUM: _fallback_medium,
    Difficulty.HARD: _fallback_hard,
}


@lru_cache(maxsize=1)
def _load_dataset() -> list[dict]:
    scenarios: list[dict] = []
    for path in sorted(Path(".").glob(DATASET_GLOB)):
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            scenarios.extend(data)
    return scenarios


def _normalize_scenario(raw: dict) -> Dict[str, object]:
    return {
        "scenario_id": raw.get("scenario_id", ""),
        "transcript": raw["transcript"],
        "developers": [Developer(**dev) for dev in raw.get("developers", [])],
        "items": [ExtractedItem(**item) for item in raw.get("items", [])],
        "events": [SprintEvent(**event) for event in raw.get("events", [])],
    }


def _dataset_scenarios_for(difficulty: Difficulty) -> list[Dict[str, object]]:
    scenarios = []
    for raw in _load_dataset():
        if str(raw.get("difficulty", "")).lower() == difficulty.value:
            try:
                scenarios.append(_normalize_scenario(raw))
            except Exception:
                continue
    return scenarios


def _split_dataset_scenarios(
    scenarios: List[Dict[str, object]],
    split: Optional[str],
) -> List[Dict[str, object]]:
    if not scenarios or split in (None, "all"):
        return scenarios

    ordered = sorted(
        scenarios,
        key=lambda scenario: str(scenario.get("scenario_id", "")) or str(scenario.get("transcript", ""))[:40],
    )
    if len(ordered) == 1:
        return ordered

    eval_count = max(1, len(ordered) // 5)
    train_count = max(1, len(ordered) - eval_count)
    if train_count == len(ordered):
        train_count = len(ordered) - 1

    train_set = ordered[:train_count]
    eval_set = ordered[train_count:]

    if split == "train":
        return train_set
    if split == "eval":
        return eval_set or train_set[-1:]
    return ordered


def get_scenario(
    difficulty: Difficulty,
    *,
    sample: bool = False,
    rng: Optional[random.Random] = None,
    scenario_index: Optional[int] = None,
    split: Optional[str] = None,
) -> Dict[str, object]:
    dataset_scenarios = _split_dataset_scenarios(_dataset_scenarios_for(difficulty), split)
    if dataset_scenarios and (sample or scenario_index is not None):
        if scenario_index is not None:
            scenario = dataset_scenarios[scenario_index % len(dataset_scenarios)]
        elif sample:
            chooser = rng or random
            scenario = chooser.choice(dataset_scenarios)
        return {
            "scenario_id": scenario.get("scenario_id", ""),
            "transcript": scenario["transcript"],
            "developers": deepcopy(scenario["developers"]),
            "items": deepcopy(scenario["items"]),
            "events": deepcopy(scenario["events"]),
        }

    fallback = _FALLBACKS[difficulty]()
    return {
        "scenario_id": f"fallback-{difficulty.value}",
        "transcript": fallback["transcript"],
        "developers": deepcopy(fallback["developers"]),
        "items": deepcopy(fallback["items"]),
        "events": deepcopy(fallback["events"]),
    }


def get_developers(difficulty: Difficulty, **kwargs) -> List[Developer]:
    return get_scenario(difficulty, **kwargs)["developers"]  # type: ignore[return-value]


def get_transcript(difficulty: Difficulty, **kwargs) -> str:
    return get_scenario(difficulty, **kwargs)["transcript"]  # type: ignore[return-value]


def get_extracted_items(difficulty: Difficulty, **kwargs) -> List[ExtractedItem]:
    return get_scenario(difficulty, **kwargs)["items"]  # type: ignore[return-value]


def get_events(difficulty: Difficulty, **kwargs) -> List[SprintEvent]:
    return get_scenario(difficulty, **kwargs)["events"]  # type: ignore[return-value]


def dataset_available() -> bool:
    return bool(_load_dataset())


def get_scenario_count(difficulty: Difficulty, split: Optional[str] = None) -> int:
    scenarios = _split_dataset_scenarios(_dataset_scenarios_for(difficulty), split)
    return len(scenarios) if scenarios else 1
