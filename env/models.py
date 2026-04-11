"""
Typed data models for the Smart Sprint Planner RL Environment.
All inter-component data contracts are enforced here via Pydantic.
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Priority(int, Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class TaskStatus(str, Enum):
    BACKLOG = "backlog"
    IN_PROGRESS = "in_progress"
    DONE = "done"
    BLOCKED = "blocked"


class Difficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class EventType(str, Enum):
    ADD_TASK = "add_task"
    CAPACITY_CHANGE = "capacity_change"
    ADD_DEPENDENCY = "add_dependency"
    REMOVE_DEPENDENCY = "remove_dependency"
    NOTE = "note"


# ---------------------------------------------------------------------------
# Core domain models
# ---------------------------------------------------------------------------

class Task(BaseModel):
    id: str
    title: str
    description: str = ""
    category: str = "general"
    story_points: int = Field(ge=1, le=13)
    deadline: int = Field(ge=1, description="Sprint day by which task must be done")
    priority: Priority = Priority.MEDIUM
    status: TaskStatus = TaskStatus.BACKLOG
    dependencies: List[str] = Field(default_factory=list, description="List of Task IDs this task depends on")
    tags: List[str] = Field(default_factory=list)
    acceptance_criteria: List[str] = Field(default_factory=list)
    owner_hint: Optional[str] = None
    urgency_reason: str = ""
    assigned_to: Optional[str] = None
    source_event: Optional[str] = None
    source_event_type: Optional[str] = None


class Developer(BaseModel):
    id: str
    name: str
    capacity: int = Field(ge=0, description="Remaining story point capacity for the sprint")
    skill: float = Field(ge=0.0, le=1.0, description="General skill multiplier")
    specializations: List[str] = Field(default_factory=list, description="e.g. ['backend', 'auth', 'payments']")
    active_tasks: List[str] = Field(default_factory=list)


class ExtractedItem(BaseModel):
    """Raw item extracted from meeting transcript before JIRA enrichment."""
    task: str
    description: str = ""
    deadline: int = 3
    priority: int = 2
    category: str = "general"
    tags: List[str] = Field(default_factory=list)
    acceptance_criteria: List[str] = Field(default_factory=list)
    dependency_hints: List[str] = Field(default_factory=list)
    owner_hint: Optional[str] = None
    urgency_reason: str = ""
    raw_text: str = ""


class SprintEvent(BaseModel):
    day: int = Field(ge=1, description="Sprint day when the event becomes active")
    type: EventType
    title: str
    description: str = ""
    payload: Dict[str, Any] = Field(default_factory=dict)


class SprintMetrics(BaseModel):
    """Accumulated metrics for reward computation."""
    tasks_completed: int = 0
    tasks_failed_deadline: int = 0
    tasks_blocked: int = 0
    total_reward: float = 0.0
    capacity_utilization: float = 0.0
    on_time_delivery_rate: float = 0.0
    disruptions_applied: int = 0
    disruption_tasks_added: int = 0
    disruption_tasks_completed: int = 0
    recovery_actions: int = 0


# ---------------------------------------------------------------------------
# RL Environment interfaces
# ---------------------------------------------------------------------------

class Observation(BaseModel):
    meeting_text: str
    extracted_items: List[ExtractedItem]
    jira_tickets: List[Task]
    developers: List[Developer]
    completed_task_ids: List[str] = Field(default_factory=list)
    sprint_day: int
    metrics: SprintMetrics = Field(default_factory=SprintMetrics)
    difficulty: Difficulty = Difficulty.MEDIUM
    recent_events: List[SprintEvent] = Field(default_factory=list)
    pending_events: List[SprintEvent] = Field(default_factory=list)

    @property
    def backlog_count(self) -> int:
        return sum(1 for t in self.jira_tickets if t.status == TaskStatus.BACKLOG)

    @property
    def total_remaining_points(self) -> int:
        return sum(t.story_points for t in self.jira_tickets if t.status != TaskStatus.DONE)


class Action(BaseModel):
    task_id: str
    developer_id: str
    notes: Optional[str] = None


class StepResult(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# API request/response schemas
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    difficulty: Difficulty = Difficulty.MEDIUM
    task: Optional[str] = None
    audio_path: Optional[str] = None
    transcript_override: Optional[str] = None


class StepRequest(BaseModel):
    task_id: str
    developer_id: str
    notes: Optional[str] = None


class GradeResponse(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    breakdown: Dict[str, float]
    summary: str


class PlanRequest(BaseModel):
    difficulty: Difficulty = Difficulty.MEDIUM
    audio_path: Optional[str] = None
    transcript: Optional[str] = None
    strategy: str = "auto"
    checkpoint: Optional[str] = None


class AssignmentRecommendation(BaseModel):
    step: int
    task_id: str
    task_title: str
    developer_id: str
    developer_name: str
    reward: float
    on_time: bool = True
    skill_match: bool = False
    source_event: Optional[str] = None


class PlanResponse(BaseModel):
    strategy: str
    transcript: str
    extracted_items: List[ExtractedItem]
    jira_tickets: List[Task]
    assignments: List[AssignmentRecommendation]
    score: float = Field(ge=0.0, le=1.0)
    breakdown: Dict[str, float]
    summary: str
    final_board: str


class TaskDescriptor(BaseModel):
    id: str
    name: str
    difficulty: Difficulty
    objective: str
    grader: str
    score_range: str = "(0.0, 1.0)"
    deterministic: bool = True
