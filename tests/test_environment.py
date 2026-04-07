"""
Unit + integration tests for SprintEnv.

Run: pytest tests/ -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from env.environment import SprintEnv
from env.models import Action, Difficulty, TaskStatus


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def env_easy():
    e = SprintEnv(difficulty=Difficulty.EASY, max_steps=20, use_llm=False)
    e.reset()
    return e


@pytest.fixture
def env_medium():
    e = SprintEnv(difficulty=Difficulty.MEDIUM, max_steps=20, use_llm=False)
    e.reset()
    return e


@pytest.fixture
def env_hard():
    e = SprintEnv(difficulty=Difficulty.HARD, max_steps=20, use_llm=False)
    e.reset()
    return e


# ---------------------------------------------------------------------------
# Reset tests
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_returns_observation(self, env_easy):
        obs = env_easy.reset()
        assert obs is not None
        assert obs.sprint_day == 0

    def test_reset_clears_state(self, env_easy):
        env_easy.step(Action(task_id="T001", developer_id="D1"))
        obs = env_easy.reset()
        assert obs.sprint_day == 0
        assert len(obs.jira_tickets) > 0

    def test_reset_with_difficulty_change(self):
        env = SprintEnv(use_llm=False)
        obs = env.reset(difficulty=Difficulty.HARD)
        assert env.difficulty == Difficulty.HARD
        assert len(obs.jira_tickets) > 5  # hard has more tasks
        assert len(env.state()["pending_events"]) >= 3

    def test_reset_with_transcript_override(self):
        env = SprintEnv(use_llm=False)
        obs = env.reset(transcript_override="Fix the login bug urgently.")
        assert "login" in obs.meeting_text.lower() or obs.meeting_text != ""

    def test_tickets_have_required_fields(self, env_medium):
        obs = env_medium.reset()
        for ticket in obs.jira_tickets:
            assert ticket.id.startswith("T")
            assert ticket.story_points >= 1
            assert ticket.deadline >= 1
            assert ticket.priority >= 1


# ---------------------------------------------------------------------------
# Step tests
# ---------------------------------------------------------------------------

class TestStep:
    def test_valid_action_returns_positive_reward(self, env_easy):
        obs = env_easy.reset()
        task = obs.jira_tickets[0]
        # Find a dev with enough capacity
        dev = next(d for d in obs.developers if d.capacity >= task.story_points)
        _, reward, _, _ = env_easy.step(Action(task_id=task.id, developer_id=dev.id))
        assert reward > -1.0  # not a catastrophic failure

    def test_invalid_task_id_penalised(self, env_easy):
        env_easy.reset()
        _, reward, _, info = env_easy.step(Action(task_id="T999", developer_id="D1"))
        assert reward < 0
        assert "invalid" in info.get("error", "").lower()

    def test_invalid_dev_id_penalised(self, env_easy):
        obs = env_easy.reset()
        task = obs.jira_tickets[0]
        _, reward, _, info = env_easy.step(Action(task_id=task.id, developer_id="D999"))
        assert reward < 0

    def test_over_capacity_penalised(self):
        env = SprintEnv(use_llm=False)
        env.reset(difficulty=Difficulty.HARD)
        # Force a big task onto an underpowered dev
        state = env.state()
        # Find a dev with low capacity
        dev = min(state["developers"], key=lambda d: d["capacity"])
        # Find a task larger than dev capacity
        big_tasks = [t for t in state["tickets"] if t["story_points"] > dev["capacity"]]
        if big_tasks:
            _, reward, _, info = env.step(Action(task_id=big_tasks[0]["id"], developer_id=dev["id"]))
            assert reward < 0

    def test_step_increments_day(self, env_easy):
        obs = env_easy.reset()
        assert obs.sprint_day == 0
        task = obs.jira_tickets[0]
        dev = next(d for d in obs.developers if d.capacity >= task.story_points)
        obs2, _, _, _ = env_easy.step(Action(task_id=task.id, developer_id=dev.id))
        assert obs2.sprint_day == 1

    def test_completed_task_removed_from_backlog(self, env_easy):
        obs = env_easy.reset()
        task = obs.jira_tickets[0]
        dev = next(d for d in obs.developers if d.capacity >= task.story_points)
        obs2, _, _, _ = env_easy.step(Action(task_id=task.id, developer_id=dev.id))
        ticket_ids = [t.id for t in obs2.jira_tickets]
        assert task.id not in ticket_ids

    def test_done_when_all_tasks_completed(self, env_easy):
        obs = env_easy.reset()
        done = False
        steps = 0
        while not done and steps < 50:
            if not obs.jira_tickets:
                break
            task = obs.jira_tickets[0]
            dev = next(
                (d for d in obs.developers if d.capacity >= task.story_points),
                obs.developers[0]
            )
            obs, _, done, _ = env_easy.step(Action(task_id=task.id, developer_id=dev.id))
            steps += 1
        assert done or steps >= 50  # Either done naturally or hit max

    def test_medium_triggers_single_event(self, env_medium):
        obs = env_medium.reset()
        initial_pending = len(env_medium.state()["pending_events"])
        assert initial_pending == 1

        for _ in range(3):
            task = obs.jira_tickets[0]
            dev = next((d for d in obs.developers if d.capacity >= task.story_points), obs.developers[0])
            obs, _, _, info = env_medium.step(Action(task_id=task.id, developer_id=dev.id))

        assert len(obs.recent_events) == 1
        assert info["events"][0]["type"] == "add_task"
        assert len(env_medium.state()["pending_events"]) == 0

    def test_hard_triggers_multiple_events_over_time(self, env_hard):
        obs = env_hard.reset()
        seen = 0

        for _ in range(6):
            task = obs.jira_tickets[0]
            dev = next((d for d in obs.developers if d.capacity >= task.story_points), obs.developers[0])
            obs, _, done, info = env_hard.step(Action(task_id=task.id, developer_id=dev.id))
            seen += len(info.get("events", []))
            if done:
                break

        assert seen >= 3
        assert len(env_hard.state()["event_history"]) >= 3


# ---------------------------------------------------------------------------
# Reward tests
# ---------------------------------------------------------------------------

class TestRewards:
    def test_on_time_gives_higher_reward_than_late(self, env_medium):
        from env.graders import compute_step_reward
        on_time_r, _ = compute_step_reward(True, True, True, False, True, True, True)
        late_r, _ = compute_step_reward(True, True, True, False, False, True, True)
        assert on_time_r > late_r

    def test_skill_match_bonus(self):
        from env.graders import compute_step_reward
        with_skill, _ = compute_step_reward(True, True, True, False, True, True, False)
        without_skill, _ = compute_step_reward(True, True, True, False, True, False, False)
        assert with_skill > without_skill

    def test_blocked_task_gives_negative_reward(self):
        from env.graders import compute_step_reward
        r, _ = compute_step_reward(True, True, True, True, False, False, False)
        assert r < 0


# ---------------------------------------------------------------------------
# Grading tests
# ---------------------------------------------------------------------------

class TestGrading:
    def test_grade_returns_score_in_range(self, env_medium):
        from env.graders import grade
        env_medium.reset()
        result = grade(env_medium)
        assert 0.0 <= result["score"] <= 1.0

    def test_grade_has_all_keys(self, env_medium):
        from env.graders import grade
        env_medium.reset()
        result = grade(env_medium)
        assert "score" in result
        assert "breakdown" in result
        assert "summary" in result
        for key in ["completion_rate", "on_time_rate", "extraction_quality",
                    "workload_balance", "efficiency"]:
            assert key in result["breakdown"]

    def test_full_completion_improves_score(self):
        from env.graders import grade
        env = SprintEnv(difficulty=Difficulty.EASY, max_steps=20, use_llm=False)
        obs = env.reset()
        # Complete all tasks
        done = False
        while not done:
            if not obs.jira_tickets:
                break
            task = obs.jira_tickets[0]
            dev = next(
                (d for d in obs.developers if d.capacity >= task.story_points),
                obs.developers[0]
            )
            obs, _, done, _ = env.step(Action(task_id=task.id, developer_id=dev.id))
        result = grade(env)
        assert result["breakdown"]["completion_rate"] == 1.0


# ---------------------------------------------------------------------------
# Determinism test
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_two_resets_produce_same_tickets(self):
        env = SprintEnv(difficulty=Difficulty.MEDIUM, use_llm=False)
        obs1 = env.reset()
        ids1 = sorted(t.id for t in obs1.jira_tickets)
        obs2 = env.reset()
        ids2 = sorted(t.id for t in obs2.jira_tickets)
        assert ids1 == ids2

    def test_same_actions_produce_same_rewards(self):
        results = []
        for _ in range(2):
            env = SprintEnv(difficulty=Difficulty.EASY, use_llm=False)
            obs = env.reset()
            task = obs.jira_tickets[0]
            dev = next(d for d in obs.developers if d.capacity >= task.story_points)
            _, reward, _, _ = env.step(Action(task_id=task.id, developer_id=dev.id))
            results.append(reward)
        assert results[0] == results[1]
