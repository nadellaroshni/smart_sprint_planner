"""
Inference script — Heuristic Sprint Planning Agent.

Agent strategy:
  1. Prioritise by: deadline urgency > task priority > story points
  2. Assign to developer with best skill match + highest capacity
  3. Respect dependency ordering (blocked tasks skipped)

Logging follows the exact openenv format required by the hackathon.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from datetime import datetime
from typing import Optional, Tuple

from env.environment import SprintEnv
from env.models import Action, Difficulty, Observation
from env.graders import grade

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("inference")


# ---------------------------------------------------------------------------
# Heuristic agent
# ---------------------------------------------------------------------------

def _priority_score(task: dict, current_day: int) -> float:
    """
    Combined urgency score (higher = assign first).
    Accounts for:
      - deadline proximity (days remaining)
      - task priority level
      - story points (prefer smaller tasks to clear backlog faster)
    """
    days_remaining = max(task["deadline"] - current_day, 0)
    urgency = 10.0 / (days_remaining + 1)          # higher when deadline is close
    priority_weight = task["priority"] * 1.5
    size_penalty = 1.0 / task["story_points"]       # prefer smaller tasks

    return urgency + priority_weight + size_penalty


def _developer_score(dev: dict, task: dict) -> float:
    """Score a developer-task pair. Higher = better fit."""
    if dev["capacity"] < task["story_points"]:
        return -999.0  # cannot assign

    skill_bonus = 2.0 if set(dev.get("specializations", [])) & set(task.get("tags", [])) else 0.0
    capacity_score = dev["capacity"] * dev["skill"]
    return capacity_score + skill_bonus


def choose_action(obs: Observation) -> Optional[Tuple[str, str]]:
    """
    Pick the best (task_id, developer_id) action given current observation.

    Returns None if no valid assignment exists.
    """
    completed_ids = set()  # We don't track this in obs directly but env handles it
    tickets = obs.jira_tickets
    developers = obs.developers

    if not tickets or not developers:
        return None

    # Exclude blocked tasks (dependencies not met)
    # We rely on env to reject blocked assignments; still try to avoid them
    # by sorting. Tickets with no dependencies go first.
    candidates = sorted(
        tickets,
        key=lambda t: (
            len(t.dependencies),                         # fewer deps first
            -_priority_score(t.model_dump(), obs.sprint_day),
        )
    )

    for task in candidates:
        # Sort devs by fit score for this task
        ranked_devs = sorted(
            developers,
            key=lambda d: _developer_score(d.model_dump(), task.model_dump()),
            reverse=True,
        )
        best_dev = ranked_devs[0]

        if best_dev.capacity >= task.story_points:
            return task.id, best_dev.id

    return None  # all tasks blocked or no capacity


# ---------------------------------------------------------------------------
# Logging helpers (strict openenv format)
# ---------------------------------------------------------------------------

def log_start(env: SprintEnv, difficulty: str, episode: int = 1) -> None:
    entry = {
        "event": "episode_start",
        "timestamp": datetime.utcnow().isoformat(),
        "episode": episode,
        "difficulty": difficulty,
        "num_tickets": len(env.state()["tickets"]),
        "num_developers": len(env.state()["developers"]),
        "max_steps": env.max_steps,
    }
    print(json.dumps(entry))


def log_step(step: int, action: Action, obs: Observation, reward: float, done: bool, info: dict) -> None:
    entry = {
        "event": "step",
        "step": step,
        "action": {"task_id": action.task_id, "developer_id": action.developer_id},
        "reward": round(reward, 4),
        "done": done,
        "backlog_remaining": obs.backlog_count,
        "info": info,
    }
    print(json.dumps(entry))


def log_end(grade_result: dict, total_steps: int, elapsed_s: float) -> None:
    entry = {
        "event": "episode_end",
        "timestamp": datetime.utcnow().isoformat(),
        "total_steps": total_steps,
        "elapsed_seconds": round(elapsed_s, 2),
        **grade_result,
    }
    print(json.dumps(entry))


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_episode(
    difficulty: Difficulty = Difficulty.MEDIUM,
    render: bool = True,
    episode: int = 1,
) -> dict:
    env = SprintEnv(difficulty=difficulty, max_steps=20, use_llm=False)
    obs = env.reset()

    start_time = time.time()
    log_start(env, difficulty.value, episode)

    if render:
        print(env.render())

    cumulative_reward = 0.0
    done = False

    while not done:
        result = choose_action(obs)

        if result is None:
            logger.warning("No valid action found — ending episode early.")
            break

        task_id, dev_id = result
        action = Action(task_id=task_id, developer_id=dev_id)

        obs, reward, done, info = env.step(action)
        cumulative_reward += reward

        log_step(env.current_step, action, obs, reward, done, info)

        if render:
            print(env.render())

    grade_result = grade(env)
    elapsed = time.time() - start_time
    log_end(grade_result, env.current_step, elapsed)

    logger.info(
        f"Episode {episode} | difficulty={difficulty.value} | "
        f"reward={cumulative_reward:.3f} | score={grade_result['score']:.3f} | "
        f"steps={env.current_step} | elapsed={elapsed:.1f}s"
    )
    return grade_result


def run_all_difficulties(render: bool = False) -> None:
    """Run one episode per difficulty level and print a summary table."""
    results = {}
    for i, diff in enumerate(Difficulty, start=1):
        logger.info(f"\n{'─' * 50}\nRunning difficulty: {diff.value}\n{'─' * 50}")
        results[diff.value] = run_episode(difficulty=diff, render=render, episode=i)

    print("\n" + "=" * 50)
    print("  FINAL RESULTS ACROSS DIFFICULTIES")
    print("=" * 50)
    for diff, r in results.items():
        print(f"  {diff:8s}  score={r['score']:.3f}  {r['summary']}")
    print("=" * 50)


if __name__ == "__main__":
    difficulty_arg = sys.argv[1] if len(sys.argv) > 1 else "medium"
    render_arg = "--render" in sys.argv
    all_arg = "--all" in sys.argv

    if all_arg:
        run_all_difficulties(render=render_arg)
    else:
        diff_map = {
            "easy": Difficulty.EASY,
            "medium": Difficulty.MEDIUM,
            "hard": Difficulty.HARD,
        }
        chosen = diff_map.get(difficulty_arg.lower(), Difficulty.MEDIUM)
        run_episode(difficulty=chosen, render=render_arg)