"""
Competition-compliant baseline inference runner.

Stdout contract:
  [START] task=<task_name> env=<benchmark> model=<model_name>
  [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

Notes:
  - Uses OpenAI client for LLM planning when credentials are configured.
  - Falls back to a deterministic heuristic when no API key is available.
  - Prints only the mandated log lines to stdout. Optional render/debug goes to stderr.
"""

from __future__ import annotations

import json
import os
import sys
from typing import List, Optional, Tuple

from openai import OpenAI

from env.environment import SprintEnv
from env.graders import grade
from env.models import Action, Difficulty, Observation

ENV_NAME = "smart_sprint_env"
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")

_client: OpenAI | str | None = None


def _get_client() -> OpenAI | str:
    global _client
    if _client is None:
        if not API_KEY:
            _client = "unavailable"
        else:
            _client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    return _client


def _priority_score(task: dict, current_day: int) -> float:
    days_remaining = max(task["deadline"] - current_day, 0)
    urgency = 10.0 / (days_remaining + 1)
    priority_weight = task["priority"] * 1.5
    size_penalty = 1.0 / max(task["story_points"], 1)
    event_bonus = 2.0 if task.get("source_event") else 0.0
    return urgency + priority_weight + size_penalty + event_bonus


def _developer_score(dev: dict, task: dict) -> float:
    if dev["capacity"] < task["story_points"]:
        return -999.0
    skill_bonus = 2.0 if set(dev.get("specializations", [])) & set(task.get("tags", [])) else 0.0
    capacity_score = dev["capacity"] * dev["skill"]
    return capacity_score + skill_bonus


def choose_action_heuristic(obs: Observation) -> Optional[Tuple[str, str]]:
    completed_ids = set(obs.completed_task_ids)
    tickets = obs.jira_tickets
    developers = obs.developers

    if not tickets or not developers:
        return None

    candidates = sorted(
        tickets,
        key=lambda t: (
            0 if t.model_dump().get("source_event") else 1,
            len(t.dependencies),
            -_priority_score(t.model_dump(), obs.sprint_day),
        ),
    )

    for task in candidates:
        if any(dep not in completed_ids for dep in task.dependencies):
            continue

        ranked_devs = sorted(
            developers,
            key=lambda d: _developer_score(d.model_dump(), task.model_dump()),
            reverse=True,
        )
        best_dev = ranked_devs[0]
        if best_dev.capacity >= task.story_points:
            return task.id, best_dev.id

    return None


def choose_action_llm(obs: Observation) -> Optional[Tuple[str, str]]:
    client = _get_client()
    if client == "unavailable":
        return None

    completed_ids = set(obs.completed_task_ids)
    candidate_actions = []
    for task in obs.jira_tickets:
        if any(dep not in completed_ids for dep in task.dependencies):
            continue
        for dev in obs.developers:
            if dev.capacity >= task.story_points:
                candidate_actions.append(
                    {
                        "task_id": task.id,
                        "developer_id": dev.id,
                        "task_title": task.title,
                        "story_points": task.story_points,
                        "priority": int(task.priority),
                        "deadline": task.deadline,
                        "tags": task.tags,
                        "developer_capacity": dev.capacity,
                        "developer_skill": dev.skill,
                        "developer_specializations": dev.specializations,
                    }
                )

    if not candidate_actions:
        return None

    payload = {
        "difficulty": obs.difficulty.value,
        "sprint_day": obs.sprint_day,
        "recent_events": [event.model_dump(mode="json") for event in obs.recent_events],
        "candidate_actions": candidate_actions[:40],
        "instructions": [
            "Choose the single best action for sprint planning.",
            "Prioritize urgent and high-value work.",
            "Absorb disruption-created work quickly.",
            "Respect developer capacity and specialization fit.",
            "Return strict JSON with keys task_id and developer_id.",
        ],
    }

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=0.0,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are choosing one sprint assignment action. "
                        "Return only JSON: {\"task_id\":\"...\",\"developer_id\":\"...\"}."
                    ),
                },
                {"role": "user", "content": json.dumps(payload)},
            ],
        )
        raw = response.choices[0].message.content.strip()
        data = json.loads(raw)
        task_id = data.get("task_id")
        developer_id = data.get("developer_id")
        if not isinstance(task_id, str) or not isinstance(developer_id, str):
            return None
        valid_pairs = {(a["task_id"], a["developer_id"]) for a in candidate_actions}
        return (task_id, developer_id) if (task_id, developer_id) in valid_pairs else None
    except Exception:
        return None


def choose_action(obs: Observation) -> tuple[Optional[Tuple[str, str]], str]:
    llm_choice = choose_action_llm(obs)
    if llm_choice is not None:
        return llm_choice, f"assign(task_id='{llm_choice[0]}',developer_id='{llm_choice[1]}',source='llm')"

    heuristic_choice = choose_action_heuristic(obs)
    if heuristic_choice is not None:
        return heuristic_choice, f"assign(task_id='{heuristic_choice[0]}',developer_id='{heuristic_choice[1]}',source='heuristic')"

    return None, "noop()"


def _bool(value: bool) -> str:
    return "true" if value else "false"


def _format_error(info: dict) -> str:
    error = info.get("error")
    if error is None:
        return "null"
    return str(error).replace("\n", " ").strip() or "null"


def log_start(task_name: str) -> None:
    print(f"[START] task={task_name} env={ENV_NAME} model={MODEL_NAME}")


def log_step(step: int, action_str: str, reward: float, done: bool, info: dict) -> None:
    print(
        f"[STEP] step={step} action={action_str} reward={reward:.2f} "
        f"done={_bool(done)} error={_format_error(info)}"
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    reward_str = ",".join(f"{reward:.2f}" for reward in rewards)
    print(f"[END] success={_bool(success)} steps={steps} score={score:.2f} rewards={reward_str}")


def run_episode(difficulty: Difficulty = Difficulty.MEDIUM, render: bool = False) -> dict:
    env = SprintEnv(difficulty=difficulty, max_steps=20, use_llm=False)
    obs = env.reset()
    rewards: List[float] = []
    done = False

    log_start(difficulty.value)

    if render:
        print(env.render(), file=sys.stderr)

    while not done:
        choice, action_str = choose_action(obs)
        if choice is None:
            action = Action(task_id="INVALID_TASK", developer_id="INVALID_DEV")
            obs, reward, done, info = env.step(action)
            rewards.append(reward)
            log_step(env.current_step, action_str, reward, done, info)
            continue

        action = Action(task_id=choice[0], developer_id=choice[1])
        obs, reward, done, info = env.step(action)
        rewards.append(reward)
        log_step(env.current_step, action_str, reward, done, info)

        if render:
            print(env.render(), file=sys.stderr)

    result = grade(env)
    success = result["score"] > 0.0
    log_end(success, env.current_step, result["score"], rewards)
    return result


def run_all_difficulties(render: bool = False) -> dict:
    results = {}
    for difficulty in Difficulty:
        results[difficulty.value] = run_episode(difficulty=difficulty, render=render)
    return results


if __name__ == "__main__":
    difficulty_arg = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else "medium"
    render_arg = "--render" in sys.argv
    all_arg = "--all" in sys.argv

    diff_map = {
        "easy": Difficulty.EASY,
        "medium": Difficulty.MEDIUM,
        "hard": Difficulty.HARD,
    }

    if all_arg:
        run_all_difficulties(render=render_arg)
    else:
        run_episode(difficulty=diff_map.get(difficulty_arg.lower(), Difficulty.MEDIUM), render=render_arg)
