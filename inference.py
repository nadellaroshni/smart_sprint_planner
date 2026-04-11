#!/usr/bin/env python3
"""
Inference Script - Smart Sprint Planner OpenEnv
===============================================

MANDATORY ENV VARS:
    API_BASE_URL   The API endpoint for the LLM
    MODEL_NAME     The model identifier to use for inference
    HF_TOKEN       Your Hugging Face / API key

STDOUT FORMAT:
    [START] task=<task> env=<benchmark> model=<model>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<null|msg>
    [END]   success=<true|false> steps=<n> score=<0.00> rewards=<r1,r2,...>
"""

from __future__ import annotations

import json
import os
import re
import textwrap
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

from env.environment import SprintEnv
from env.models import Action, Difficulty, Observation

API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"

BENCHMARK = "smart_sprint_planner"
TEMPERATURE = 0.1
MAX_COMPLETION_TOKENS = 500
SUCCESS_SCORE_THRESHOLD = 0.3
MAX_CANDIDATES = 6

TASKS = {
    "easy": {"max_steps": 10, "difficulty": Difficulty.EASY},
    "medium": {"max_steps": 15, "difficulty": Difficulty.MEDIUM},
    "hard": {"max_steps": 20, "difficulty": Difficulty.HARD},
}

STRATEGIES = {
    "easy": (
        "Finish urgent backlog items first, preserve on-time delivery, and match developers to their strongest skills."
    ),
    "medium": (
        "Handle the first disruption quickly, absorb urgent event work, and avoid assignments that leave the board infeasible."
    ),
    "hard": (
        "Continuously re-plan for capacity loss, dependency changes, and urgent work while protecting overall completion."
    ),
}

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an engineering manager assigning sprint work.

    Return EXACTLY ONE JSON object with this shape:
    {"action_type":"task_assignment","task_id":"<id>","developer_id":"<id>"}

    Rules:
    - Choose only from VALID_ASSIGNMENTS when any are available.
    - Prefer urgent, high-priority, event-created, and skill-matched tasks.
    - Avoid blocked or infeasible assignments.
    - If uncertain, return the recommended assignment exactly.
    """
).strip()


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={error if error else 'null'}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{reward:.2f}" for reward in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


def parse_action_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    depth = 0
    start = None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    start = None

    raise ValueError(f"Could not parse JSON from: {text[:200]}")


def _valid_pairs(obs: Observation) -> List[Tuple[str, str]]:
    completed_ids = set(obs.completed_task_ids)
    pairs: List[Tuple[str, str]] = []
    for task in obs.jira_tickets:
        if any(dep not in completed_ids for dep in task.dependencies):
            continue
        for dev in obs.developers:
            if dev.capacity >= task.story_points:
                pairs.append((task.id, dev.id))
    return pairs


def _score_pair(obs: Observation, task, dev) -> float:
    days_left = max(task.deadline - obs.sprint_day, 0)
    urgency = 10.0 / (days_left + 1)
    priority = int(task.priority) * 2.0
    size_bonus = 1.0 / max(task.story_points, 1)
    skill_bonus = 2.5 if set(task.tags) & set(dev.specializations) else 0.0
    event_bonus = 1.5 if task.source_event else 0.0
    return urgency + priority + size_bonus + skill_bonus + event_bonus + (dev.capacity * 0.05)


def candidate_actions(obs: Observation, limit: int = MAX_CANDIDATES) -> List[Tuple[str, str]]:
    task_map = {task.id: task for task in obs.jira_tickets}
    dev_map = {dev.id: dev for dev in obs.developers}
    ranked = [
        (_score_pair(obs, task_map[task_id], dev_map[dev_id]), (task_id, dev_id))
        for task_id, dev_id in _valid_pairs(obs)
    ]
    ranked.sort(key=lambda item: item[0], reverse=True)
    return [pair for _, pair in ranked[:limit]]


def build_prompt(obs: Observation, task_id: str) -> Tuple[str, Optional[Tuple[str, str]]]:
    candidates = candidate_actions(obs)
    recommended = candidates[0] if candidates else None

    task_lines = []
    completed_ids = set(obs.completed_task_ids)
    for task in obs.jira_tickets:
        blocked = any(dep not in completed_ids for dep in task.dependencies)
        task_lines.append(
            f"{task.id}: title={task.title}; priority={int(task.priority)}; points={task.story_points}; "
            f"deadline={task.deadline}; blocked={'yes' if blocked else 'no'}; "
            f"tags={','.join(task.tags) or 'none'}; event={task.source_event or 'none'}"
        )

    dev_lines = []
    for dev in obs.developers:
        dev_lines.append(
            f"{dev.id}: name={dev.name}; capacity={dev.capacity}; skill={dev.skill:.2f}; "
            f"specializations={','.join(dev.specializations) or 'none'}"
        )

    candidate_lines = []
    task_map = {task.id: task for task in obs.jira_tickets}
    dev_map = {dev.id: dev for dev in obs.developers}
    for cand_task_id, cand_dev_id in candidates:
        task = task_map[cand_task_id]
        dev = dev_map[cand_dev_id]
        candidate_lines.append(
            f"- task_id={cand_task_id}, developer_id={cand_dev_id}, "
            f"task={task.title}, deadline={task.deadline}, priority={int(task.priority)}, "
            f"skill_match={'yes' if set(task.tags) & set(dev.specializations) else 'no'}"
        )

    observation_block = "\n".join(
        [
            f"TASK={task_id}",
            f"SPRINT_DAY={obs.sprint_day}",
            f"COMPLETED={','.join(obs.completed_task_ids) or 'none'}",
            f"RECENT_EVENTS={len(obs.recent_events)}",
            f"PENDING_EVENTS={len(obs.pending_events)}",
            "TASKS:",
            *task_lines,
            "DEVELOPERS:",
            *dev_lines,
            (
                f"RECOMMENDED_ASSIGNMENT: task_id={recommended[0]}, developer_id={recommended[1]}"
                if recommended else
                "RECOMMENDED_ASSIGNMENT: none"
            ),
            "VALID_ASSIGNMENTS:",
            *(candidate_lines or ["- none"]),
        ]
    )
    return observation_block, recommended


def get_llm_action(
    client: OpenAI,
    task_id: str,
    observation_text: str,
    history: List[str],
    step_num: int,
    max_steps: int,
    fallback_pair: Optional[Tuple[str, str]],
) -> Tuple[Dict[str, Any], Optional[str]]:
    strategy = STRATEGIES.get(task_id, "Assign work effectively.")
    remaining = max_steps - step_num
    urgency = " URGENT: finalize a strong assignment now." if remaining <= 2 else ""
    history_block = "\n".join(history[-4:]) if history else "None"

    user_prompt = textwrap.dedent(
        f"""
        TASK_STRATEGY: {strategy}
        Step {step_num}/{max_steps}.{urgency}

        OBSERVATION:
        {observation_text}

        HISTORY:
        {history_block}

        JSON:
        """
    ).strip()

    error_msg: Optional[str] = None
    action_dict: Dict[str, Any]

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_COMPLETION_TOKENS,
        )
        raw = (response.choices[0].message.content or "").strip()
        action_dict = parse_action_json(raw)
    except Exception as exc:
        error_msg = str(exc)
        action_dict = {}

    if fallback_pair and (
        action_dict.get("task_id") is None or
        action_dict.get("developer_id") is None
    ):
        action_dict = {
            "action_type": "task_assignment",
            "task_id": fallback_pair[0],
            "developer_id": fallback_pair[1],
        }

    return action_dict, error_msg


def format_action_str(task_id: Optional[str], developer_id: Optional[str]) -> str:
    if task_id and developer_id:
        return f"task_assignment(task_id='{task_id}',developer_id='{developer_id}')"
    return "skip"


def run_task(task_id: str, client: OpenAI) -> float:
    max_steps = TASKS[task_id]["max_steps"]
    difficulty = TASKS[task_id]["difficulty"]

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    env = SprintEnv(difficulty=difficulty, max_steps=max_steps, use_llm=False, seed=42)

    try:
        obs = env.reset()
        history: List[str] = []

        for step in range(1, max_steps + 1):
            observation_text, fallback_pair = build_prompt(obs, task_id)
            action_dict, llm_error = get_llm_action(
                client=client,
                task_id=task_id,
                observation_text=observation_text,
                history=history,
                step_num=step,
                max_steps=max_steps,
                fallback_pair=fallback_pair,
            )

            task_item_id = action_dict.get("task_id")
            developer_id = action_dict.get("developer_id")
            valid_pairs = set(_valid_pairs(obs))

            if (task_item_id, developer_id) not in valid_pairs and fallback_pair:
                task_item_id, developer_id = fallback_pair

            if task_item_id and developer_id:
                action = Action(task_id=task_item_id, developer_id=developer_id)
                obs, reward, done, _ = env.step(action)
                rewards.append(reward)
                steps_taken = step
                action_str = format_action_str(task_item_id, developer_id)
                log_step(
                    step=step,
                    action=action_str,
                    reward=reward,
                    done=done,
                    error=llm_error,
                )
                history.append(f"step={step} task={task_item_id} dev={developer_id} reward={reward:.2f}")
                if done:
                    break
            else:
                rewards.append(0.0)
                log_step(
                    step=step,
                    action="skip",
                    reward=0.0,
                    done=False,
                    error=llm_error or "bad_params",
                )

        if rewards:
            avg_reward = sum(rewards) / len(rewards)
            score = max(0.0, min(1.0, 0.5 + (avg_reward * 0.49)))
        success = score >= SUCCESS_SCORE_THRESHOLD
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    for task_id in TASKS:
        run_task(task_id, client)


if __name__ == "__main__":
    main()
