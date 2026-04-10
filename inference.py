#!/usr/bin/env python3
"""
Inference Script - Smart Sprint Planner OpenEnv
===============================================

Submission credentials:
    API_BASE_URL   The validator LLM proxy endpoint
    API_KEY        The validator-injected proxy key
    MODEL_NAME     The model identifier to use (optional)

Local convenience:
    HF_TOKEN may be used when API_KEY is not set.

STDOUT FORMAT:
    [START] task=<task> env=<benchmark> model=<model>
    [STEP]  step=<n> action=<action> reward=<0.00> done=<true|false> error=<null|msg>
    [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...>
"""

from __future__ import annotations

import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

from env.environment import SprintEnv
from env.models import Action, Difficulty

BENCHMARK = "smart_sprint_planner"
DEFAULT_API_BASE_URL = "https://router.huggingface.co/v1"
DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-72B-Instruct"
TEMPERATURE = 0.3
MAX_COMPLETION_TOKENS = 800
MAX_CANDIDATES = 8


def _get_api_credentials() -> tuple[str, str]:
    api_key = os.environ.get("API_KEY") or os.environ.get("HF_TOKEN")
    if not api_key:
        raise RuntimeError("Missing API_KEY or HF_TOKEN for inference.")
    return api_key, os.environ.get("API_BASE_URL", DEFAULT_API_BASE_URL)


API_KEY, API_BASE_URL = _get_api_credentials()
MODEL_NAME = os.environ.get("MODEL_NAME", DEFAULT_MODEL_NAME)

TASKS = {
    "easy": {"max_steps": 10, "difficulty": Difficulty.EASY},
    "medium": {"max_steps": 15, "difficulty": Difficulty.MEDIUM},
    "hard": {"max_steps": 20, "difficulty": Difficulty.HARD},
}

STRATEGIES = {
    "easy": (
        "1) Analyze backlog and developer capacities. "
        "2) Prioritize tasks by deadline and urgency. "
        "3) Assign tasks to developers with matching skills and capacity. "
        "4) Optimize for completion rate and on-time delivery."
    ),
    "medium": (
        "1) Handle incoming disruption gracefully. "
        "2) Re-assess current assignments and priorities. "
        "3) Reallocate resources to accommodate new urgent work. "
        "4) Maintain feasibility while preserving progress."
    ),
    "hard": (
        "1) Monitor for multiple disruption waves. "
        "2) Adapt dynamically to changing conditions. "
        "3) Maintain overall task completion while handling disruptions. "
        "4) Demonstrate sophisticated replanning under volatility."
    ),
}


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
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


def log_stderr(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def parse_action_json(text: str) -> Dict[str, Any]:
    """Extract a JSON object from the model response."""
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
    for index, char in enumerate(text):
        if char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    return json.loads(text[start : index + 1])
                except json.JSONDecodeError:
                    start = None

    raise ValueError(f"Could not parse JSON from: {text[:200]}")


def _valid_pairs(obs) -> List[Tuple[str, str]]:
    completed_ids = set(obs.completed_task_ids)
    pairs: List[Tuple[str, str]] = []
    for task in obs.jira_tickets:
        if any(dep not in completed_ids for dep in task.dependencies):
            continue
        for dev in obs.developers:
            if dev.capacity >= task.story_points:
                pairs.append((task.id, dev.id))
    return pairs


def _score_pair(obs, task, dev) -> float:
    days_left = max(task.deadline - obs.sprint_day, 0)
    urgency = 10.0 / (days_left + 1)
    priority = int(task.priority) * 2.0
    size_bonus = 1.0 / max(task.story_points, 1)
    skill_match = 2.5 if set(task.tags) & set(dev.specializations) else 0.0
    event_bonus = 1.2 if task.source_event else 0.0
    return urgency + priority + size_bonus + skill_match + event_bonus + (dev.capacity * 0.05)


def _candidate_actions(obs, limit: int = MAX_CANDIDATES) -> List[Tuple[str, str]]:
    scored: List[Tuple[float, Tuple[str, str]]] = []
    task_map = {task.id: task for task in obs.jira_tickets}
    dev_map = {dev.id: dev for dev in obs.developers}
    for task_id, dev_id in _valid_pairs(obs):
        task = task_map[task_id]
        dev = dev_map[dev_id]
        scored.append((_score_pair(obs, task, dev), (task_id, dev_id)))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [pair for _, pair in scored[:limit]]


def _build_prompt_context(obs) -> Tuple[str, str]:
    completed_ids = set(obs.completed_task_ids)
    task_lines = []
    for task in obs.jira_tickets:
        blocked = any(dep not in completed_ids for dep in task.dependencies)
        task_lines.append(
            f"{task.id}: title={task.title}; priority={int(task.priority)}; points={task.story_points}; "
            f"deadline={task.deadline}; tags={','.join(task.tags) or 'none'}; "
            f"deps={','.join(task.dependencies) or 'none'}; source_event={task.source_event or 'none'}; "
            f"blocked={'yes' if blocked else 'no'}"
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
    ranked_candidates = _candidate_actions(obs)
    for task_id, dev_id in ranked_candidates:
        task = task_map[task_id]
        dev = dev_map[dev_id]
        candidate_lines.append(
            f"- task_id={task_id}, developer_id={dev_id}, task={task.title}, "
            f"deadline={task.deadline}, priority={int(task.priority)}, "
            f"match={'yes' if set(task.tags) & set(dev.specializations) else 'no'}"
        )
    recommended = ranked_candidates[0] if ranked_candidates else None

    observation = (
        f"sprint_day={obs.sprint_day}\n"
        f"completed={','.join(obs.completed_task_ids) or 'none'}\n"
        f"recent_events={len(obs.recent_events)} pending_events={len(obs.pending_events)}\n"
        "TASKS:\n"
        + "\n".join(task_lines)
        + "\nDEVELOPERS:\n"
        + "\n".join(dev_lines)
    )
    recommendation_line = (
        f"RECOMMENDED_ASSIGNMENT: task_id={recommended[0]}, developer_id={recommended[1]}"
        if recommended else
        "RECOMMENDED_ASSIGNMENT: none"
    )
    candidates = (
        recommendation_line + "\nVALID_ASSIGNMENTS:\n" +
        ("\n".join(candidate_lines) if candidate_lines else "- none")
    )
    return observation, candidates


def get_llm_action(
    client: OpenAI,
    task_id: str,
    team_info: str,
    observation: str,
    valid_actions: str,
    step_num: int,
    max_steps: int,
    history: List[str],
    fallback_pair: Optional[Tuple[str, str]] = None,
) -> Dict[str, Any]:
    """Request one assignment from the validator-provided LLM proxy."""
    strategy = STRATEGIES.get(task_id, "Assign tasks optimally.")
    remaining = max_steps - step_num
    urgency = ""
    if remaining <= 2:
        urgency = " CRITICAL: Must submit NOW!"
    elif remaining <= 4:
        urgency = f" WARNING: Only {remaining} steps remaining."

    system_prompt = (
        f"You are a sprint planning agent managing task assignments.\n"
        f"TASK: {task_id}\n"
        f"STRATEGY: {strategy}\n"
        f"TEAM: {team_info}\n"
        "RULES:\n"
        "- Output EXACTLY ONE JSON object. No markdown, no text.\n"
        "- Format: {\"action_type\": \"task_assignment\", \"task_id\": \"<id>\", \"developer_id\": \"<id>\"}\n"
        "- Choose a pair from VALID_ASSIGNMENTS whenever one is available.\n"
        "- Do not invent task ids or developer ids.\n"
        "- If uncertain, return RECOMMENDED_ASSIGNMENT exactly.\n"
        "- Prefer urgent, high-priority, on-time, skill-matched assignments.\n"
        f"- You have {max_steps} steps maximum."
    )

    history_text = "\n".join(history[-4:]) if history else "None"
    user_prompt = (
        f"Step {step_num}/{max_steps}.{urgency}\n\n"
        f"OBS:\n{observation}\n\n"
        f"{valid_actions}\n\n"
        f"HIST:\n{history_text}\n\n"
        "JSON:"
    )

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_COMPLETION_TOKENS,
        )
        raw_text = (response.choices[0].message.content or "").strip()
        try:
            return parse_action_json(raw_text)
        except Exception as exc:
            if fallback_pair is not None:
                log_stderr(f"DEBUG: Falling back after unparsable LLM output: {exc}")
                return {
                    "action_type": "task_assignment",
                    "task_id": fallback_pair[0],
                    "developer_id": fallback_pair[1],
                }
            raise
    except Exception as exc:
        error_msg = f"API call failed: {type(exc).__name__}: {str(exc)[:100]}"
        log_stderr(f"ERROR: {error_msg}")
        raise


def run_task(task_id: str, client: OpenAI) -> float:
    """Run one task and return a clamped score in [0.01, 0.99]."""
    task_info = TASKS.get(task_id, {})
    max_steps = task_info.get("max_steps", 20)
    difficulty = task_info.get("difficulty", Difficulty.EASY)

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    rewards: List[float] = []
    steps_taken = 0
    score = 0.01
    success = False

    try:
        env = SprintEnv(difficulty=difficulty, max_steps=max_steps, use_llm=False, seed=42)
        obs = env.reset()
        team_info = f"Size: {len(obs.developers)}, Capacity: {sum(dev.capacity for dev in obs.developers)}"
        history: List[str] = []

        for step in range(1, max_steps + 1):
            observation_text, valid_actions_text = _build_prompt_context(obs)
            best_fallback = _candidate_actions(obs, limit=1)

            try:
                action_dict = get_llm_action(
                    client,
                    task_id,
                    team_info,
                    observation_text,
                    valid_actions_text,
                    step,
                    max_steps,
                    history,
                    best_fallback[0] if best_fallback else None,
                )
            except Exception as exc:
                log_stderr(f"ERROR: Task {task_id} failed at step {step}: {exc}")
                log_step(step=step, action="api_error", reward=0.0, done=True, error=str(exc))
                raise

            action_type = action_dict.get("action_type", "unknown")
            task_item_id = action_dict.get("task_id")
            developer_id = action_dict.get("developer_id")
            valid_pairs = set(_valid_pairs(obs))

            if (task_item_id, developer_id) not in valid_pairs:
                if best_fallback:
                    task_item_id, developer_id = best_fallback[0]
                    action_type = "task_assignment"

            try:
                if task_item_id and developer_id and obs.jira_tickets and obs.developers:
                    action = Action(task_id=task_item_id, developer_id=developer_id)
                    obs, reward, done, _ = env.step(action)
                    rewards.append(reward)
                    steps_taken = step
                    log_step(step=step, action=action_type, reward=reward, done=done, error=None)
                    history.append(f"S{step}: {action_type}->{reward:.2f}")
                    if done:
                        break
                else:
                    rewards.append(0.0)
                    log_step(step=step, action="skip", reward=0.0, done=False, error="bad_params")
            except Exception as exc:
                log_stderr(f"DEBUG: Step error: {exc}")
                log_step(step=step, action="error", reward=0.0, done=True, error=str(exc))
                break

        if rewards:
            avg_reward = sum(rewards) / len(rewards)
            score = max(0.01, min(0.99, 0.5 + (avg_reward * 0.49)))

        success = score > 0.3
    except Exception as exc:
        if "API call failed" in str(exc) or "api_error" in str(exc):
            log_stderr(f"ERROR: Task {task_id} had API error: {exc}")
            raise
        log_stderr(f"DEBUG: Task {task_id} environment error: {exc}")
        score = 0.01
        success = False
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


def main() -> None:
    """Run all benchmark tasks in sequence."""
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    for task_id in TASKS:
        run_task(task_id, client)


if __name__ == "__main__":
    main()
