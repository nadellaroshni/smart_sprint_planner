#!/usr/bin/env python3
"""
Inference Script - Smart Sprint Planner OpenEnv
===============================================

MANDATORY ENV VARS:
    API_BASE_URL   The API endpoint for the LLM (default: https://router.huggingface.co/v1)
    MODEL_NAME     The model identifier to use (default: Qwen/Qwen2.5-72B-Instruct)
    OPENAI_API_KEY Your API key for LLM access

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
import textwrap
from typing import Any, Dict, List, Optional

from openai import OpenAI

from env.environment import SprintEnv
from env.models import Action, Difficulty, Observation
from env.task_catalog import get_task_catalog

BENCHMARK = "smart_sprint_planner"

# Configuration - Prioritize validator-injected environment variables
# The validator injects API_KEY and API_BASE_URL
API_KEY = os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("HF_TOKEN")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
TEMPERATURE = 0.3
MAX_COMPLETION_TOKENS = 800

TASKS = {
    "task1_easy": {"max_steps": 10, "difficulty": Difficulty.EASY},
    "task2_medium": {"max_steps": 15, "difficulty": Difficulty.MEDIUM},
    "task3_hard": {"max_steps": 20, "difficulty": Difficulty.HARD},
}

STRATEGIES = {
    "task1_easy": (
        "1) Analyze backlog and developer capacities. "
        "2) Prioritize tasks by deadline and urgency. "
        "3) Assign tasks to developers with matching skills and capacity. "
        "4) Optimize for completion rate and on-time delivery."
    ),
    "task2_medium": (
        "1) Handle incoming disruption gracefully. "
        "2) Re-assess current assignments and priorities. "
        "3) Reallocate resources to accommodate new urgent work. "
        "4) Maintain feasibility while preserving progress."
    ),
    "task3_hard": (
        "1) Monitor for multiple disruption waves. "
        "2) Adapt dynamically to changing conditions. "
        "3) Maintain overall task completion while handling disruptions. "
        "4) Demonstrate sophisticated replanning under volatility."
    ),
}


# Logging functions
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={error if error else 'null'}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# JSON parsing
def parse_action_json(text: str) -> Dict[str, Any]:
    """Extract JSON from LLM response."""
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


# LLM call
def get_llm_action(
    client: OpenAI,
    task_id: str,
    team_info: str,
    observation: str,
    step_num: int,
    max_steps: int,
    history: List[str],
) -> Dict[str, Any]:
    """Get next action from LLM."""
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
        f"RULES:\n"
        f"- Output EXACTLY ONE JSON object. No markdown, no text.\n"
        f"- Format: {{\"action_type\": \"task_assignment\", \"task_id\": \"<id>\", \"developer_id\": \"<id>\"}}\n"
        f"- You have {max_steps} steps maximum."
    )

    history_text = "\n".join(history[-4:]) if history else "None"
    user_prompt = f"Step {step_num}/{max_steps}.{urgency}\n\nOBS:\n{observation}\n\nHIST:\n{history_text}\n\nJSON:"

    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_completion_tokens=MAX_COMPLETION_TOKENS,
        )
        return parse_action_json((resp.choices[0].message.content or "").strip())
    except Exception as exc:
        return {
            "action_type": "task_assignment",
            "task_id": task_id,
            "developer_id": "unknown",
            "error": str(exc),
        }


# Task runner
def run_task(task_id: str, client: OpenAI) -> float:
    """Run one task and return clamped score in [0.01, 0.99]."""
    task_info = TASKS.get(task_id, {})
    max_steps = task_info.get("max_steps", 20)
    difficulty = task_info.get("difficulty", Difficulty.EASY)

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    rewards: List[float] = []
    steps_taken = 0
    score = 0.01  # Never return exactly 0
    success = False

    try:
        env = SprintEnv(difficulty=difficulty, max_steps=max_steps, use_llm=False, seed=42)
        obs = env.reset()

        team_members = obs.developers if hasattr(obs, "developers") else []
        team_info = f"Size: {len(team_members)}, Capacity: {sum(d.capacity for d in team_members) if team_members else 0}"

        history: List[str] = []

        for step in range(1, max_steps + 1):
            obs_str = str(obs)[:500]

            try:
                action_dict = get_llm_action(client, task_id, team_info, obs_str, step, max_steps, history)
            except Exception as e:
                print(f"[DEBUG] LLM error: {e}", flush=True)
                action_dict = {
                    "action_type": "task_assignment",
                    "task_id": task_id,
                    "developer_id": "unknown",
                }

            action_type = action_dict.get("action_type", "unknown")
            task_item_id = action_dict.get("task_id")
            dev_id = action_dict.get("developer_id")

            try:
                if task_item_id and dev_id and obs.jira_tickets and obs.developers:
                    action = Action(task_id=task_item_id, developer_id=dev_id)
                    obs, reward, done, info = env.step(action)
                    rewards.append(reward)
                    steps_taken = step

                    log_step(step=step, action=action_type, reward=reward, done=done, error=None)
                    history.append(f"S{step}: {action_type}→{reward:.2f}")

                    if done:
                        break
                else:
                    rewards.append(0.0)
                    log_step(step=step, action="skip", reward=0.0, done=False, error="bad_params")

            except Exception as e:
                print(f"[DEBUG] Step error: {e}", flush=True)
                log_step(step=step, action="error", reward=0.0, done=True, error=str(e))
                break

        # Calculate score, clamp to [0.01, 0.99]
        if rewards:
            total_reward = sum(rewards)
            avg_reward = total_reward / len(rewards)
            # Scale to [0.01, 0.99]
            score = max(0.01, min(0.99, 0.5 + (avg_reward * 0.49)))
        else:
            score = 0.01

        success = score > 0.3

    except Exception as exc:
        print(f"[DEBUG] Task {task_id} failed: {exc}", flush=True)
        score = 0.01
        success = False

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


# Main
def main() -> None:
    """Run all tasks in sequence."""
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    print(f"[DEBUG] Model: {MODEL_NAME}", flush=True)
    print(f"[DEBUG] API: {API_BASE_URL}", flush=True)

    scores = []
    for task_id in TASKS:
        try:
            task_score = run_task(task_id, client)
            scores.append(task_score)
        except Exception as e:
            print(f"[DEBUG] Task {task_id} exception: {e}", flush=True)
            scores.append(0.01)

    if scores:
        overall = max(0.01, min(0.99, sum(scores) / len(scores)))
        print(f"\n[SUMMARY] overall_score={overall:.3f} tasks={len(scores)}", flush=True)


if __name__ == "__main__":
    main()
