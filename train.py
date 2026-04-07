"""
Train a DDQN agent on SmartSprintEnv.
"""

from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
import time
from collections import deque
from pathlib import Path
from typing import Deque, Optional

import numpy as np

from agent.dqn_agent import DDQNAgent
from agent.features import action_embedding, action_space, encode, valid_action_embeddings
from agent.heuristic_agent import HeuristicAgent
from env.environment import SprintEnv
from env.graders import grade
from env.models import Difficulty
from env.tasks import dataset_available, get_scenario_count

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("train")

CHECKPOINT_DIR = Path("checkpoints")
LOG_PATH = Path("logs/train.jsonl")
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

CURRICULUM_LIMITS = {
    Difficulty.EASY: {"graduate_after": 35, "threshold": 0.72},
    Difficulty.MEDIUM: {"graduate_after": 95, "threshold": 0.74},
}


def get_difficulty(
    episode: int,
    fixed: Optional[Difficulty] = None,
    rolling_score: Optional[float] = None,
) -> Difficulty:
    if fixed is not None:
        return fixed

    easy_limit = CURRICULUM_LIMITS[Difficulty.EASY]
    medium_limit = CURRICULUM_LIMITS[Difficulty.MEDIUM]

    if episode <= easy_limit["graduate_after"]:
        if rolling_score is not None and episode >= 30 and rolling_score >= easy_limit["threshold"]:
            return Difficulty.MEDIUM
        return Difficulty.EASY

    if episode <= medium_limit["graduate_after"]:
        if rolling_score is not None and episode >= 72 and rolling_score >= medium_limit["threshold"]:
            return Difficulty.HARD
        if episode >= 70 and episode % 2 == 0:
            return Difficulty.HARD
        return Difficulty.MEDIUM

    return Difficulty.HARD


def build_env(sample_scenarios: bool, scenario_split: Optional[str], seed: int) -> SprintEnv:
    return SprintEnv(
        max_steps=20,
        use_llm=False,
        sample_scenarios=sample_scenarios,
        scenario_split=scenario_split,
        seed=seed,
    )


def run_episode(
    env: SprintEnv,
    agent: DDQNAgent,
    difficulty: Difficulty,
    render: bool = False,
    train: bool = True,
    scenario_index: Optional[int] = None,
) -> dict:
    obs = env.reset(difficulty=difficulty, scenario_index=scenario_index)
    state_vec = encode(obs)

    episode_reward = 0.0
    steps = 0
    done = False

    if render:
        print(env.render())

    while not done:
        action = agent.act(obs, deterministic=not train)
        if action is None:
            logger.debug("No valid action; ending episode early.")
            break

        next_obs, reward, done, _ = env.step(action)
        next_state_vec = encode(next_obs)

        if train:
            agent.observe(
                state=state_vec,
                action_features=action_embedding(obs, action.task_id, action.developer_id),
                reward=reward,
                next_state=next_state_vec,
                next_valid_action_features=valid_action_embeddings(next_obs),
                done=done,
                task_id=action.task_id,
                dev_id=action.developer_id,
            )

        episode_reward += reward
        steps += 1
        state_vec = next_state_vec
        obs = next_obs

        if render:
            print(env.render())

    grade_result = grade(env)
    return {
        "reward": episode_reward,
        "score": grade_result["score"],
        "breakdown": grade_result["breakdown"],
        "steps": steps,
        "difficulty": difficulty.value,
        "scenario_id": env.state().get("scenario_id", ""),
        **agent.stats,
    }


def evaluate_agent(env: SprintEnv, agent: DDQNAgent, episode: int, n_runs: int = 5) -> dict:
    logger.info("")
    logger.info("%s", "-" * 60)
    logger.info("Evaluation at episode %d", episode)
    logger.info("%s", "-" * 60)

    per_diff = {}
    aggregate_scores = []
    aggregate_adaptability = []

    for diff in Difficulty:
        scores = []
        adaptability = []
        scenario_count = get_scenario_count(diff, split=env.scenario_split)
        for run_idx in range(n_runs):
            result = run_episode(env, agent, diff, train=False, scenario_index=run_idx % scenario_count)
            scores.append(result["score"])
            adaptability.append(result["breakdown"].get("adaptability", 0.0))

        diff_summary = {
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
            "min": float(np.min(scores)),
            "max": float(np.max(scores)),
            "adaptability_mean": float(np.mean(adaptability)),
        }
        per_diff[diff.value] = diff_summary
        aggregate_scores.extend(scores)
        aggregate_adaptability.extend(adaptability)
        logger.info(
            "  %-8s | mean=%.3f | std=%.3f | min=%.3f | max=%.3f | adapt=%.3f",
            diff.value,
            diff_summary["mean"],
            diff_summary["std"],
            diff_summary["min"],
            diff_summary["max"],
            diff_summary["adaptability_mean"],
        )

    summary = {
        "episode": episode,
        "mean_score": float(np.mean(aggregate_scores)),
        "mean_adaptability": float(np.mean(aggregate_adaptability)),
        "per_difficulty": per_diff,
    }
    logger.info("%s", "-" * 60)
    return summary


def pretrain_from_heuristic(
    agent: DDQNAgent,
    env: SprintEnv,
    epochs: int = 6,
    batch_size: int = 64,
) -> None:
    heuristic = HeuristicAgent()
    features = []
    targets = []

    for difficulty in Difficulty:
        scenario_count = get_scenario_count(difficulty, split=env.scenario_split)
        for scenario_index in range(scenario_count):
            obs = env.reset(difficulty=difficulty, scenario_index=scenario_index)
            done = False

            while not done:
                valid_pairs = action_space(obs)
                if not valid_pairs:
                    break

                chosen = heuristic.act(obs)
                chosen_pair = (chosen.task_id, chosen.developer_id) if chosen is not None else None
                state_vec = encode(obs)

                for task_id, dev_id in valid_pairs:
                    features.append(np.concatenate([state_vec, action_embedding(obs, task_id, dev_id)]))
                    targets.append([1.0 if chosen_pair == (task_id, dev_id) else 0.0])

                if chosen is None:
                    break
                obs, _, done, _ = env.step(chosen)

    if not features:
        logger.info("Skipping heuristic pretraining because no demonstrations were available.")
        return

    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(targets, dtype=np.float32)
    rng = np.random.default_rng(agent.seed)

    logger.info("Heuristic pretraining on %d state-action examples for %d epochs", len(x), epochs)
    for epoch in range(epochs):
        indices = rng.permutation(len(x))
        epoch_losses = []
        for start in range(0, len(indices), batch_size):
            batch_idx = indices[start : start + batch_size]
            loss = agent.online.update(x[batch_idx], y[batch_idx])
            epoch_losses.append(loss)
        logger.info("  pretrain epoch %d/%d | loss=%.5f", epoch + 1, epochs, float(np.mean(epoch_losses)))

    agent.target.copy_weights_from(agent.online)


def train(
    total_episodes: int = 400,
    fixed_difficulty: Optional[Difficulty] = None,
    render: bool = False,
    resume: Optional[str] = None,
    eval_every: int = 25,
    save_every: int = 50,
) -> DDQNAgent:
    use_dataset = dataset_available()
    train_split = "train" if use_dataset else None
    eval_split = "eval" if use_dataset else None

    env = build_env(sample_scenarios=use_dataset, scenario_split=train_split, seed=42)
    eval_env = build_env(sample_scenarios=use_dataset, scenario_split=eval_split, seed=142)

    agent = DDQNAgent(
        lr=5e-4,
        gamma=0.98,
        epsilon_start=1.0,
        epsilon_end=0.02,
        decay_steps=total_episodes * 12,
        batch_size=64,
        target_update=200,
        buffer_size=30_000,
        guided_exploration_prob=0.7,
        seed=42,
    )

    if resume:
        agent.load(resume)
        logger.info("Resumed from %s", resume)
    elif use_dataset:
        pretrain_from_heuristic(agent, env, epochs=8)

    reward_window: Deque[float] = deque(maxlen=50)
    score_window: Deque[float] = deque(maxlen=50)
    best_eval_score = float("-inf")
    seen_counts = {difficulty: 0 for difficulty in Difficulty}

    stop_requested = [False]

    def _handler(sig, frame):
        logger.info("Interrupt received; saving final checkpoint before exit.")
        stop_requested[0] = True

    signal.signal(signal.SIGINT, _handler)

    start_time = time.time()
    logger.info(
        "Training for %d episodes | curriculum=%s | dataset=%s | train_split=%s | eval_split=%s",
        total_episodes,
        fixed_difficulty.value if fixed_difficulty else "auto",
        use_dataset,
        train_split or "fallback",
        eval_split or "fallback",
    )

    log_mode = "a" if resume else "w"
    with LOG_PATH.open(log_mode, encoding="utf-8") as log_file:
        for ep in range(1, total_episodes + 1):
            if stop_requested[0]:
                break

            rolling_score = float(np.mean(score_window)) if score_window else None
            difficulty = get_difficulty(ep, fixed_difficulty, rolling_score=rolling_score)
            scenario_count = get_scenario_count(difficulty, split=train_split)
            scenario_index = seen_counts[difficulty] % scenario_count
            result = run_episode(env, agent, difficulty, render=render, train=True, scenario_index=scenario_index)
            seen_counts[difficulty] += 1
            agent.episode = ep

            reward_window.append(result["reward"])
            score_window.append(result["score"])

            log_entry = {
                "episode": ep,
                "timestamp": time.time(),
                **result,
                "mean_reward_50": round(float(np.mean(reward_window)), 4),
                "mean_score_50": round(float(np.mean(score_window)), 4),
            }
            log_file.write(json.dumps(log_entry) + "\n")
            log_file.flush()

            if ep % 10 == 0:
                logger.info(
                    "Ep %4d/%d | diff=%-6s | reward=%6.2f | score=%.3f | mean50=%.3f | eps=%.3f | loss=%.5f | buf=%d | scenario=%s",
                    ep,
                    total_episodes,
                    difficulty.value,
                    result["reward"],
                    result["score"],
                    float(np.mean(score_window)),
                    result["epsilon"],
                    result["mean_loss_100"],
                    result["buffer_size"],
                    result["scenario_id"] or "-",
                )

            if ep % save_every == 0:
                agent.save(str(CHECKPOINT_DIR / f"ep{ep:04d}"))

            if ep % eval_every == 0:
                evaluation = evaluate_agent(eval_env, agent, ep)
                if evaluation["mean_score"] > best_eval_score:
                    best_eval_score = evaluation["mean_score"]
                    agent.save(str(CHECKPOINT_DIR / "best"))
                    logger.info(
                        "New best eval checkpoint: mean_score=%.3f | mean_adaptability=%.3f",
                        evaluation["mean_score"],
                        evaluation["mean_adaptability"],
                    )

    elapsed = time.time() - start_time
    logger.info("")
    logger.info("Training complete | %d episodes | %.1fs", agent.episode, elapsed)
    agent.save(str(CHECKPOINT_DIR / "final"))
    return agent


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DDQN agent on SmartSprintEnv")
    parser.add_argument("--episodes", type=int, default=400, help="Total training episodes")
    parser.add_argument("--difficulty", type=str, default=None, help="Fix difficulty (easy/medium/hard)")
    parser.add_argument("--render", action="store_true", help="Render sprint board each step")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint directory")
    parser.add_argument("--eval-every", type=int, default=25, help="Evaluation interval in episodes")
    parser.add_argument("--save-every", type=int, default=50, help="Checkpoint interval in episodes")
    args = parser.parse_args()

    diff_map = {
        "easy": Difficulty.EASY,
        "medium": Difficulty.MEDIUM,
        "hard": Difficulty.HARD,
    }
    fixed = diff_map.get(args.difficulty) if args.difficulty else None

    train(
        total_episodes=args.episodes,
        fixed_difficulty=fixed,
        render=args.render,
        resume=args.resume,
        eval_every=args.eval_every,
        save_every=args.save_every,
    )
