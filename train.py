"""
Training Script — DDQN Agent on SmartSprintEnv.

Features:
  ─ Curriculum learning: start on EASY, graduate to HARD
  ─ Rolling stats (last 50 episodes) to track learning
  ─ Checkpoint saving (best + periodic)
  ─ JSONL experiment log (parseable by eval.py)
  ─ Graceful Ctrl-C: saves agent before exit
  ─ Deterministic: fixed seeds

Usage:
    python train.py                          # default config
    python train.py --episodes 500 --render
    python train.py --difficulty hard --episodes 200
    python train.py --resume checkpoints/best
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import time
from collections import deque
from pathlib import Path
from typing import Deque, List, Optional

import numpy as np

from env.environment import SprintEnv
from env.models import Action, Difficulty
from env.graders import grade
from agent.dqn_agent import DDQNAgent
from agent.features import encode

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("train")

# ── Paths ────────────────────────────────────────────────────────────────────
CHECKPOINT_DIR = Path("checkpoints")
LOG_PATH       = Path("logs/train.jsonl")
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

# ── Curriculum schedule ──────────────────────────────────────────────────────
#  (start_episode, difficulty)
CURRICULUM_LIMITS = {
    Difficulty.EASY: {"graduate_after": 60, "threshold": 0.68},
    Difficulty.MEDIUM: {"graduate_after": 180, "threshold": 0.64},
}


def get_difficulty(
    episode: int,
    fixed: Optional[Difficulty] = None,
    rolling_score: Optional[float] = None,
) -> Difficulty:
    if fixed is not None:
        return fixed

    if episode <= CURRICULUM_LIMITS[Difficulty.EASY]["graduate_after"]:
        if rolling_score is not None and episode >= 30 and rolling_score >= CURRICULUM_LIMITS[Difficulty.EASY]["threshold"]:
            return Difficulty.MEDIUM
        return Difficulty.EASY

    if episode <= CURRICULUM_LIMITS[Difficulty.MEDIUM]["graduate_after"]:
        if rolling_score is not None and episode >= 120 and rolling_score >= CURRICULUM_LIMITS[Difficulty.MEDIUM]["threshold"]:
            return Difficulty.HARD
        return Difficulty.MEDIUM

    return Difficulty.HARD


# ── Episode runner ────────────────────────────────────────────────────────────

def run_episode(
    env:         SprintEnv,
    agent:       DDQNAgent,
    difficulty:  Difficulty,
    render:      bool = False,
    train:       bool = True,
) -> dict:
    """
    Run one full episode.

    Returns a dict with episode metrics for logging.
    """
    obs = env.reset(difficulty=difficulty)
    state_vec = encode(obs)

    episode_reward  = 0.0
    steps           = 0
    done            = False

    if render:
        print(env.render())

    while not done:
        action = agent.act(obs, deterministic=not train)

        if action is None:
            logger.debug("No valid action — ending episode early.")
            break

        next_obs, reward, done, info = env.step(action)
        next_state_vec = encode(next_obs)

        if train:
            action_idx = agent.encode_action(obs, action.task_id, action.developer_id)
            agent.observe(
                state      = state_vec,
                action_idx = action_idx,
                reward     = reward,
                next_state = next_state_vec,
                done       = done,
                task_id    = action.task_id,
                dev_id     = action.developer_id,
            )

        episode_reward += reward
        steps          += 1
        state_vec       = next_state_vec
        obs             = next_obs

        if render:
            print(env.render())

    grade_result = grade(env)
    return {
        "reward":     episode_reward,
        "score":      grade_result["score"],
        "breakdown":  grade_result["breakdown"],
        "steps":      steps,
        "difficulty": difficulty.value,
        **agent.stats,
    }


# ── Main training loop ────────────────────────────────────────────────────────

def train(
    total_episodes: int       = 400,
    fixed_difficulty: Optional[Difficulty] = None,
    render:  bool             = False,
    resume:  Optional[str]    = None,
    eval_every: int           = 25,
    save_every: int           = 50,
):
    env   = SprintEnv(max_steps=20, use_llm=False, sample_scenarios=True, seed=42)
    agent = DDQNAgent(
        lr             = 1e-3,
        gamma          = 0.95,
        epsilon_start  = 1.0,
        epsilon_end    = 0.05,
        decay_steps    = total_episodes * 10,
        batch_size     = 64,
        target_update  = 200,
        buffer_size    = 10_000,
        seed           = 42,
    )

    if resume:
        agent.load(resume)
        logger.info(f"Resumed from {resume}")

    # Rolling windows
    reward_window: Deque[float] = deque(maxlen=50)
    score_window:  Deque[float] = deque(maxlen=50)
    best_score = 0.0

    # Graceful shutdown
    _stop = [False]
    def _handler(sig, frame):
        logger.info("Interrupt received — saving and exiting…")
        _stop[0] = True
    signal.signal(signal.SIGINT, _handler)

    log_file = LOG_PATH.open("a")
    start_time = time.time()

    logger.info(f"Training for {total_episodes} episodes | curriculum={fixed_difficulty or 'auto'}")

    for ep in range(1, total_episodes + 1):
        if _stop[0]:
            break

        rolling_score = float(np.mean(score_window)) if score_window else None
        difficulty = get_difficulty(ep, fixed_difficulty, rolling_score=rolling_score)
        result     = run_episode(env, agent, difficulty, render=render, train=True)
        agent.episode = ep

        reward_window.append(result["reward"])
        score_window.append(result["score"])

        # ── Logging ──────────────────────────────────────────────────────────
        log_entry = {
            "episode":       ep,
            "timestamp":     time.time(),
            **result,
            "mean_reward_50": round(float(np.mean(reward_window)), 4),
            "mean_score_50":  round(float(np.mean(score_window)),  4),
        }
        log_file.write(json.dumps(log_entry) + "\n")
        log_file.flush()

        if ep % 10 == 0:
            logger.info(
                f"Ep {ep:4d}/{total_episodes} | diff={difficulty.value:6s} | "
                f"reward={result['reward']:6.2f} | score={result['score']:.3f} | "
                f"mean50={np.mean(score_window):.3f} | ε={result['epsilon']:.3f} | "
                f"loss={result['mean_loss_100']:.5f} | buf={result['buffer_size']}"
            )

        # ── Periodic save ─────────────────────────────────────────────────────
        if ep % save_every == 0:
            agent.save(str(CHECKPOINT_DIR / f"ep{ep:04d}"))

        # ── Best model save ───────────────────────────────────────────────────
        if result["score"] > best_score:
            best_score = result["score"]
            agent.save(str(CHECKPOINT_DIR / "best"))
            logger.info(f"  ★ New best score: {best_score:.3f} (ep {ep})")

        # ── Evaluation pass ───────────────────────────────────────────────────
        if ep % eval_every == 0:
            _evaluate(env, agent, ep)

    elapsed = time.time() - start_time
    logger.info(f"\nTraining complete | {ep} episodes | {elapsed:.1f}s")
    agent.save(str(CHECKPOINT_DIR / "final"))
    log_file.close()

    return agent


# ── Evaluation (deterministic, all difficulties) ──────────────────────────────

def _evaluate(env: SprintEnv, agent: DDQNAgent, episode: int) -> None:
    logger.info(f"\n{'─' * 55}")
    logger.info(f"  EVALUATION at episode {episode}")
    logger.info(f"{'─' * 55}")
    for diff in Difficulty:
        scores = []
        for _ in range(5):
            result = run_episode(env, agent, diff, render=False, train=False)
            scores.append(result["score"])
        logger.info(
            f"  {diff.value:8s} | mean={np.mean(scores):.3f} | "
            f"std={np.std(scores):.3f} | min={min(scores):.3f} | max={max(scores):.3f}"
        )
    logger.info(f"{'─' * 55}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DDQN agent on SmartSprintEnv")
    parser.add_argument("--episodes",   type=int,  default=400,    help="Total training episodes")
    parser.add_argument("--difficulty", type=str,  default=None,   help="Fix difficulty (easy/medium/hard)")
    parser.add_argument("--render",     action="store_true",       help="Render sprint board each step")
    parser.add_argument("--resume",     type=str,  default=None,   help="Path to checkpoint directory")
    parser.add_argument("--eval-every", type=int,  default=25,     help="Evaluation interval (episodes)")
    parser.add_argument("--save-every", type=int,  default=50,     help="Checkpoint interval (episodes)")
    args = parser.parse_args()

    diff_map = {"easy": Difficulty.EASY, "medium": Difficulty.MEDIUM, "hard": Difficulty.HARD}
    fixed = diff_map.get(args.difficulty) if args.difficulty else None

    train(
        total_episodes    = args.episodes,
        fixed_difficulty  = fixed,
        render            = args.render,
        resume            = args.resume,
        eval_every        = args.eval_every,
        save_every        = args.save_every,
    )
