"""
Evaluation Script — Compare agents across all difficulty levels.

Produces:
  1. Console table with mean ± std scores per difficulty per agent
  2. logs/eval_report.json  — machine-readable full results
  3. logs/train_curve.txt   — rolling mean score from training log

Usage:
    python eval.py                             # compare heuristic vs DDQN (best checkpoint)
    python eval.py --checkpoint checkpoints/ep0200
    python eval.py --episodes 20               # episodes per difficulty per agent
    python eval.py --report-only               # just parse logs/train.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from env.environment import SprintEnv
from env.models import Difficulty
from env.graders import grade
from agent.dqn_agent import DDQNAgent
from agent.heuristic_agent import HeuristicAgent
from agent.features import encode

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("eval")


# ── Single episode runner ─────────────────────────────────────────────────────

def run_episode_heuristic(env: SprintEnv, difficulty: Difficulty) -> dict:
    agent = HeuristicAgent()
    obs = env.reset(difficulty=difficulty)
    done = False
    total_reward = 0.0
    while not done:
        action = agent.act(obs)
        if action is None:
            break
        obs, reward, done, _ = env.step(action)
        total_reward += reward
    result = grade(env)
    return {"reward": total_reward, **result}


def run_episode_ddqn(env: SprintEnv, agent: DDQNAgent, difficulty: Difficulty) -> dict:
    obs = env.reset(difficulty=difficulty)
    done = False
    total_reward = 0.0
    while not done:
        action = agent.act(obs, deterministic=True)
        if action is None:
            break
        obs, reward, done, _ = env.step(action)
        total_reward += reward
    result = grade(env)
    return {"reward": total_reward, **result}


# ── Multi-episode evaluation ──────────────────────────────────────────────────

def evaluate_agent(
    agent_name: str,
    run_fn,
    difficulties: List[Difficulty],
    n_episodes: int,
) -> Dict[str, dict]:
    results = {}
    for diff in difficulties:
        scores   = []
        rewards  = []
        breakdowns: List[dict] = []

        for _ in range(n_episodes):
            r = run_fn(diff)
            scores.append(r["score"])
            rewards.append(r["reward"])
            breakdowns.append(r.get("breakdown", {}))

        # Aggregate breakdown dimensions
        agg_bd = {}
        for key in (breakdowns[0] if breakdowns else {}):
            vals = [bd[key] for bd in breakdowns if key in bd]
            agg_bd[key] = {
                "mean": round(float(np.mean(vals)), 3),
                "std":  round(float(np.std(vals)),  3),
            }

        results[diff.value] = {
            "score_mean":   round(float(np.mean(scores)),  3),
            "score_std":    round(float(np.std(scores)),   3),
            "score_min":    round(float(np.min(scores)),   3),
            "score_max":    round(float(np.max(scores)),   3),
            "reward_mean":  round(float(np.mean(rewards)), 3),
            "breakdown":    agg_bd,
        }
        logger.info(
            f"  [{agent_name:10s}] {diff.value:8s} | "
            f"score={results[diff.value]['score_mean']:.3f}±{results[diff.value]['score_std']:.3f} | "
            f"reward={results[diff.value]['reward_mean']:.2f}"
        )

    return results


# ── Training curve parser ─────────────────────────────────────────────────────

def parse_training_curve(log_path: str = "logs/train.jsonl") -> None:
    p = Path(log_path)
    if not p.exists():
        logger.info(f"No training log found at {log_path}")
        return

    entries = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    if not entries:
        return

    episodes   = [e["episode"]        for e in entries]
    mean_scores = [e.get("mean_score_50", 0) for e in entries]
    epsilons    = [e.get("epsilon", 0)        for e in entries]

    out_lines = ["episode,mean_score_50,epsilon"]
    for ep, ms, eps in zip(episodes, mean_scores, epsilons):
        out_lines.append(f"{ep},{ms:.4f},{eps:.4f}")

    curve_path = Path("logs/train_curve.csv")
    curve_path.write_text("\n".join(out_lines))
    logger.info(f"Training curve saved to {curve_path}")

    # Print a simple ASCII plot (every 10th episode)
    logger.info("\n  Training curve (mean score, last 50 eps):")
    max_val = max(mean_scores) if mean_scores else 1.0
    for i in range(0, len(episodes), max(1, len(episodes) // 20)):
        bar_len = int(mean_scores[i] / max(max_val, 0.01) * 30)
        bar = "█" * bar_len + "░" * (30 - bar_len)
        logger.info(f"  ep{episodes[i]:4d} [{bar}] {mean_scores[i]:.3f}")


# ── Comparison table ──────────────────────────────────────────────────────────

def print_comparison_table(all_results: dict) -> None:
    diffs = [d.value for d in Difficulty]
    agents = list(all_results.keys())

    header = f"{'Agent':12s} | " + " | ".join(f"{d:8s}" for d in diffs) + " | MEAN"
    print("\n" + "=" * len(header))
    print("  EVALUATION RESULTS")
    print("=" * len(header))
    print("  " + header)
    print("  " + "-" * (len(header) - 2))

    for agent_name, diff_results in all_results.items():
        scores  = [diff_results.get(d, {}).get("score_mean", 0.0) for d in diffs]
        overall = float(np.mean(scores))
        row     = f"{agent_name:12s} | " + " | ".join(f"{s:.3f}   " for s in scores) + f" | {overall:.3f}"
        print("  " + row)

    print("=" * len(header) + "\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate Sprint Planning agents")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best",
                        help="DDQN checkpoint directory")
    parser.add_argument("--episodes",   type=int, default=10,
                        help="Episodes per difficulty per agent")
    parser.add_argument("--report-only", action="store_true",
                        help="Only parse training log, skip evaluation")
    args = parser.parse_args()

    parse_training_curve()

    if args.report_only:
        return

    env = SprintEnv(max_steps=20, use_llm=False)
    difficulties = list(Difficulty)
    all_results: Dict[str, dict] = {}

    # ── Heuristic baseline ────────────────────────────────────────────────────
    logger.info("\nEvaluating Heuristic agent…")
    all_results["Heuristic"] = evaluate_agent(
        agent_name  = "Heuristic",
        run_fn      = lambda d: run_episode_heuristic(env, d),
        difficulties = difficulties,
        n_episodes  = args.episodes,
    )

    # ── DDQN agent ────────────────────────────────────────────────────────────
    ckpt = Path(args.checkpoint)
    if (ckpt / "online.pkl").exists():
        logger.info(f"\nEvaluating DDQN agent from {ckpt}…")
        ddqn = DDQNAgent()
        ddqn.load(str(ckpt))
        all_results["DDQN"] = evaluate_agent(
            agent_name  = "DDQN",
            run_fn      = lambda d: run_episode_ddqn(env, ddqn, d),
            difficulties = difficulties,
            n_episodes  = args.episodes,
        )
    else:
        logger.warning(f"No DDQN checkpoint found at {ckpt}. Run train.py first.")

    # ── Output ────────────────────────────────────────────────────────────────
    print_comparison_table(all_results)

    report_path = Path("logs/eval_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(all_results, indent=2))
    logger.info(f"Full report saved to {report_path}")


if __name__ == "__main__":
    main()