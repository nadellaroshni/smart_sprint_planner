"""
Evaluate heuristic and DDQN agents.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

from agent.dqn_agent import DDQNAgent
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
logger = logging.getLogger("eval")


def build_env(use_dataset: bool, split: str | None, seed: int) -> SprintEnv:
    return SprintEnv(
        max_steps=20,
        use_llm=False,
        sample_scenarios=use_dataset,
        scenario_split=split,
        seed=seed,
    )


def run_episode_heuristic(env: SprintEnv, difficulty: Difficulty, scenario_index: int | None = None) -> dict:
    agent = HeuristicAgent()
    obs = env.reset(difficulty=difficulty, scenario_index=scenario_index)
    done = False
    total_reward = 0.0

    while not done:
        action = agent.act(obs)
        if action is None:
            break
        obs, reward, done, _ = env.step(action)
        total_reward += reward

    result = grade(env)
    return {
        "reward": total_reward,
        "scenario_id": env.state().get("scenario_id", ""),
        **result,
    }


def run_episode_ddqn(env: SprintEnv, agent: DDQNAgent, difficulty: Difficulty, scenario_index: int | None = None) -> dict:
    obs = env.reset(difficulty=difficulty, scenario_index=scenario_index)
    done = False
    total_reward = 0.0

    while not done:
        action = agent.act(obs, deterministic=True)
        if action is None:
            break
        obs, reward, done, _ = env.step(action)
        total_reward += reward

    result = grade(env)
    return {
        "reward": total_reward,
        "scenario_id": env.state().get("scenario_id", ""),
        **result,
    }


def evaluate_agent(
    agent_name: str,
    run_fn,
    difficulties: List[Difficulty],
    n_episodes: int,
    env: SprintEnv,
) -> Dict[str, dict]:
    results: Dict[str, dict] = {}

    for diff in difficulties:
        scores = []
        rewards = []
        adaptability = []
        scenario_ids = []
        scenario_count = get_scenario_count(diff, split=env.scenario_split)

        for run_idx in range(n_episodes):
            run = run_fn(diff, run_idx % scenario_count)
            scores.append(run["score"])
            rewards.append(run["reward"])
            adaptability.append(run.get("breakdown", {}).get("adaptability", 0.0))
            scenario_ids.append(run.get("scenario_id", ""))

        results[diff.value] = {
            "score_mean": round(float(np.mean(scores)), 3),
            "score_std": round(float(np.std(scores)), 3),
            "score_min": round(float(np.min(scores)), 3),
            "score_max": round(float(np.max(scores)), 3),
            "reward_mean": round(float(np.mean(rewards)), 3),
            "adaptability_mean": round(float(np.mean(adaptability)), 3),
            "scenarios": sorted({scenario_id for scenario_id in scenario_ids if scenario_id}),
        }
        logger.info(
            "[%s] %-8s | score=%.3f +/- %.3f | reward=%.2f | adapt=%.3f",
            agent_name,
            diff.value,
            results[diff.value]["score_mean"],
            results[diff.value]["score_std"],
            results[diff.value]["reward_mean"],
            results[diff.value]["adaptability_mean"],
        )

    return results


def parse_training_curve(log_path: str = "logs/train.jsonl") -> None:
    path = Path(log_path)
    if not path.exists():
        logger.info("No training log found at %s", log_path)
        return

    entries = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not entries:
        return

    episodes = [entry["episode"] for entry in entries]
    mean_scores = [entry.get("mean_score_50", 0.0) for entry in entries]
    epsilons = [entry.get("epsilon", 0.0) for entry in entries]

    curve_path = Path("logs/train_curve.csv")
    rows = ["episode,mean_score_50,epsilon"]
    rows.extend(f"{ep},{score:.4f},{eps:.4f}" for ep, score, eps in zip(episodes, mean_scores, epsilons))
    curve_path.write_text("\n".join(rows), encoding="utf-8")
    logger.info("Training curve saved to %s", curve_path)

    logger.info("")
    logger.info("Training curve (mean score over last 50 episodes):")
    max_val = max(mean_scores) if mean_scores else 1.0
    step = max(1, len(episodes) // 20)
    for index in range(0, len(episodes), step):
        bar_len = int(mean_scores[index] / max(max_val, 0.01) * 30)
        bar = "#" * bar_len + "." * (30 - bar_len)
        logger.info("  ep%4d [%s] %.3f", episodes[index], bar, mean_scores[index])


def print_comparison_table(all_results: dict) -> None:
    diffs = [diff.value for diff in Difficulty]
    header = f"{'Agent':12s} | " + " | ".join(f"{diff:8s}" for diff in diffs) + " | MEAN"

    print("\n" + "=" * len(header))
    print("  EVALUATION RESULTS")
    print("=" * len(header))
    print("  " + header)
    print("  " + "-" * (len(header) - 2))

    for agent_name, diff_results in all_results.items():
        scores = [diff_results.get(diff, {}).get("score_mean", 0.0) for diff in diffs]
        overall = float(np.mean(scores))
        row = f"{agent_name:12s} | " + " | ".join(f"{score:.3f}   " for score in scores) + f" | {overall:.3f}"
        print("  " + row)

    print("=" * len(header) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Sprint Planning agents")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best", help="DDQN checkpoint directory")
    parser.add_argument("--episodes", type=int, default=10, help="Episodes per difficulty per agent")
    parser.add_argument("--report-only", action="store_true", help="Only parse training log")
    parser.add_argument(
        "--scenario-source",
        choices=["auto", "fallback", "dataset-train", "dataset-eval"],
        default="auto",
        help="Choose evaluation scenario source",
    )
    args = parser.parse_args()

    parse_training_curve()
    if args.report_only:
        return

    has_dataset = dataset_available()
    if args.scenario_source == "auto":
        use_dataset = has_dataset
        split = "eval" if has_dataset else None
    elif args.scenario_source == "fallback":
        use_dataset = False
        split = None
    elif args.scenario_source == "dataset-train":
        use_dataset = True
        split = "train"
    else:
        use_dataset = True
        split = "eval"

    env = build_env(use_dataset=use_dataset, split=split, seed=99)
    difficulties = list(Difficulty)
    all_results: Dict[str, dict] = {}

    logger.info("")
    logger.info("Evaluating Heuristic agent...")
    all_results["Heuristic"] = evaluate_agent(
        agent_name="Heuristic",
        run_fn=lambda diff, idx: run_episode_heuristic(env, diff, scenario_index=idx),
        difficulties=difficulties,
        n_episodes=args.episodes,
        env=env,
    )

    checkpoint = Path(args.checkpoint)
    if (checkpoint / "online.pkl").exists():
        logger.info("")
        logger.info("Evaluating DDQN agent from %s...", checkpoint)
        ddqn = DDQNAgent()
        ddqn.load(str(checkpoint))
        all_results["DDQN"] = evaluate_agent(
            agent_name="DDQN",
            run_fn=lambda diff, idx: run_episode_ddqn(env, ddqn, diff, scenario_index=idx),
            difficulties=difficulties,
            n_episodes=args.episodes,
            env=env,
        )
    else:
        logger.warning("No DDQN checkpoint found at %s", checkpoint)

    print_comparison_table(all_results)

    report_path = Path("logs/eval_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "scenario_source": args.scenario_source,
        "resolved_source": {
            "use_dataset": use_dataset,
            "split": split or "fallback",
        },
        "results": all_results,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Full report saved to %s", report_path)


if __name__ == "__main__":
    main()
