"""
End-to-end planning pipeline from transcript or audio to assignments.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Tuple

from agent.dqn_agent import DDQNAgent
from agent.heuristic_agent import HeuristicAgent
from env.environment import SprintEnv
from env.graders import grade
from env.models import (
    Action,
    AssignmentRecommendation,
    Difficulty,
    Observation,
    PlanResponse,
)


def _load_agent(strategy: str, checkpoint: Optional[str]) -> Tuple[object, str]:
    normalized = strategy.lower()
    if normalized == "auto":
        normalized = "ddqn" if (Path(checkpoint or "checkpoints/best") / "online.pkl").exists() else "heuristic"

    if normalized == "ddqn":
        ckpt = checkpoint or "checkpoints/best"
        if (Path(ckpt) / "online.pkl").exists():
            agent = DDQNAgent()
            agent.load(ckpt)
            return agent, "ddqn"
        normalized = "heuristic"

    return HeuristicAgent(), normalized


def _choose_action(agent: object, obs: Observation) -> Optional[Action]:
    if isinstance(agent, DDQNAgent):
        return agent.act(obs, deterministic=True)
    if isinstance(agent, HeuristicAgent):
        return agent.act(obs)
    raise TypeError(f"Unsupported planning agent: {type(agent)!r}")


def generate_plan(
    *,
    difficulty: Difficulty = Difficulty.MEDIUM,
    audio_path: Optional[str] = None,
    transcript: Optional[str] = None,
    strategy: str = "auto",
    checkpoint: Optional[str] = None,
) -> PlanResponse:
    env = SprintEnv(difficulty=difficulty, max_steps=20, use_llm=True, sample_scenarios=False)
    agent, resolved_strategy = _load_agent(strategy, checkpoint)

    obs = env.reset(
        difficulty=difficulty,
        audio_path=audio_path,
        transcript_override=transcript,
    )
    initial_obs = obs.model_copy(deep=True)
    assignments: list[AssignmentRecommendation] = []
    done = False

    while not done:
        action = _choose_action(agent, obs)
        if action is None:
            break

        chosen_task = next((task for task in obs.jira_tickets if task.id == action.task_id), None)
        chosen_dev = next((dev for dev in obs.developers if dev.id == action.developer_id), None)
        obs, reward, done, info = env.step(action)

        if chosen_task is None or chosen_dev is None:
            continue

        assignments.append(
            AssignmentRecommendation(
                step=env.current_step,
                task_id=chosen_task.id,
                task_title=chosen_task.title,
                developer_id=chosen_dev.id,
                developer_name=chosen_dev.name,
                reward=reward,
                on_time=bool(info.get("on_time", True)),
                skill_match=bool(info.get("skill_match", False)),
                source_event=chosen_task.source_event,
            )
        )

    result = grade(env)
    state = env.state()
    return PlanResponse(
        strategy=resolved_strategy,
        transcript=state["meeting_text"],
        extracted_items=initial_obs.extracted_items,
        jira_tickets=initial_obs.jira_tickets,
        assignments=assignments,
        score=result["score"],
        breakdown=result["breakdown"],
        summary=result["summary"],
        final_board=env.render(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a sprint plan from transcript or audio")
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard"], default="medium")
    parser.add_argument("--audio-path", type=str, default=None)
    parser.add_argument("--transcript", type=str, default=None)
    parser.add_argument("--strategy", choices=["auto", "heuristic", "ddqn"], default="auto")
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    diff = Difficulty(args.difficulty)
    response = generate_plan(
        difficulty=diff,
        audio_path=args.audio_path,
        transcript=args.transcript,
        strategy=args.strategy,
        checkpoint=args.checkpoint,
    )
    print(json.dumps(response.model_dump(mode="json"), indent=2))


if __name__ == "__main__":
    main()
