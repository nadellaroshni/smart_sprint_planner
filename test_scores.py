#!/usr/bin/env python3
"""Quick test of new grader scores."""

from env.environment import SprintEnv
from env.graders import grade_easy, grade_medium, grade_hard
from env.models import Difficulty, Action

print("Testing new hard task grader fix:")
print()

scores = {}
for difficulty, grader_func, name in [
    (Difficulty.EASY, grade_easy, "easy"),
    (Difficulty.MEDIUM, grade_medium, "medium"),
    (Difficulty.HARD, grade_hard, "hard"),
]:
    env = SprintEnv(difficulty=difficulty, max_steps=20, use_llm=False, seed=42)
    obs = env.reset()
    
    # Take a step if possible
    if obs.jira_tickets and obs.developers:
        for task in obs.jira_tickets:
            for dev in obs.developers:
                if dev.capacity >= task.story_points:
                    action = Action(task_id=task.id, developer_id=dev.id)
                    obs, reward, done, info = env.step(action)
                    break
    
    result = grader_func(env)
    score = result["score"]
    scores[name] = score
    print(f"{name:8} = {score:.3f}  {result['breakdown']}")

print()
print("=" * 70)
print("Score Comparison:")
print("=" * 70)
print(f"easy   = {scores['easy']:.3f}")
print(f"medium = {scores['medium']:.3f}")
print(f"hard   = {scores['hard']:.3f}")
print()

if scores['hard'] > scores['easy']:
    print("SUCCESS: Hard > Easy")
else:
    print("FAIL: Hard <= Easy")

if scores['hard'] > scores['medium'] or scores['hard'] >= scores['medium'] - 0.01:
    print("SUCCESS: Hard >= Medium")
else:
    print("FAIL: Hard < Medium")

print()
print("All scores are different:", len(set(scores.values())) == 3)
