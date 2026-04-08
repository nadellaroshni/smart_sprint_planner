#!/usr/bin/env python3
"""
Smoke test matching Phase 2 validator expectations.
Tests that the environment has 3 tasks with proper graders.
"""

import sys
from env.environment import SprintEnv
from env.graders import grade_easy, grade_medium, grade_hard
from env.models import Difficulty
from env.task_catalog import get_task_catalog


def test_task_catalog():
    """Test 1: Verify task catalog has 3 tasks with graders."""
    print("\n" + "=" * 70)
    print("TEST 1: Task Catalog")
    print("=" * 70)
    
    tasks = get_task_catalog()
    print(f"Found {len(tasks)} tasks:")
    
    if len(tasks) < 3:
        print(f"❌ FAIL: Expected 3+ tasks, got {len(tasks)}")
        return False
    
    for task in tasks:
        has_grader = task.grader is not None and len(task.grader) > 0
        status = "✓" if has_grader else "❌"
        print(f"  {status} {task.id:10} | grader={task.grader}")
        if not has_grader:
            return False
    
    print("✓ PASS: 3 tasks with graders defined")
    return True


def test_grader_execution(difficulty, grader_func):
    """Test 2: Execute a single grader and verify score is in [0.0, 1.0]."""
    try:
        env = SprintEnv(difficulty=difficulty, max_steps=20, use_llm=False)
        obs = env.reset()
        
        # Take at least one step to have some state
        if obs.jira_tickets and obs.developers:
            action_task = obs.jira_tickets[0]
            action_dev = obs.developers[0]
            if action_dev.capacity >= action_task.story_points:
                from env.models import Action
                action = Action(task_id=action_task.id, developer_id=action_dev.id)
                obs, reward, done, info = env.step(action)
        
        # Grade the environment
        result = grader_func(env)
        score = result.get("score")
        
        # Validate score
        if score is None:
            print(f"  ❌ {difficulty.value}: grader returned None for score")
            return False
        
        if not isinstance(score, (int, float)):
            print(f"  ❌ {difficulty.value}: score is not numeric (got {type(score).__name__})")
            return False
        
        if not (0.0 <= score <= 1.0):
            print(f"  ❌ {difficulty.value}: score {score} outside [0.0, 1.0]")
            return False
        
        print(f"  ✓ {difficulty.value:8} | score={score:.3f} | deterministic")
        return True
    
    except Exception as e:
        print(f"  ❌ {difficulty.value}: {type(e).__name__}: {e}")
        return False


def test_all_graders():
    """Test 2: Verify all 3 graders return valid scores."""
    print("\n" + "=" * 70)
    print("TEST 2: Grader Execution")
    print("=" * 70)
    
    graders = [
        (Difficulty.EASY, grade_easy),
        (Difficulty.MEDIUM, grade_medium),
        (Difficulty.HARD, grade_hard),
    ]
    
    results = []
    for difficulty, grader_func in graders:
        result = test_grader_execution(difficulty, grader_func)
        results.append(result)
    
    if all(results):
        print("✓ PASS: All 3 graders execute and return valid scores")
        return True
    else:
        print("❌ FAIL: Some graders failed")
        return False


def test_determinism():
    """Test 3: Verify same input produces same score (deterministic)."""
    print("\n" + "=" * 70)
    print("TEST 3: Determinism Check")
    print("=" * 70)
    
    try:
        # Run same environment setup twice
        scores_1 = []
        scores_2 = []
        
        for run in range(2):
            for difficulty, grader_func in [
                (Difficulty.EASY, grade_easy),
                (Difficulty.MEDIUM, grade_medium),
                (Difficulty.HARD, grade_hard),
            ]:
                env = SprintEnv(difficulty=difficulty, max_steps=20, use_llm=False, seed=42)
                env.reset()
                score = grader_func(env).get("score")
                
                if run == 0:
                    scores_1.append(score)
                else:
                    scores_2.append(score)
        
        # Check if scores match
        if scores_1 == scores_2:
            print(f"  ✓ Scores consistent across runs:")
            for i, (s1, s2) in enumerate(zip(scores_1, scores_2)):
                print(f"    Run 1: {s1:.3f} | Run 2: {s2:.3f}")
            print("✓ PASS: Deterministic scoring verified")
            return True
        else:
            print(f"❌ Score mismatch: {scores_1} vs {scores_2}")
            return False
    
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def main():
    """Run all smoke tests."""
    print("\n" + "=" * 70)
    print("PHASE 2 SMOKE TEST - Smart Sprint Planner")
    print("=" * 70)
    
    results = []
    
    # Test 1: Catalog
    results.append(("Task Catalog", test_task_catalog()))
    
    # Test 2: Graders
    results.append(("Grader Execution", test_all_graders()))
    
    # Test 3: Determinism
    results.append(("Determinism", test_determinism()))
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    for name, passed in results:
        status = "✓ PASS" if passed else "❌ FAIL"
        print(f"  {status}: {name}")
    
    all_passed = all(r[1] for r in results)
    
    print("=" * 70)
    if all_passed:
        print("✓ ALL TESTS PASSED - Ready for Phase 2 validation!")
        return 0
    else:
        print("❌ SOME TESTS FAILED - Fix issues before submission")
        return 1


if __name__ == "__main__":
    sys.exit(main())
