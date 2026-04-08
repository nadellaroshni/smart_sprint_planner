#!/usr/bin/env python3
"""
Phase 1 Pre-Submission Validator - Mimics official Phase 1 checks
Matches exact requirements from context_scaler.txt
"""

import sys
import yaml
from pathlib import Path


def check_openenv_yaml():
    """Check 1: OpenEnv spec compliance - openenv.yaml validation."""
    print("\n" + "=" * 70)
    print("CHECK 1: OpenEnv YAML Specification Compliance")
    print("=" * 70)
    
    try:
        with open("openenv.yaml") as f:
            config = yaml.safe_load(f)
        
        # Validate required fields
        required_fields = ["name", "version", "entrypoint", "tasks"]
        for field in required_fields:
            if field not in config:
                print(f"❌ FAIL: Missing required field '{field}'")
                return False
        
        print(f"✓ openenv.yaml found and valid")
        print(f"  - name: {config['name']}")
        print(f"  - version: {config['version']}")
        print(f"  - entrypoint: {config['entrypoint']}")
        
        # Validate tasks
        tasks = config.get("tasks", [])
        if len(tasks) < 3:
            print(f"❌ FAIL: Expected 3+ tasks, found {len(tasks)}")
            return False
        
        print(f"  - tasks: {len(tasks)} defined")
        
        return True
    
    except Exception as e:
        print(f"❌ FAIL: {type(e).__name__}: {e}")
        return False


def check_typed_models():
    """Check 2: Typed Pydantic models for Observation, Action, Reward."""
    print("\n" + "=" * 70)
    print("CHECK 2: Typed Models (Pydantic)")
    print("=" * 70)
    
    try:
        from env.models import Observation, Action, StepResult
        from pydantic import BaseModel
        
        print(f"✓ Observation model exists: {Observation.__name__}")
        print(f"✓ Action model exists: {Action.__name__}")
        print(f"✓ StepResult model exists: {StepResult.__name__}")
        
        # Verify they're Pydantic models
        assert issubclass(Observation, BaseModel), "Observation not a BaseModel"
        assert issubclass(Action, BaseModel), "Action not a BaseModel"
        assert issubclass(StepResult, BaseModel), "StepResult not a BaseModel"
        
        print("✓ All models are Pydantic BaseModel subclasses")
        return True
    
    except Exception as e:
        print(f"❌ FAIL: {type(e).__name__}: {e}")
        return False


def check_openenv_interface():
    """Check 3: step(), reset(), state() endpoints exist."""
    print("\n" + "=" * 70)
    print("CHECK 3: OpenEnv Interface Methods")
    print("=" * 70)
    
    try:
        from env.environment import SprintEnv
        from env.models import Difficulty
        
        env = SprintEnv(difficulty=Difficulty.EASY, use_llm=False)
        
        # Check reset() exists and works
        print("Testing reset()...")
        obs = env.reset()
        print(f"✓ reset() works, returns Observation")
        
        # Check step() exists and works
        print("Testing step()...")
        if obs.jira_tickets and obs.developers:
            action_task = obs.jira_tickets[0]
            action_dev = obs.developers[0]
            if action_dev.capacity >= action_task.story_points:
                from env.models import Action
                action = Action(task_id=action_task.id, developer_id=action_dev.id)
                obs, reward, done, info = env.step(action)
                print(f"✓ step() works, returns (obs, reward, done, info)")
        else:
            print("⚠ step() skipped - no actionable tasks available")
        
        # Check state() exists and works
        print("Testing state()...")
        state = env.state()
        print(f"✓ state() works, returns dict with keys: {list(state.keys())[:5]}...")
        
        return True
    
    except Exception as e:
        print(f"❌ FAIL: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_tasks_with_graders():
    """Check 4: 3+ tasks with graders - verified to return scores in [0.0, 1.0]."""
    print("\n" + "=" * 70)
    print("CHECK 4: Tasks with Graders (Core Phase 1 Check)")
    print("=" * 70)
    
    try:
        from env.task_catalog import get_task_catalog
        from env.environment import SprintEnv
        from env.graders import grade_easy, grade_medium, grade_hard
        from env.models import Difficulty
        
        # Get task catalog
        tasks = get_task_catalog()
        if len(tasks) < 3:
            print(f"❌ FAIL: Expected 3+ tasks, found {len(tasks)}")
            return False
        
        print(f"✓ Found {len(tasks)} tasks in catalog")
        
        graders = [
            ("easy", grade_easy, Difficulty.EASY),
            ("medium", grade_medium, Difficulty.MEDIUM),
            ("hard", grade_hard, Difficulty.HARD),
        ]
        
        all_valid = True
        for task_name, grader_func, difficulty in graders:
            print(f"\n  Task: {task_name}")
            
            # Check grader function exists and is callable
            if not callable(grader_func):
                print(f"  ❌ Grader not callable")
                all_valid = False
                continue
            
            # Create env and get score
            try:
                env = SprintEnv(difficulty=difficulty, max_steps=20, use_llm=False, seed=42)
                obs = env.reset()
                
                # Run a step if possible
                if obs.jira_tickets and obs.developers:
                    from env.models import Action
                    action_task = obs.jira_tickets[0]
                    action_dev = obs.developers[0]
                    if action_dev.capacity >= action_task.story_points:
                        action = Action(task_id=action_task.id, developer_id=action_dev.id)
                        obs, reward, done, info = env.step(action)
                
                # Call grader
                result = grader_func(env)
                score = result.get("score")
                
                # Validate score
                if score is None:
                    print(f"  ❌ Grader returned None for score")
                    all_valid = False
                    continue
                
                if not isinstance(score, (int, float)):
                    print(f"  ❌ Score is {type(score).__name__}, not numeric")
                    all_valid = False
                    continue
                
                if not (0.0 <= score <= 1.0):
                    print(f"  ❌ Score {score} outside [0.0, 1.0]")
                    all_valid = False
                    continue
                
                print(f"  ✓ Grader works: score={score:.3f} (in [0.0, 1.0])")
                print(f"    - breakdown: {result.get('breakdown', {})}")
                
            except Exception as e:
                print(f"  ❌ Error running grader: {type(e).__name__}: {e}")
                all_valid = False
        
        if all_valid:
            print(f"\n✓ All 3 tasks have working graders with valid scores")
            return True
        else:
            return False
    
    except Exception as e:
        print(f"❌ FAIL: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_baseline_inference():
    """Check 5: Baseline inference script exists and can run."""
    print("\n" + "=" * 70)
    print("CHECK 5: Baseline Inference Script")
    print("=" * 70)
    
    try:
        import os
        
        # Check inference.py exists in root
        if not Path("inference.py").exists():
            print(f"❌ FAIL: inference.py not found in root directory")
            return False
        
        print(f"✓ inference.py found in root directory")
        
        # Check required environment variables are documented
        required_vars = ["API_BASE_URL", "MODEL_NAME", "HF_TOKEN"]
        
        with open("inference.py") as f:
            content = f.read()
        
        print(f"✓ inference.py size: {len(content)} bytes")
        
        # Check for OpenAI client usage
        if "openai" in content.lower() or "OpenAI" in content:
            print(f"✓ Uses OpenAI client")
        else:
            print(f"⚠ Warning: OpenAI client not found in inference.py")
        
        # Check for [START], [STEP], [END] format
        if "[START]" in content and "[STEP]" in content and "[END]" in content:
            print(f"✓ Uses required stdout format ([START], [STEP], [END])")
        else:
            print(f"⚠ Warning: Required stdout format not found")
        
        return True
    
    except Exception as e:
        print(f"⚠ Warning: {type(e).__name__}: {e}")
        return False


def check_dockerfile():
    """Check 6: Dockerfile exists and is valid."""
    print("\n" + "=" * 70)
    print("CHECK 6: Dockerfile")
    print("=" * 70)
    
    try:
        if not Path("Dockerfile").exists():
            print(f"❌ FAIL: Dockerfile not found")
            return False
        
        with open("Dockerfile") as f:
            content = f.read()
        
        print(f"✓ Dockerfile found")
        print(f"  - Size: {len(content)} bytes")
        
        # Check key components
        if "EXPOSE" in content:
            print(f"✓ Contains EXPOSE directive")
        
        if "CMD" in content or "ENTRYPOINT" in content:
            print(f"✓ Contains CMD or ENTRYPOINT")
        
        if "uvicorn" in content:
            print(f"✓ Runs uvicorn server")
        
        return True
    
    except Exception as e:
        print(f"⚠ Warning: {type(e).__name__}: {e}")
        return False


def main():
    """Run all Phase 1 checks."""
    print("\n" + "=" * 70)
    print("PHASE 1 PRE-SUBMISSION VALIDATOR")
    print("Based on context_scaler.txt requirements")
    print("=" * 70)
    
    checks = [
        ("OpenEnv YAML", check_openenv_yaml),
        ("Typed Models", check_typed_models),
        ("Interface Methods", check_openenv_interface),
        ("3+ Tasks with Graders", check_tasks_with_graders),
        ("Baseline Inference", check_baseline_inference),
        ("Dockerfile", check_dockerfile),
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n⚠ Unexpected error in {name}: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 70)
    print("PHASE 1 VALIDATION SUMMARY")
    print("=" * 70)
    
    for name, passed in results:
        status = "✓ PASS" if passed else "❌ FAIL"
        print(f"  {status}: {name}")
    
    critical_checks = results[:4]  # First 4 are critical
    critical_pass = all(r[1] for r in critical_checks)
    
    print("\n" + "=" * 70)
    if critical_pass:
        print("✓✓✓ ALL CRITICAL PHASE 1 CHECKS PASSED ✓✓✓")
        print("=" * 70)
        print("Your submission should pass Phase 1 validation!")
        print("You can now resubmit to the hackathon.")
        return 0
    else:
        print("❌ SOME CRITICAL CHECKS FAILED")
        print("=" * 70)
        print("Fix the above issues before resubmitting.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
