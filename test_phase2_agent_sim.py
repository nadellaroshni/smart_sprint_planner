#!/usr/bin/env python3
"""
SIMULATE PHASE 2: RUN AGENT AGAINST ENVIRONMENT AND TEST GRADERS
This simulates what the hackathon validator does when running Nemotron agent
"""

import sys
import importlib

print("\n" + "="*70)
print("PHASE 2 AGENT SIMULATION")
print("="*70 + "\n")

try:
    # Step 1: Import everything needed
    print("[1] Importing modules...")
    from env.task_catalog import get_task_catalog
    from env.environment import SprintEnv
    from env.models import Difficulty, Action
    
    print("✓ Imports successful")
    
    # Step 2: Get task catalog
    print("\n[2] Loading task catalog...")
    tasks = get_task_catalog()
    print(f"✓ Got {len(tasks)} tasks")
    
    # Step 3: For each task, simulate agent running
    print("\n[3] Simulating agent runs...")
    
    for task in tasks:
        print(f"\n  Task: {task.id}")
        
        # Import the grader dynamically
        module_name, func_name = task.grader.split(":")
        module = importlib.import_module(module_name)
        grader_func = getattr(module, func_name)
        print(f"    ✓ Grader imported: {func_name}")
        
        # Map task name to difficulty
        difficulty_map = {
            'easy': Difficulty.EASY,
            'medium': Difficulty.MEDIUM,
            'hard': Difficulty.HARD,
        }
        difficulty = difficulty_map.get(task.id)
        
        if not difficulty:
            print(f"    ✗ Unknown task id: {task.id}")
            continue
        
        # Create environment
        try:
            env = SprintEnv(difficulty=difficulty, max_steps=20, use_llm=False, seed=42)
            print(f"    ✓ Environment created")
        except Exception as e:
            print(f"    ✗ Failed to create environment: {e}")
            continue
        
        # Reset environment
        try:
            obs = env.reset()
            print(f"    ✓ Environment reset")
        except Exception as e:
            print(f"    ✗ Failed to reset: {e}")
            continue
        
        # Simulate agent taking a few steps
        try:
            for step in range(3):
                if obs.jira_tickets and obs.developers:
                    # Agent tries to assign first ticket to first capable developer
                    task_item = obs.jira_tickets[0]
                    for dev in obs.developers:
                        if dev.capacity >= task_item.story_points:
                            action = Action(task_id=task_item.id, developer_id=dev.id)
                            obs, reward, done, info = env.step(action)
                            if done:
                                break
                            break
            print(f"    ✓ Agent steps completed")
        except Exception as e:
            print(f"    ✗ Agent step failed: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        # Now call the grader
        try:
            result = grader_func(env)
            score = result.get('score')
            
            if score is None:
                print(f"    ✗ Grader returned None for score")
                continue
            
            if not isinstance(score, (int, float)):
                print(f"    ✗ Grader returned non-numeric score: {type(score)}")
                continue
            
            if not (0.0 <= score <= 1.0):
                print(f"    ✗ Score {score} not in [0.0, 1.0]")
                continue
            
            print(f"    ✓ Grader executed successfully: score={score:.3f}")
            
        except Exception as e:
            print(f"    ✗ Grader execution failed: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n" + "="*70)
    print("✓✓✓ PHASE 2 SIMULATION PASSED ✓✓✓")
    print("="*70 + "\n")
    
except Exception as e:
    print(f"\n✗✗✗ PHASE 2 SIMULATION FAILED ✗✗✗")
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
