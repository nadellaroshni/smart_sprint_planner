#!/usr/bin/env python3
"""
PHASE 2 VALIDATOR SIMULATION
Exactly what the hackathon's Phase 2 validator runs.
Run this and show me the output.
"""

import sys
import os
import importlib
import traceback

print("\n" + "="*70)
print("PHASE 2 VALIDATOR SIMULATION")
print("="*70 + "\n")

# STEP 1: Import get_task_catalog
print("[1] Importing get_task_catalog...")
try:
    from env.task_catalog import get_task_catalog
    print("    ✓ Successfully imported get_task_catalog")
except Exception as e:
    print(f"    ✗ FAILED: {e}")
    traceback.print_exc()
    sys.exit(1)

# STEP 2: Call get_task_catalog() and check output
print("\n[2] Calling get_task_catalog()...")
try:
    tasks = get_task_catalog()
    print(f"    ✓ Got {len(tasks)} tasks")
    if len(tasks) < 3:
        print(f"    ✗ ERROR: Need 3+ tasks, got {len(tasks)}")
        sys.exit(1)
    print("    ✓ Has 3+ tasks")
except Exception as e:
    print(f"    ✗ FAILED: {e}")
    traceback.print_exc()
    sys.exit(1)

# STEP 3: Check each task has grader attribute
print("\n[3] Checking task.grader attributes...")
graders_to_test = []
for task in tasks:
    print(f"\n    Task: {task.id}")
    if not hasattr(task, 'grader'):
        print(f"      ✗ No 'grader' attribute!")
        sys.exit(1)
    print(f"      grader attribute: {task.grader}")
    if not task.grader:
        print(f"      ✗ grader is None/empty!")
        sys.exit(1)
    print(f"      ✓ Has grader: {task.grader}")
    graders_to_test.append((task.id, task.grader))

# STEP 4: Try to import and call each grader
print("\n[4] Importing and testing grader functions...")
graders_working = 0

for task_id, grader_string in graders_to_test:
    print(f"\n    Task: {task_id}")
    print(f"    Grader string: {grader_string}")
    
    try:
        # Parse "env.graders:grade_easy" → module="env.graders", func="grade_easy"
        if ':' not in grader_string:
            print(f"      ✗ Grader string doesn't contain ':' - invalid format")
            continue
        
        module_name, func_name = grader_string.split(":")
        print(f"      Importing {module_name}.{func_name}...")
        
        # Import the module
        module = importlib.import_module(module_name)
        print(f"      ✓ Module imported")
        
        # Get the function
        grader_func = getattr(module, func_name, None)
        if grader_func is None:
            available = [x for x in dir(module) if not x.startswith('_')]
            print(f"      ✗ Function '{func_name}' not found in module")
            print(f"      Available items: {available}")
            continue
        
        print(f"      ✓ Function imported: {grader_func}")
        
        # Now try to CREATE an environment and CALL the grader
        print(f"      Testing grader with environment...")
        try:
            from env.environment import SprintEnv
            from env.models import Difficulty
            
            # Map task_id to difficulty
            difficulty_map = {
                'easy': Difficulty.EASY,
                'medium': Difficulty.MEDIUM,
                'hard': Difficulty.HARD,
            }
            difficulty = difficulty_map.get(task_id)
            if not difficulty:
                print(f"      ✗ Unknown task_id: {task_id}")
                continue
            
            # Create environment
            env = SprintEnv(difficulty=difficulty, max_steps=20, use_llm=False, seed=42)
            obs = env.reset()
            print(f"      ✓ Environment created and reset")
            
            # Call the grader
            result = grader_func(env)
            score = result.get('score')
            
            if score is None:
                print(f"      ✗ Grader returned None for score")
                continue
            
            if not isinstance(score, (int, float)):
                print(f"      ✗ Grader returned non-numeric score: {type(score)}")
                continue
            
            if not (0.0 <= score <= 1.0):
                print(f"      ✗ Score {score} not in [0.0, 1.0]")
                continue
            
            print(f"      ✓ Grader callable and returns valid score: {score}")
            graders_working += 1
            
        except Exception as e:
            print(f"      ✗ Error calling grader: {e}")
            traceback.print_exc()
            continue
        
    except Exception as e:
        print(f"      ✗ Error: {e}")
        traceback.print_exc()
        continue

# FINAL RESULT
print("\n" + "="*70)
print("RESULT SUMMARY")
print("="*70)
print(f"Tasks found: {len(tasks)}")
print(f"Graders working: {graders_working}/3")

if graders_working == 3:
    print("\n✓✓✓ VALIDATOR SHOULD PASS ✓✓✓")
    sys.exit(0)
else:
    print(f"\n✗✗✗ VALIDATOR WILL FAIL - only {graders_working}/3 graders working ✗✗✗")
    sys.exit(1)
