#!/usr/bin/env python3
"""
SIMULATE HACKATHON PHASE 2 VALIDATOR
This mimics EXACTLY what the validator does when it clones the repo
"""

import sys
import os
import tempfile
import subprocess
import importlib

print("\n" + "="*70)
print("PHASE 2 VALIDATOR SIMULATION - FRESH CLONE TEST")
print("="*70 + "\n")

# Create a temp directory and clone the repo there
with tempfile.TemporaryDirectory() as tmpdir:
    print(f"[1] Cloning repo to temporary directory: {tmpdir}")
    
    clone_result = subprocess.run(
        ["git", "clone", "https://github.com/nadellaroshni/smart_sprint_planner.git", "."],
        cwd=tmpdir,
        capture_output=True,
        text=True,
        timeout=60
    )
    
    if clone_result.returncode != 0:
        print(f"✗ Clone failed: {clone_result.stderr}")
        sys.exit(1)
    
    print("✓ Repository cloned successfully")
    
    # Add the cloned repo to sys.path
    sys.path.insert(0, tmpdir)
    
    # Now try to import what the validator imports
    print(f"\n[2] Attempting phase 2 validator imports...")
    
    try:
        # Try to get task catalog
        from env.task_catalog import get_task_catalog
        print("✓ Imported get_task_catalog")
        
        # Get tasks
        tasks = get_task_catalog()
        print(f"✓ Got {len(tasks)} tasks")
        
        # For each task, try to import the grader
        print(f"\n[3] Testing grader imports for each task...")
        graders_found = 0
        
        for task in tasks:
            print(f"\n  Task: {task.id}")
            print(f"  Grader spec: {task.grader}")
            
            if not task.grader:
                print(f"  ✗ No grader specified!")
                continue
            
            try:
                # Parse the grader spec
                module_name, func_name = task.grader.split(":")
                print(f"  Importing {module_name}:{func_name}...")
                
                # Dynamic import
                module = importlib.import_module(module_name)
                grader_func = getattr(module, func_name)
                
                print(f"  ✓ Grader imported successfully")
                graders_found += 1
                
            except Exception as e:
                print(f"  ✗ Failed to import grader: {e}")
                import traceback
                traceback.print_exc()
        
        print(f"\n[4] RESULT: {graders_found}/{len(tasks)} graders found")
        
        if graders_found == 3:
            print("\n✓✓✓ VALIDATOR SHOULD PASS ✓✓✓")
        else:
            print(f"\n✗✗✗ VALIDATOR WILL FAIL ✗✗✗")
            sys.exit(1)
        
    except Exception as e:
        print(f"✗ Import failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

print("\n" + "="*70)
