#!/usr/bin/env python3
"""Simulate exactly what Phase 2 validator does to find graders"""

import sys
import importlib

# Simulate what validator does
from env.task_catalog import get_task_catalog

tasks = get_task_catalog()
print(f"Tasks found: {len(tasks)}\n")

graders_found = 0

for task in tasks:
    print(f"Task: {task.id}")
    print(f"  Grader string: {task.grader}")
    
    if not task.grader:
        print(f"  ✗ NO GRADER!")
        continue
    
    try:
        # Parse module:func
        parts = task.grader.split(":")
        if len(parts) != 2:
            print(f"  ✗ Invalid grader format: {task.grader}")
            continue
        
        module_path, func_name = parts
        print(f"  Importing {module_path} → {func_name}")
        
        # Import module
        module = importlib.import_module(module_path)
        print(f"  ✓ Module imported")
        
        # Get function
        grader_func = getattr(module, func_name, None)
        if grader_func is None:
            print(f"  ✗ Function {func_name} not found in {module_path}")
            print(f"  Available: {[x for x in dir(module) if not x.startswith('_')]}")
            continue
        
        print(f"  ✓ Function found: {grader_func}")
        graders_found += 1
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print()

print(f"═" * 50)
print(f"RESULT: {graders_found}/{len(tasks)} graders successfully imported")
if graders_found == len(tasks):
    print("✓ VALIDATOR WILL PASS")
else:
    print(f"✗ VALIDATOR WILL FAIL - missing {len(tasks) - graders_found} graders")
