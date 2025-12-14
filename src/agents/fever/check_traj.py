#!/usr/bin/env python
"""Check what's in the trajectory for the Paramore question"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from react_agent import run_react

print("Testing index 3687 (Paramore question)")
print("="*70)

r, info = run_react(idx=3687, to_print=False)

print(f"Question: {info['question_text']}")
print(f"Answer: {info['answer']}")
print(f"GT Answer: {info['gt_answer']}")
print(f"\nTrajectory:")
print("="*70)
print(info['traj'])
print("="*70)
