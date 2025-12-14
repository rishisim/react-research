#!/usr/bin/env python
"""
Quick test to verify FEVER agent fixes work.
Tests that observations are populated with actual Wikipedia data.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from react_agent import run_react

print("\n" + "="*70)
print("TESTING: ReAct Agent with FEVER Example 3687")
print("Claim: Paramore is not from Tennessee")
print("="*70 + "\n")

# Run the agent
reward, info = run_react(idx=3687, to_print=True)

# Check results
print("\n" + "="*70)
print("TEST RESULTS")
print("="*70)
print(f"Answer: {info['answer']}")
print(f"Ground Truth: {info['gt_answer']}")
print(f"EM Score: {info['em']}")
print(f"LLM Calls: {info['n_calls']}")

# Verify observations were populated
traj = info.get('traj', '')
if 'Observation 1:' in traj:
    obs_text = traj.split('Observation 1:')[1].split('\n')[0].strip()
    if len(obs_text) > 10:
        print(f"\n✓ Observations are populated!")
        print(f"  First observation preview: {obs_text[:150]}...")
    else:
        print(f"\n✗ Observation is EMPTY (problem not fixed)")
else:
    print(f"\n✗ No observation found in trajectory")

print("="*70)
