#!/usr/bin/env python3
"""Quick FEVEROUS test: 2 questions with ReAct and Nexus agents."""

import sys
import os

# Add agent paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src/agents/feverous'))

from react_agent import run_react
from nexus_wrapper import run_nexus

def main():
    test_indices = [0, 1]  # Two questions
    
    print("\n" + "="*70)
    print("FEVEROUS AGENT TEST - 2 Questions with ReAct and Nexus")
    print("="*70)
    
    results = []
    
    for idx in test_indices:
        print(f"\n{'#'*70}")
        print(f"# QUESTION {idx}")
        print(f"{'#'*70}")
        
        # Test ReAct
        print("\n--- Running ReAct ---")
        try:
            react_reward, react_info = run_react(idx=idx, to_print=True)
            results.append({
                'idx': idx,
                'framework': 'react',
                'answer': react_info['answer'],
                'gt': react_info['gt_answer'],
                'em': react_info['em']
            })
        except Exception as e:
            print(f"ReAct Error: {e}")
            results.append({'idx': idx, 'framework': 'react', 'error': str(e)})
        
        # Test Nexus
        print("\n--- Running Nexus ---")
        try:
            nexus_reward, nexus_info = run_nexus(idx=idx, to_print=True)
            results.append({
                'idx': idx,
                'framework': 'nexus',
                'answer': nexus_info['answer'],
                'gt': nexus_info['gt_answer'],
                'em': nexus_info['em']
            })
        except Exception as e:
            print(f"Nexus Error: {e}")
            results.append({'idx': idx, 'framework': 'nexus', 'error': str(e)})
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"{'Idx':<5} {'Framework':<10} {'Answer':<20} {'GT':<20} {'EM':<5}")
    print("-"*70)
    for r in results:
        if 'error' in r:
            print(f"{r['idx']:<5} {r['framework']:<10} ERROR: {r['error'][:40]}")
        else:
            print(f"{r['idx']:<5} {r['framework']:<10} {r['answer']:<20} {r['gt']:<20} {r['em']:<5}")
    
    print("\n✅ Test complete!")

if __name__ == '__main__':
    main()
