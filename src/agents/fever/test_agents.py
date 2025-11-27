"""
Quick test script to verify all FEVER agents work correctly.

Tests each agent framework with a single example.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from react_agent import run_react
from reflexion_react_agent import run_reflexion_react
from majority_voting_agent import run_majority_voting
from cot_sc_agent import run_cot_sc


def test_agent(agent_name, agent_func, idx=3687):
    """Test a single agent framework."""
    print(f"\n{'='*70}")
    print(f"TESTING: {agent_name}")
    print(f"{'='*70}\n")
    
    try:
        reward, info = agent_func(idx=idx, to_print=True)
        
        print(f"\n[TEST RESULT] {agent_name}")
        print(f"  Question Index: {info.get('question_idx')}")
        print(f"  Answer: {info.get('answer')}")
        print(f"  Ground Truth: {info.get('gt_answer')}")
        print(f"  EM Score: {info.get('em')}")
        print(f"  LLM Calls: {info.get('n_calls')}")
        print(f"  Status: SUCCESS")
        
        return True
        
    except Exception as e:
        print(f"\n[TEST RESULT] {agent_name}")
        print(f"  Status: FAILED")
        print(f"  Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all agent tests."""
    print("\n" + "="*70)
    print("FEVER AGENT FRAMEWORK TEST SUITE")
    print("="*70)
    print("Testing all 4 agent frameworks with a sample FEVER example")
    print("="*70)
    
    test_idx = 3687  # Known example from FEVER dev set
    
    results = {}
    
    # Test 1: ReAct
    results['react'] = test_agent("ReAct (Single Trace)", run_react, test_idx)
    
    # Test 2: Reflexion ReAct
    results['reflexion'] = test_agent("Reflexion ReAct (2 Traces)", run_reflexion_react, test_idx)
    
    # Test 3: Majority Voting
    results['majority_voting'] = test_agent("Majority Voting (3 Traces)", run_majority_voting, test_idx)
    
    # Test 4: CoT-SC
    results['cot_sc'] = test_agent("CoT-SC (3 Traces + LLM Synthesis)", run_cot_sc, test_idx)
    
    # Summary
    print(f"\n{'='*70}")
    print("TEST SUMMARY")
    print(f"{'='*70}")
    
    total = len(results)
    passed = sum(1 for success in results.values() if success)
    failed = total - passed
    
    for name, success in results.items():
        status = "[PASS]" if success else "[FAIL]"
        print(f"  {status} {name}")
    
    print(f"\n  Total: {total} | Passed: {passed} | Failed: {failed}")
    
    if failed == 0:
        print("\n  All tests passed successfully!")
    else:
        print(f"\n  WARNING: {failed} test(s) failed!")
    
    print("="*70 + "\n")
    
    return failed == 0


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
