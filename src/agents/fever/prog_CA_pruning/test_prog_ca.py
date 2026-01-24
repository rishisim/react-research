"""
Test script for Programmatic Combined Action & Context Pruning agent.

Tests the pruner on a small sample of FEVER examples with detailed logging.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from prog_ca_pruning_agent import run_prog_ca_pruning_react
import json


def test_single_example(idx: int, verbose: bool = True):
    """
    Test the agent on a single FEVER example.
    
    Args:
        idx: Index of FEVER example to test
        verbose: Whether to print detailed output
        
    Returns:
        Dictionary with results
    """
    print("\n" + "="*80)
    print(f"[TEST] Running on example {idx}")
    print("="*80 + "\n")
    
    reward, info = run_prog_ca_pruning_react(idx=idx, to_print=verbose)
    
    # Print summary
    print("\n" + "="*80)
    print("[TEST SUMMARY]")
    print("="*80)
    print(f"Index: {info['question_idx']}")
    print(f"Question: {info['question_text']}")
    print(f"Answer: {info['answer']}")
    print(f"Ground Truth: {info['gt_answer']}")
    print(f"EM Score: {info['em']}")
    print(f"Reward: {reward}")
    print(f"\n[EFFICIENCY METRICS]")
    print(f"LLM Calls: {info['n_calls']}")
    print(f"Bad Calls: {info['n_badcalls']}")
    print(f"Input Tokens: {info['input_tokens']}")
    print(f"Output Tokens: {info['output_tokens']}")
    print(f"Total Tokens: {info['total_tokens']}")
    
    if 'pruning' in info:
        pruning = info['pruning']
        print(f"\n[PRUNING STATS]")
        if 'action_pruning' in pruning:
            ap = pruning['action_pruning']
            print(f"Actions Pruned: {ap.get('total_pruned', 0)}")
            if ap.get('pruned_actions'):
                print(f"Pruning Reasons:")
                for prune_info in ap['pruned_actions'][:3]:
                    print(f"  - Step {prune_info.get('step', '?')}: {prune_info.get('reason', '?')}")
        
        if 'context_pruning' in pruning:
            cp = pruning['context_pruning']
            print(f"Evidence Items Kept: {cp.get('evidence_items', 0)}")
            print(f"Visited Pages: {cp.get('visited_pages', 0)}")
            print(f"Observations Retained: {cp.get('observations_retained', 0)}")
            print(f"Failures Tracked: {cp.get('failures_tracked', 0)}")
    
    print("="*80 + "\n")
    
    return info


def test_batch(indices: list, output_file: str = None):
    """
    Test agent on multiple examples.
    
    Args:
        indices: List of indices to test
        output_file: Optional file to save results to
    """
    results = []
    
    for idx in indices:
        try:
            result = test_single_example(idx, verbose=False)
            results.append(result)
        except Exception as e:
            print(f"\n[ERROR] Failed on index {idx}: {e}")
            results.append({
                'question_idx': idx,
                'error': str(e),
                'em': 0.0,
            })
    
    # Print aggregate stats
    print("\n" + "="*80)
    print("[AGGREGATE RESULTS]")
    print("="*80)
    
    em_scores = [r.get('em', 0.0) for r in results if 'em' in r]
    total_tokens = sum(r.get('total_tokens', 0) for r in results if 'total_tokens' in r)
    total_calls = sum(r.get('n_calls', 0) for r in results if 'n_calls' in r)
    
    avg_em = sum(em_scores) / len(em_scores) if em_scores else 0.0
    avg_tokens_per_example = total_tokens / len(results) if results else 0
    avg_calls_per_example = total_calls / len(results) if results else 0
    
    print(f"Examples Tested: {len(results)}")
    print(f"Average EM: {avg_em:.2%}")
    print(f"Total Tokens: {total_tokens:,}")
    print(f"Avg Tokens/Example: {avg_tokens_per_example:.0f}")
    print(f"Avg LLM Calls/Example: {avg_calls_per_example:.1f}")
    
    total_pruned = sum(
        r.get('pruning', {}).get('action_pruning', {}).get('total_pruned', 0)
        for r in results if 'pruning' in r
    )
    total_evidence = sum(
        r.get('pruning', {}).get('context_pruning', {}).get('evidence_items', 0)
        for r in results if 'pruning' in r
    )
    
    if total_pruned > 0 or total_evidence > 0:
        print(f"\n[PRUNING AGGREGATE]")
        print(f"Total Actions Pruned: {total_pruned}")
        print(f"Total Evidence Items Kept: {total_evidence}")
    
    print("="*80 + "\n")
    
    # Save results if requested
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"[SAVE] Results saved to {output_file}\n")
    
    return results


if __name__ == '__main__':
    # Test on a few examples
    test_indices = [3687, 3688, 3689]  # Small sample for testing
    
    print("\n[PROG-CA-PRUNING TESTS]")
    print("="*80)
    print("Testing Programmatic Combined Action & Context Pruning")
    print("="*80)
    
    results = test_batch(test_indices)
