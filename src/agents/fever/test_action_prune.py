"""
Test Action Prune ReAct Agent on 10 FEVER examples
"""

import sys
import os
import json
import random
from datetime import datetime
from pathlib import Path

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from action_prune_react_agent import run_action_prune_react

def main():
    """Run 10 tests of Action Prune ReAct agent"""
    
    # Test parameters
    num_examples = 10
    seed = 42
    random.seed(seed)
    
    # Generate 10 random indices from FEVER dev set (7405 examples)
    test_indices = random.sample(range(7405), num_examples)
    
    print("\n" + "="*70)
    print("FEVER ACTION PRUNE REACT - 10 EXAMPLES TEST")
    print("="*70)
    print(f"Seed: {seed}")
    print(f"Test indices: {test_indices}")
    print("="*70 + "\n")
    
    all_results = []
    correct = 0
    total_calls = 0
    
    for i, test_idx in enumerate(test_indices, 1):
        print("\n" + "="*70)
        print(f"[{i}/{num_examples}] Testing example {test_idx}")
        print("="*70)
        
        # Run the agent
        reward, info = run_action_prune_react(idx=test_idx, to_print=True)
        
        all_results.append(info)
        correct += info['em']
        total_calls += info['n_calls']
        
        print(f"\n[Result {i}] Answer: {info['answer']} | GT: {info['gt_answer']} | EM: {info['em']}")
    
    # Calculate metrics
    accuracy = correct / num_examples
    avg_calls = total_calls / num_examples
    
    # Prepare summary results
    summary = {
        'test_info': {
            'timestamp': datetime.now().isoformat(),
            'framework': 'action_prune_react',
            'num_examples': num_examples,
            'seed': seed,
            'test_indices': test_indices
        },
        'metrics': {
            'accuracy': accuracy,
            'correct': correct,
            'total': num_examples,
            'avg_llm_calls': avg_calls,
            'total_calls': total_calls
        },
        'results': all_results
    }
    
    # Save results to action_prune folder
    script_dir = Path(__file__).parent
    results_dir = (script_dir / "../../../results/fever/action_prune").resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = results_dir / f"action_prune_n{num_examples}_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "="*70)
    print("ALL TESTS COMPLETE")
    print("="*70)
    print(f"Accuracy: {accuracy:.2%} ({correct}/{num_examples})")
    print(f"Average LLM calls: {avg_calls:.1f}")
    print(f"Total LLM calls: {total_calls}")
    print(f"Results saved to: {output_file}")
    print("="*70 + "\n")
    
    return summary

if __name__ == '__main__':
    main()
