"""
Example usage of FEVER agent frameworks.

This script demonstrates how to use each framework individually.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import all frameworks
from react_agent import run_react
from reflexion_react_agent import run_reflexion_react
from majority_voting_agent import run_majority_voting
from cot_sc_agent import run_cot_sc


def example_react():
    """Example: Run standard ReAct agent."""
    print("\n" + "="*70)
    print("EXAMPLE: Standard ReAct Agent")
    print("="*70 + "\n")
    
    idx = 3687  # Sample FEVER question index
    reward, info = run_react(idx=idx, to_print=True)
    
    print(f"\nReward: {reward}")
    print(f"Answer: {info['answer']}")
    print(f"Ground Truth: {info['gt_answer']}")


def example_reflexion():
    """Example: Run Reflexion ReAct agent."""
    print("\n" + "="*70)
    print("EXAMPLE: Reflexion ReAct Agent (2 Traces)")
    print("="*70 + "\n")
    
    idx = 3687
    reward, info = run_reflexion_react(idx=idx, to_print=True)
    
    print(f"\nReward: {reward}")
    print(f"Final Answer (from Trace 2): {info['answer']}")
    print(f"Trace 1 Answer: {info['trace_1']['answer']}")
    print(f"Reflexion: {info['reflexion'][:100]}...")


def example_majority_voting():
    """Example: Run Majority Voting agent."""
    print("\n" + "="*70)
    print("EXAMPLE: Majority Voting Agent (3 Traces)")
    print("="*70 + "\n")
    
    idx = 3687
    reward, info = run_majority_voting(idx=idx, to_print=True, num_traces=3)
    
    print(f"\nReward: {reward}")
    print(f"Individual Votes: {info['individual_votes']}")
    print(f"Majority Vote: {info['answer']}")


def example_cot_sc():
    """Example: Run CoT-SC agent."""
    print("\n" + "="*70)
    print("EXAMPLE: CoT-SC Agent (3 Traces + LLM Synthesis)")
    print("="*70 + "\n")
    
    idx = 3687
    reward, info = run_cot_sc(idx=idx, to_print=True, num_traces=3)
    
    print(f"\nReward: {reward}")
    print(f"Synthesized Answer: {info['answer']}")
    print(f"Individual Trace Answers: {[t['answer'] for t in info['trace_summaries']]}")


if __name__ == '__main__':
    """
    Run all examples.
    
    Note: This will make multiple LLM calls and may take a few minutes.
    Each framework is demonstrated with the same FEVER example.
    """
    
    print("\n" + "="*70)
    print("FEVER AGENT FRAMEWORKS - USAGE EXAMPLES")
    print("="*70)
    print("\nThis script demonstrates how to use each of the 4 frameworks.")
    print("Press Ctrl+C to stop at any time.\n")
    
    try:
        # Run examples
        example_react()
        
        input("\nPress Enter to continue to Reflexion example...")
        example_reflexion()
        
        input("\nPress Enter to continue to Majority Voting example...")
        example_majority_voting()
        
        input("\nPress Enter to continue to CoT-SC example...")
        example_cot_sc()
        
        print("\n" + "="*70)
        print("All examples completed!")
        print("="*70 + "\n")
        
    except KeyboardInterrupt:
        print("\n\nExamples interrupted by user.")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
