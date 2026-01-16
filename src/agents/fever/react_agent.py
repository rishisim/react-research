"""
ReAct Agent for FEVER

Standard single-trace ReAct agent for fact verification.
Executes one reasoning trace to verify a claim.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fever_utils import (
    run_single_trace,
    WEBTHINK_PROMPT_TEMPLATE
)


def run_react(idx, prompt_template=None, to_print=True):
    """
    Run standard ReAct agent with a single trace.
    
    Args:
        idx: Question index from FEVER dataset
        prompt_template: Custom prompt template (uses default if None)
        to_print: Whether to print progress during execution
        
    Returns:
        Tuple of (reward, info_dict) where:
        - reward: EM score (0.0 or 1.0)
        - info_dict: Dictionary with keys:
            - question_idx: Index of the question
            - question_text: The claim text
            - answer: Agent's answer (SUPPORTS/REFUTES/NOT ENOUGH INFO)
            - gt_answer: Ground truth answer
            - em: Exact match score
            - f1: F1 score (same as EM for FEVER)
            - reward: Same as EM
            - n_calls: Number of LLM calls
            - n_badcalls: Number of failed LLM calls
            - traj: Full trajectory string
    """
    if prompt_template is None:
        prompt_template = WEBTHINK_PROMPT_TEMPLATE
    
    if to_print:
        print("="*60)
        print("[FRAMEWORK] Standard ReAct (Single Trace)")
        print("="*60)
    
    # Run single trace with temperature=0.0 (deterministic)
    trace_info = run_single_trace(
        idx=idx,
        initial_prompt_template=prompt_template,
        to_print=to_print,
        temperature=0.0
    )
    
    # Calculate reward
    reward = trace_info.get('em', 0.0)
    
    # Prepare info dict with standardized format
    info_dict = {
        'question_idx': trace_info.get('question_idx'),
        'question_text': trace_info.get('question_text'),
        'answer': trace_info.get('answer'),
        'gt_answer': trace_info.get('gt_answer'),
        'em': trace_info.get('em', 0.0),
        'f1': trace_info.get('em', 0.0),  # For FEVER, F1 = EM
        'reward': reward,
        'n_calls': trace_info.get('n_calls', 0),
        'n_badcalls': trace_info.get('n_badcalls', 0),
        'input_tokens': trace_info.get('input_tokens', 0),
        'output_tokens': trace_info.get('output_tokens', 0),
        'total_tokens': trace_info.get('total_tokens', 0),
        'traj': trace_info.get('traj', ''),
        'framework': 'react'
    }
    
    if to_print:
        print("="*60)
        print(f"[FINAL] Answer: {info_dict['answer']} | GT: {info_dict['gt_answer']} | EM: {info_dict['em']}")
        print("="*60)
    
    return reward, info_dict


if __name__ == '__main__':
    # Test with a sample FEVER example
    print("\n[TEST] Running ReAct agent test\n")
    reward, info = run_react(idx=3687, to_print=True)
    print(f"\n[TEST RESULT] Reward: {reward}, Answer: {info['answer']}")
