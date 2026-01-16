"""
Majority Voting Agent for FEVER

Runs 3 independent ReAct traces and uses simple majority voting
to determine the final answer. Ties default to "NOT ENOUGH INFO".
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from datetime import datetime

from collections import Counter
from fever_utils import (
    run_single_trace,
    extract_final_answer_from_trace_string,
    append_to_json,
    get_next_run_number,
    WEBTHINK_PROMPT_TEMPLATE
)


def majority_vote(answers):
    """
    Apply majority voting to a list of answers.
    
    Args:
        answers: List of answer strings
        
    Returns:
        The majority answer, or "NOT ENOUGH INFO" if there's a tie
    """
    if not answers:
        return "NOT ENOUGH INFO"
    
    # Count occurrences
    counter = Counter(answers)
    most_common = counter.most_common()
    
    # Check if there's a clear majority (at least 2 out of 3)
    if len(most_common) > 0 and most_common[0][1] >= 2:
        return most_common[0][0]
    
    # Tie case: all different or no clear majority
    return "NOT ENOUGH INFO"


def run_majority_voting(idx, prompt_template=None, to_print=True, num_traces=3):
    """
    Run Multi-Trace ReAct with majority voting.
    
    Args:
        idx: Question index from FEVER dataset
        prompt_template: Custom prompt template (uses default if None)
        to_print: Whether to print progress during execution
        num_traces: Number of independent traces to run (default: 3)
        
    Returns:
        Tuple of (reward, info_dict) where:
        - reward: EM score of the voted answer
        - info_dict: Dictionary with aggregated results and individual traces
    """
    if prompt_template is None:
        prompt_template = WEBTHINK_PROMPT_TEMPLATE
    
    if to_print:
        print("="*60)
        print(f"[FRAMEWORK] Majority Voting ReAct ({num_traces} Traces)")
        print("="*60)
    
    all_traces = []
    answers = []
    question_text = None
    gt_answer = None
    
    # Run multiple independent traces
    for trace_num in range(num_traces):
        if to_print:
            print(f"\n--- Trace {trace_num + 1}/{num_traces} ---")
        
        trace_info = run_single_trace(
            idx=idx,
            initial_prompt_template=prompt_template,
            to_print=to_print,
            temperature=0.7  # Use temperature for diversity
        )
        
        all_traces.append(trace_info)
        
        # Extract answer
        answer = trace_info.get('answer')
        if answer:
            answers.append(answer)
        
        # Store question text and GT answer from first trace
        if trace_num == 0:
            question_text = trace_info.get('question_text')
            gt_answer = trace_info.get('gt_answer')
        
        if to_print:
            print(f"[TRACE {trace_num + 1}] Answer: {answer}")
    
    # Apply majority voting
    if to_print:
        print(f"\n{'='*60}")
        print("[VOTING] Applying majority voting")
        print(f"[VOTES] {answers}")
    
    voted_answer = majority_vote(answers)
    
    if to_print:
        print(f"[MAJORITY] {voted_answer}")
    
    # Calculate metrics
    em_score = 1.0 if voted_answer == gt_answer else 0.0
    
    # Aggregate call counts and tokens
    total_calls = sum(t.get('n_calls', 0) for t in all_traces)
    total_badcalls = sum(t.get('n_badcalls', 0) for t in all_traces)
    total_input_tokens = sum(t.get('input_tokens', 0) for t in all_traces)
    total_output_tokens = sum(t.get('output_tokens', 0) for t in all_traces)
    
    # Prepare trace summaries (without full trajectories for space efficiency)
    trace_summaries = []
    for i, trace in enumerate(all_traces):
        trace_summaries.append({
            'trace_num': i + 1,
            'answer': trace.get('answer'),
            'em': trace.get('em', 0.0),
            'n_calls': trace.get('n_calls', 0),
            'input_tokens': trace.get('input_tokens', 0),
            'output_tokens': trace.get('output_tokens', 0)
        })
    
    info_dict = {
        'question_idx': idx,
        'question_text': question_text,
        'answer': voted_answer,
        'gt_answer': gt_answer,
        'em': em_score,
        'f1': em_score,
        'reward': em_score,
        'n_calls': total_calls,
        'n_badcalls': total_badcalls,
        'input_tokens': total_input_tokens,
        'output_tokens': total_output_tokens,
        'total_tokens': total_input_tokens + total_output_tokens,
        'num_traces_run': num_traces,
        'individual_votes': answers,
        'trace_summaries': trace_summaries,
        'full_traces': all_traces,  # Store complete traces for analysis
        'framework': 'majority_voting'
    }
    
    if to_print:
        print("="*60)
        print(f"[FINAL] Voted Answer: {voted_answer} | GT: {gt_answer} | EM: {em_score}")
        print("="*60)
    
    # Log the result to file with framework/run folder structure
    # Updated to point to results/fever/majority_voting
    framework_folder = os.path.join(os.path.dirname(__file__), '../../../results/fever/majority_voting')
    run_name = get_next_run_number(framework_folder)
    run_folder = os.path.join(framework_folder, run_name)
    os.makedirs(run_folder, exist_ok=True)
    log_file = os.path.join(run_folder, 'results.jsonl')
    append_to_json(info_dict, log_file)
    
    return em_score, info_dict


if __name__ == '__main__':
    # Test with a sample FEVER example
    print("\n[TEST] Running Majority Voting agent test\n")
    reward, info = run_majority_voting(idx=3687, to_print=True)
    print(f"\n[TEST RESULT] Reward: {reward}, Answer: {info['answer']}")
    print(f"[VOTES] {info['individual_votes']}")
