"""
Majority Voting Agent for HotPotQA

Runs 3 independent ReAct traces and uses LLM-based semantic majority voting
to determine the final answer. Handles free-form answers by grouping 
semantically equivalent responses.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from datetime import datetime

from hotpotqa_utils import (
    run_single_trace,
    majority_vote_semantic,
    llm_judge_answer,
    get_hotpotqa_env,
    append_to_json,
    get_next_run_number,
    WEBTHINK_PROMPT_TEMPLATE
)


def run_majority_voting(idx, prompt_template=None, to_print=True, num_traces=3):
    """
    Run Multi-Trace ReAct with semantic majority voting.
    
    Args:
        idx: Question index from HotPotQA dataset
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
    
    # Apply semantic majority voting
    if to_print:
        print(f"\n{'='*60}")
        print("[VOTING] Applying semantic majority voting")
        print(f"[VOTES] {answers}")
    
    voted_answer = majority_vote_semantic(answers, question_text)
    
    if to_print:
        print(f"[MAJORITY] {voted_answer}")
    
    # Get metrics using environment
    hotpotqa_env = get_hotpotqa_env()
    inner_env = hotpotqa_env
    while hasattr(inner_env, 'env') and not hasattr(inner_env, 'get_metrics'):
        inner_env = inner_env.env
    
    if hasattr(inner_env, 'get_metrics'):
        metrics = inner_env.get_metrics({'answer': voted_answer})
    else:
        metrics = {'em': 0, 'f1': 0, 'reward': 0}
    
    em_score = metrics.get('em', 0.0)
    f1_score = metrics.get('f1', 0.0)
    
    # LLM-as-judge evaluation
    llm_eval = llm_judge_answer(question_text, voted_answer, gt_answer)
    
    # Aggregate call counts
    total_calls = sum(t.get('n_calls', 0) for t in all_traces)
    total_badcalls = sum(t.get('n_badcalls', 0) for t in all_traces)
    
    # Prepare trace summaries (without full trajectories for space efficiency)
    trace_summaries = []
    for i, trace in enumerate(all_traces):
        trace_summaries.append({
            'trace_num': i + 1,
            'answer': trace.get('answer'),
            'em': trace.get('em', 0.0),
            'n_calls': trace.get('n_calls', 0)
        })
    
    info_dict = {
        'question_idx': idx,
        'question_text': question_text,
        'answer': voted_answer,
        'gt_answer': gt_answer,
        'em': em_score,
        'f1': f1_score,
        'reward': em_score,
        'n_calls': total_calls,
        'n_badcalls': total_badcalls,
        'num_traces_run': num_traces,
        'individual_votes': answers,
        'trace_summaries': trace_summaries,
        'full_traces': all_traces,
        'llm_correct': llm_eval['llm_correct'],
        'llm_explanation': llm_eval['llm_explanation'],
        'framework': 'majority_voting'
    }
    
    if to_print:
        print("="*60)
        print(f"[FINAL] Voted Answer: {voted_answer} | GT: {gt_answer} | EM: {em_score} | LLM: {llm_eval['llm_correct']}")
        print("="*60)
    
    # Log the result to file with framework/run folder structure
    framework_folder = os.path.join(os.path.dirname(__file__), '../../../results/hotpotqa/majority_voting')
    run_name = get_next_run_number(framework_folder)
    run_folder = os.path.join(framework_folder, run_name)
    os.makedirs(run_folder, exist_ok=True)
    log_file = os.path.join(run_folder, 'results.jsonl')
    append_to_json(info_dict, log_file)
    
    return em_score, info_dict


if __name__ == '__main__':
    # Test with a sample HotPotQA example
    print("\n[TEST] Running Majority Voting agent test\n")
    reward, info = run_majority_voting(idx=0, to_print=True)
    print(f"\n[TEST RESULT] Reward: {reward}, Answer: {info['answer']}")
    print(f"[VOTES] {info['individual_votes']}")
