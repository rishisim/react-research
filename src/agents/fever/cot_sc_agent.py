"""
CoT-SC (Chain-of-Thought Self-Consistency) Agent for FEVER

Runs multiple independent ReAct traces and synthesizes the final answer
using an LLM to evaluate all reasoning trajectories.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from datetime import datetime

from fever_utils import (
    run_single_trace,
    extract_trajectories_from_traces,
    synthesize_answer_with_llm,
    append_to_json,
    get_next_run_number,
    WEBTHINK_PROMPT_TEMPLATE
)


def run_cot_sc(idx, prompt_template=None, to_print=True, num_traces=3):
    """
    Run Multi-Trace ReAct with Chain-of-Thought Self-Consistency.
    
    Uses an LLM to synthesize the final answer by reviewing all reasoning
    trajectories, not just counting votes.
    
    Args:
        idx: Question index from FEVER dataset
        prompt_template: Custom prompt template (uses default if None)
        to_print: Whether to print progress during execution
        num_traces: Number of independent traces to run (default: 3)
        
    Returns:
        Tuple of (reward, info_dict) where:
        - reward: EM score of the synthesized answer
        - info_dict: Dictionary with aggregated results and individual traces
    """
    if prompt_template is None:
        prompt_template = WEBTHINK_PROMPT_TEMPLATE
    
    if to_print:
        print("="*60)
        print(f"[FRAMEWORK] CoT-SC (Multi-Trace + LLM Synthesis)")
        print(f"[CONFIG] Number of traces: {num_traces}")
        print("="*60)
    
    all_traces = []
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
        
        # Store question text and GT answer from first trace
        if trace_num == 0:
            question_text = trace_info.get('question_text')
            gt_answer = trace_info.get('gt_answer')
        
        if to_print:
            print(f"[TRACE {trace_num + 1}] Answer: {trace_info.get('answer')}")
    
    # Extract trajectories for synthesis
    if to_print:
        print(f"\n{'='*60}")
        print("[SYNTHESIS] Extracting trajectories for LLM synthesis")
    
    trajectories = extract_trajectories_from_traces(all_traces)
    
    if to_print:
        print(f"[SYNTHESIS] Extracted {len(trajectories)} trajectories")
        print("[SYNTHESIS] Calling LLM to synthesize final answer...")
    
    # Synthesize answer using LLM
    synthesis_tokens = {'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0}
    if not trajectories:
        synthesized_answer = "NOT ENOUGH INFO"
        if to_print:
            print("[WARNING] No trajectories extracted, defaulting to NOT ENOUGH INFO")
    else:
        synthesized_answer, synthesis_tokens = synthesize_answer_with_llm(trajectories, question_text)
    
    if to_print:
        print(f"[SYNTHESIZED] {synthesized_answer}")
    
    # Calculate metrics
    em_score = 1.0 if synthesized_answer == gt_answer else 0.0
    
    # Aggregate call counts and tokens
    total_calls = sum(t.get('n_calls', 0) for t in all_traces)
    total_badcalls = sum(t.get('n_badcalls', 0) for t in all_traces)
    total_input_tokens = sum(t.get('input_tokens', 0) for t in all_traces)
    total_output_tokens = sum(t.get('output_tokens', 0) for t in all_traces)
    
    # Add synthesis call tokens
    total_input_tokens += synthesis_tokens['input_tokens']
    total_output_tokens += synthesis_tokens['output_tokens']
    
    # Prepare trace summaries
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
        'answer': synthesized_answer,
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
        'trace_summaries': trace_summaries,
        'full_traces': all_traces,
        'framework': 'cot_sc'
    }
    
    if to_print:
        print("="*60)
        print(f"[FINAL] Synthesized Answer: {synthesized_answer} | GT: {gt_answer} | EM: {em_score}")
        print("="*60)
    
    # Log the result to file with framework/run folder structure
    framework_folder = os.path.join(os.path.dirname(__file__), '../../../results/fever/cot_sc')
    run_name = get_next_run_number(framework_folder)
    run_folder = os.path.join(framework_folder, run_name)
    os.makedirs(run_folder, exist_ok=True)
    log_file = os.path.join(run_folder, 'results.jsonl')
    append_to_json(info_dict, log_file)
    
    return em_score, info_dict


if __name__ == '__main__':
    # Test with a sample FEVER example
    print("\n[TEST] Running CoT-SC agent test\n")
    reward, info = run_cot_sc(idx=3687, to_print=True, num_traces=3)
    print(f"\n[TEST RESULT] Reward: {reward}, Answer: {info['answer']}")
