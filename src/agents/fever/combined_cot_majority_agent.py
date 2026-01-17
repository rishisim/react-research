"""
Combined Majority Voting and CoT-SC Agent for FEVER

Optimization: Runs 3 independent ReAct traces ONCE and applies both:
1. Majority Voting (simple counting)
2. CoT-SC (LLM synthesis of trajectories)

This saves 50% of the LLM calls required for the traces compared to running them separately.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from collections import Counter
from datetime import datetime

from fever_utils import (
    run_single_trace,
    extract_trajectories_from_traces,
    synthesize_answer_with_llm,
    WEBTHINK_PROMPT_TEMPLATE
)

def majority_vote(answers):
    """
    Apply majority voting to a list of answers.
    Borrowed from majority_voting_agent.py
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

def run_combined_agent(idx, prompt_template=None, to_print=True, num_traces=3):
    """
    Run 3 independent traces and produce outputs for BOTH Majority Voting and CoT-SC.
    
    Args:
        idx: Question index from FEVER dataset
        prompt_template: Custom prompt template
        to_print: Whether to print progress
        num_traces: Number of independent traces (default 3)
        
    Returns:
        dict: containing 'majority_voting' and 'cot_sc' result dictionaries
    """
    if prompt_template is None:
        prompt_template = WEBTHINK_PROMPT_TEMPLATE
    
    if to_print:
        print("="*60)
        print(f"[FRAMEWORK] COMBINED Optimized Run (Majority Vote + CoT-SC)")
        print(f"[CONFIG] Number of traces: {num_traces}")
        print("="*60)
    
    all_traces = []
    answers = []
    question_text = None
    gt_answer = None
    
    # --- PHASE 1: Run Independent Traces (Shared) ---
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
        
        # Collect answer for Majority Voting
        answer = trace_info.get('answer')
        if answer:
            answers.append(answer)
            
        # Store metadata from first trace
        if trace_num == 0:
            question_text = trace_info.get('question_text')
            gt_answer = trace_info.get('gt_answer')
            
        if to_print:
            print(f"[TRACE {trace_num + 1}] Answer: {answer}")

    # Aggregated metrics for traces
    total_calls = sum(t.get('n_calls', 0) for t in all_traces)
    total_badcalls = sum(t.get('n_badcalls', 0) for t in all_traces)
    total_input_tokens = sum(t.get('input_tokens', 0) for t in all_traces)
    total_output_tokens = sum(t.get('output_tokens', 0) for t in all_traces)
    
    # Trace summaries
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

    # --- PHASE 2: Majority Voting Logic ---
    if to_print:
        print(f"\n{'='*60}")
        print("[LOGIC] Applying Majority Voting")
    
    mv_answer = majority_vote(answers)
    mv_em = 1.0 if mv_answer == gt_answer else 0.0
    
    mv_info_dict = {
        'question_idx': idx,
        'question_text': question_text,
        'answer': mv_answer,
        'gt_answer': gt_answer,
        'em': mv_em,
        'f1': mv_em,
        'reward': mv_em,
        'n_calls': total_calls,
        'n_badcalls': total_badcalls,
        'input_tokens': total_input_tokens,
        'output_tokens': total_output_tokens,
        'total_tokens': total_input_tokens + total_output_tokens,
        'num_traces_run': num_traces,
        'individual_votes': answers,
        'trace_summaries': trace_summaries,
        'full_traces': all_traces,
        'framework': 'majority_voting'
    }
    
    if to_print:
        print(f"[MAJORITY] Answer: {mv_answer} | EM: {mv_em}")

    # --- PHASE 3: CoT-SC Logic ---
    if to_print:
        print(f"\n{'='*60}")
        print("[LOGIC] Applying CoT-SC (LLM Synthesis)")
    
    trajectories = extract_trajectories_from_traces(all_traces)
    
    # Synthesize using LLM
    synthesis_tokens = {'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0}
    if not trajectories:
        cot_answer = "NOT ENOUGH INFO"
    else:
        cot_answer, synthesis_tokens = synthesize_answer_with_llm(trajectories, question_text)
        
    cot_em = 1.0 if cot_answer == gt_answer else 0.0
    
    # CoT-SC tokens include synthesis cost
    cot_input_tokens = total_input_tokens + synthesis_tokens['input_tokens']
    cot_output_tokens = total_output_tokens + synthesis_tokens['output_tokens']
    
    cot_info_dict = {
        'question_idx': idx,
        'question_text': question_text,
        'answer': cot_answer,
        'gt_answer': gt_answer,
        'em': cot_em,
        'f1': cot_em,
        'reward': cot_em,
        'n_calls': total_calls, # Note: Does not count the synthesis call in n_calls usually, but let's stick to standard
        'n_badcalls': total_badcalls,
        'input_tokens': cot_input_tokens,
        'output_tokens': cot_output_tokens,
        'total_tokens': cot_input_tokens + cot_output_tokens,
        'num_traces_run': num_traces,
        'trace_summaries': trace_summaries,
        'full_traces': all_traces,
        'framework': 'cot_sc'
    }
    
    if to_print:
        print(f"[COT-SC] Answer: {cot_answer} | EM: {cot_em}")
        print("="*60)
        
    return {
        'majority_voting': mv_info_dict,
        'cot_sc': cot_info_dict
    }

if __name__ == "__main__":
    # Simple test
    print("Testing Combined Agent...")
    results = run_combined_agent(3687)
    print("\nTest Complete.")
    print(f"MV EM: {results['majority_voting']['em']}")
    print(f"CoT EM: {results['cot_sc']['em']}")
