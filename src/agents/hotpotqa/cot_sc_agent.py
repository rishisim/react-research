"""
CoT-SC (Chain-of-Thought Self-Consistency) Agent for HotPotQA

Runs multiple independent ReAct traces and synthesizes the final answer
using an LLM to evaluate all reasoning trajectories.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from hotpotqa_utils import (
    run_single_trace,
    extract_trajectories_from_traces,
    synthesize_answer_with_llm,
    llm_judge_answer,
    get_hotpotqa_env,
    WEBTHINK_PROMPT_TEMPLATE
)


def run_cot_sc(idx, prompt_template=None, to_print=True, num_traces=3):
    """
    Run Multi-Trace ReAct with Chain-of-Thought Self-Consistency.
    
    Uses an LLM to synthesize the final answer by reviewing all reasoning
    trajectories, not just counting votes.
    
    Args:
        idx: Question index from HotPotQA dataset
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
    if not trajectories:
        synthesized_answer = "null"
        if to_print:
            print("[WARNING] No trajectories extracted, defaulting to null")
    else:
        synthesized_answer = synthesize_answer_with_llm(trajectories, question_text)
    
    if to_print:
        print(f"[SYNTHESIZED] {synthesized_answer}")
    
    # Get metrics using environment
    hotpotqa_env = get_hotpotqa_env()
    # Unwrap to get the underlying env with get_metrics
    inner_env = hotpotqa_env
    while hasattr(inner_env, 'env') and not hasattr(inner_env, 'get_metrics'):
        inner_env = inner_env.env
    
    if hasattr(inner_env, 'get_metrics'):
        metrics = inner_env.get_metrics({'answer': synthesized_answer})
    else:
        metrics = {'em': 0, 'f1': 0, 'reward': 0}
    
    em_score = metrics.get('em', 0.0)
    f1_score = metrics.get('f1', 0.0)
    
    # LLM-as-judge evaluation on synthesized answer
    llm_eval = llm_judge_answer(question_text, synthesized_answer, gt_answer)
    
    # Aggregate call counts
    total_calls = sum(t.get('n_calls', 0) for t in all_traces)
    total_badcalls = sum(t.get('n_badcalls', 0) for t in all_traces)
    
    # Prepare trace summaries
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
        'answer': synthesized_answer,
        'gt_answer': gt_answer,
        'em': em_score,
        'f1': f1_score,
        'reward': em_score,
        'n_calls': total_calls,
        'n_badcalls': total_badcalls,
        'num_traces_run': num_traces,
        'trace_summaries': trace_summaries,
        'full_traces': all_traces,
        'llm_correct': llm_eval['llm_correct'],
        'llm_explanation': llm_eval['llm_explanation'],
        'framework': 'cot_sc'
    }
    
    if to_print:
        print("="*60)
        print(f"[FINAL] Synthesized Answer: {synthesized_answer} | GT: {gt_answer} | EM: {em_score} | LLM: {llm_eval['llm_correct']}")
        print("="*60)
    
    return em_score, info_dict


if __name__ == '__main__':
    # Test with a sample HotPotQA example
    print("\n[TEST] Running CoT-SC agent test\n")
    reward, info = run_cot_sc(idx=0, to_print=True, num_traces=3)
    print(f"\n[TEST RESULT] Reward: {reward}, Answer: {info['answer']}")
