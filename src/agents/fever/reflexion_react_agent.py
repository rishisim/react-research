"""
Reflexion ReAct Agent for FEVER

Runs 2 traces with reflexion:
1. Initial trace
2. Generate reflexion from initial trace
3. Second trace with reflexion as context
4. Final answer = second trace answer (no synthesis)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import json

from fever_utils import (
    run_single_trace,
    llm,
    WEBTHINK_PROMPT_TEMPLATE
)


def load_reflexion_prompt():
    """Load the reflexion prompt from JSON file."""
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        prompt_path = os.path.join(script_dir, 'prompts', 'fever_reflexion.json')
        with open(prompt_path, 'r') as f:
            reflexion_prompts = json.load(f)
        return reflexion_prompts['reflexion']
    except (FileNotFoundError, KeyError) as e:
        print(f"[WARNING] Could not load reflexion prompt: {e}")
        return "Reflect on your previous attempt and create a new plan."


def generate_reflexion(trajectory_info, reflexion_prompt, to_print=False):
    """
    Generate a reflexion based on the trajectory information.
    
    Args:
        trajectory_info: Dictionary with trajectory information
        reflexion_prompt: The reflexion prompt template
        to_print: Whether to print the reflexion
        
    Returns:
        Reflexion string (plan for next attempt)
    """
    # Extract trajectory string
    if isinstance(trajectory_info, dict):
        trajectory_str = trajectory_info.get('traj', '')
        if not trajectory_str:
            # Fallback: reconstruct from available info
            question = trajectory_info.get('question_text', '')
            answer = trajectory_info.get('answer', '')
            trajectory_str = f"Claim: {question}\n[Details not available]\nFinal Answer: {answer}"
    else:
        trajectory_str = str(trajectory_info)
    
    # Create full prompt
    full_prompt = f"{reflexion_prompt}\n\n{trajectory_str}\nPlan:"
    
    # Generate reflexion using LLM
    try:
        reflexion_response = llm(full_prompt, stop=["\n"], num_traces=1)
        
        # Extract the plan from the response
        if "Plan:" in reflexion_response:
            plan = reflexion_response.split("Plan:")[-1].strip()
        else:
            plan = reflexion_response.strip()
        
        if to_print:
            print(f"[REFLEXION] {plan}")
        
        return plan
    except Exception as e:
        if to_print:
            print(f"[ERROR] Error generating reflexion: {e}")
        return "Continue with a more systematic approach to gather evidence."


def run_reflexion_react(idx, prompt_template=None, to_print=True):
    """
    Run Reflexion ReAct with 2 traces.
    
    Process:
    1. Run initial trace
    2. Generate reflexion from trace 1
    3. Run second trace with reflexion as context
    4. Return answer from trace 2 (no synthesis)
    
    Args:
        idx: Question index from FEVER dataset
        prompt_template: Custom prompt template (uses default if None)
        to_print: Whether to print progress during execution
        
    Returns:
        Tuple of (reward, info_dict) where:
        - reward: EM score of the second trace
        - info_dict: Dictionary with both traces and reflexion
    """
    if prompt_template is None:
        prompt_template = WEBTHINK_PROMPT_TEMPLATE
    
    # Load reflexion prompt
    reflexion_prompt = load_reflexion_prompt()
    
    if to_print:
        print("="*60)
        print("[FRAMEWORK] Reflexion ReAct (2 Traces)")
        print("="*60)
    
    # --- Trace 1: Initial attempt ---
    if to_print:
        print("\n--- Trace 1: Initial Attempt ---")
    
    trace_1 = run_single_trace(
        idx=idx,
        initial_prompt_template=prompt_template,
        to_print=to_print,
        temperature=0.0
    )
    
    question_text = trace_1.get('question_text')
    gt_answer = trace_1.get('gt_answer')
    
    if to_print:
        print(f"[TRACE 1] Answer: {trace_1.get('answer')}")
    
    # --- Generate Reflexion ---
    if to_print:
        print(f"\n{'='*60}")
        print("[REFLEXION] Generating reflexion from Trace 1...")
    
    reflexion = generate_reflexion(trace_1, reflexion_prompt, to_print=to_print)
    
    # --- Trace 2: With reflexion context ---
    if to_print:
        print(f"\n{'='*60}")
        print("--- Trace 2: With Reflexion Context ---")
    
    # Modify prompt with reflexion
    modified_prompt = prompt_template + f"Plans from past attempts: {reflexion}\n\n"
    
    trace_2 = run_single_trace(
        idx=idx,
        initial_prompt_template=modified_prompt,
        to_print=to_print,
        temperature=0.0
    )
    
    if to_print:
        print(f"[TRACE 2] Answer: {trace_2.get('answer')}")
    
    # --- Final answer from trace 2 (no synthesis) ---
    final_answer = trace_2.get('answer')
    em_score = 1.0 if final_answer == gt_answer else 0.0
    
    # Aggregate call counts
    total_calls = trace_1.get('n_calls', 0) + trace_2.get('n_calls', 0)
    total_badcalls = trace_1.get('n_badcalls', 0) + trace_2.get('n_badcalls', 0)
    
    info_dict = {
        'question_idx': idx,
        'question_text': question_text,
        'answer': final_answer,
        'gt_answer': gt_answer,
        'em': em_score,
        'f1': em_score,
        'reward': em_score,
        'n_calls': total_calls,
        'n_badcalls': total_badcalls,
        'num_traces_run': 2,
        'reflexion': reflexion,
        'trace_1': {
            'answer': trace_1.get('answer'),
            'em': trace_1.get('em', 0.0),
            'n_calls': trace_1.get('n_calls', 0)
        },
        'trace_2': {
            'answer': trace_2.get('answer'),
            'em': trace_2.get('em', 0.0),
            'n_calls': trace_2.get('n_calls', 0)
        },
        'full_trace_1': trace_1,
        'full_trace_2': trace_2,
        'framework': 'reflexion_react'
    }
    
    if to_print:
        print("="*60)
        print(f"[FINAL] Answer (from Trace 2): {final_answer} | GT: {gt_answer} | EM: {em_score}")
        print("="*60)
    
    return em_score, info_dict


if __name__ == '__main__':
    # Test with a sample FEVER example
    print("\n[TEST] Running Reflexion ReAct agent test\n")
    reward, info = run_reflexion_react(idx=3687, to_print=True)
    print(f"\n[TEST RESULT] Reward: {reward}, Answer: {info['answer']}")
    print(f"[REFLEXION] {info['reflexion']}")
