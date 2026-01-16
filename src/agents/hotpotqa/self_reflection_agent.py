"""
Self-Reflection Agent for HotPotQA

Runs a single ReAct trace, then uses LLM to verify the answer based on the trace.
If the verification determines the answer is incorrect, outputs the corrected answer
based solely on the evidence in the trace.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import json
from datetime import datetime

from hotpotqa_utils import (
    run_single_trace,
    llm,
    llm_judge_answer,
    get_hotpotqa_env,
    append_to_json,
    get_next_run_number,
    WEBTHINK_PROMPT_TEMPLATE
)


def load_self_reflection_prompt():
    """Load the self-reflection verification prompt from JSON file."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    prompt_path = os.path.join(script_dir, 'prompts', 'hotpotqa_self_reflection.json')
    try:
        with open(prompt_path, 'r', encoding='utf-8') as f:
            prompts = json.load(f)
        return prompts.get('self_reflection_verification', '')
    except Exception as e:
        print(f"[WARNING] Could not load self-reflection prompt: {e}")
        return ""


def verify_and_correct_answer(trajectory_info, verification_prompt, to_print=False):
    """
    Verify if the answer in the trajectory is correct using LLM, and provide correction if needed.
    
    Args:
        trajectory_info: Dictionary with trajectory information
        verification_prompt: The verification prompt template
        to_print: Whether to print verification output
        
    Returns:
        Dictionary with:
        - verification_status: 'CORRECT' or 'INCORRECT'
        - verification_reasoning: Explanation from the LLM
        - corrected_answer: The final answer (original if correct, corrected if incorrect)
        - input_tokens, output_tokens, total_tokens: Token usage
    """
    question = trajectory_info.get('question_text', '')
    trajectory = trajectory_info.get('traj', '')
    answer = trajectory_info.get('answer', '')
    
    # Build the verification prompt
    full_prompt = f"""{verification_prompt}

Now verify this trajectory:

Question: "{question}"

{trajectory}

The agent's final answer is: {answer}

Provide your verification:"""
    
    response, token_usage = llm(full_prompt, stop=[], num_traces=1)
    
    # Parse the response
    verification_status = 'CORRECT'
    if 'INCORRECT' in response.upper():
        verification_status = 'INCORRECT'
    
    # Extract corrected answer if provided
    corrected_answer = answer  # Default to original
    if verification_status == 'INCORRECT':
        # Look for "Final Answer:" in response
        if 'Final Answer:' in response:
            parts = response.split('Final Answer:')
            if len(parts) > 1:
                corrected_answer = parts[-1].strip().split('\n')[0].strip()
    
    if to_print:
        print(f"\n[SELF-REFLECTION] Status: {verification_status}")
        print(f"[SELF-REFLECTION] Original: {answer}")
        if verification_status == 'INCORRECT':
            print(f"[SELF-REFLECTION] Corrected: {corrected_answer}")
    
    return {
        'verification_status': verification_status,
        'verification_reasoning': response.strip(),
        'corrected_answer': corrected_answer,
        'input_tokens': token_usage['input_tokens'],
        'output_tokens': token_usage['output_tokens'],
        'total_tokens': token_usage['total_tokens']
    }


def run_self_reflection(idx, prompt_template=None, to_print=True):
    """
    Run Self-Reflection agent with single trace and verification.
    
    Process:
    1. Run a single ReAct trace
    2. Verify the answer using LLM based on the trace
    3. If INCORRECT, use the corrected answer from verification
    4. Return final answer
    
    Args:
        idx: Question index from HotPotQA dataset
        prompt_template: Custom prompt template (uses default if None)
        to_print: Whether to print progress during execution
        
    Returns:
        Tuple of (reward, info_dict) where:
        - reward: EM score of the final answer
        - info_dict: Dictionary with trace and verification results
    """
    if prompt_template is None:
        prompt_template = WEBTHINK_PROMPT_TEMPLATE
    
    if to_print:
        print("="*60)
        print("[FRAMEWORK] Self-Reflection (Single Trace + Verification)")
        print("="*60)
    
    # Load verification prompt
    verification_prompt = load_self_reflection_prompt()
    
    # Run single trace
    if to_print:
        print("\n--- ReAct Trace ---")
    
    trace_info = run_single_trace(
        idx=idx,
        initial_prompt_template=prompt_template,
        to_print=to_print,
        temperature=0.0  # Deterministic
    )
    
    question_text = trace_info.get('question_text', '')
    gt_answer = trace_info.get('gt_answer', '')
    initial_answer = trace_info.get('answer', '')
    
    if to_print:
        print(f"\n[INITIAL] Answer: {initial_answer}")
    
    # Verify and potentially correct the answer
    if to_print:
        print("\n--- Self-Reflection Phase ---")
    
    verification_result = verify_and_correct_answer(trace_info, verification_prompt, to_print=to_print)
    
    # Determine final answer
    final_answer = verification_result['corrected_answer']
    answer_was_corrected = (verification_result['verification_status'] == 'INCORRECT')
    
    # Get metrics for final answer
    hotpotqa_env = get_hotpotqa_env()
    inner_env = hotpotqa_env
    while hasattr(inner_env, 'env') and not hasattr(inner_env, 'get_metrics'):
        inner_env = inner_env.env
    
    if hasattr(inner_env, 'get_metrics'):
        metrics = inner_env.get_metrics({'answer': final_answer})
    else:
        metrics = {'em': 0, 'f1': 0, 'reward': 0}
    
    em_score = metrics.get('em', 0.0)
    f1_score = metrics.get('f1', 0.0)
    
    # LLM-as-judge evaluation on final answer
    llm_eval = llm_judge_answer(question_text, final_answer, gt_answer)
    
    # Calculate total calls and tokens
    total_calls = trace_info.get('n_calls', 0) + 1  # +1 for verification
    total_badcalls = trace_info.get('n_badcalls', 0)
    total_input_tokens = trace_info.get('input_tokens', 0) + verification_result.get('input_tokens', 0)
    total_output_tokens = trace_info.get('output_tokens', 0) + verification_result.get('output_tokens', 0)
    
    # Add LLM judge tokens
    total_input_tokens += llm_eval.get('judge_input_tokens', 0)
    total_output_tokens += llm_eval.get('judge_output_tokens', 0)
    
    info_dict = {
        'question_idx': idx,
        'question_text': question_text,
        'initial_answer': initial_answer,
        'answer': final_answer,
        'gt_answer': gt_answer,
        'answer_was_corrected': answer_was_corrected,
        'em': em_score,
        'f1': f1_score,
        'reward': em_score,
        'n_calls': total_calls,
        'n_badcalls': total_badcalls,
        'input_tokens': total_input_tokens,
        'output_tokens': total_output_tokens,
        'total_tokens': total_input_tokens + total_output_tokens,
        'verification': {
            'verification_status': verification_result['verification_status'],
            'verification_reasoning': verification_result['verification_reasoning']
        },
        'traj': trace_info.get('traj', ''),
        'llm_correct': llm_eval['llm_correct'],
        'llm_explanation': llm_eval['llm_explanation'],
        'framework': 'self_reflection'
    }
    
    if to_print:
        print("\n" + "="*60)
        print(f"[FINAL] Answer: {final_answer} | GT: {gt_answer} | EM: {em_score} | LLM: {llm_eval['llm_correct']}")
        print(f"[FINAL] Corrected: {answer_was_corrected} | Verification: {verification_result['verification_status']}")
        print("="*60)
    
    # Log the result to file with framework/run folder structure
    framework_folder = os.path.join(os.path.dirname(__file__), '../../../results/hotpotqa/self_reflection')
    run_name = get_next_run_number(framework_folder)
    run_folder = os.path.join(framework_folder, run_name)
    os.makedirs(run_folder, exist_ok=True)
    log_file = os.path.join(run_folder, 'results.jsonl')
    append_to_json(info_dict, log_file)
    
    return em_score, info_dict


if __name__ == '__main__':
    # Test with a sample HotPotQA example
    print("\n[TEST] Running Self-Reflection agent test\n")
    reward, info = run_self_reflection(idx=0, to_print=True)
    print(f"\n[TEST RESULT] Reward: {reward}, Final Answer: {info['answer']}")
    print(f"[INITIAL ANSWER] {info['initial_answer']}")
    print(f"[VERIFICATION STATUS] {info['verification']['verification_status']}")
