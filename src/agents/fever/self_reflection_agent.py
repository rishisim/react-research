"""
Self-Reflection Agent for FEVER

Runs a single ReAct trace, then uses LLM to verify the answer based on the trace.
If the verification determines the answer is incorrect, outputs the corrected answer
based solely on the evidence in the trace.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import json
from datetime import datetime

from fever_utils import (
    run_single_trace,
    llm,
    append_to_json,
    get_next_run_number,
    WEBTHINK_PROMPT_TEMPLATE
)


def load_self_reflection_prompt():
    """Load the self-reflection verification prompt from JSON file."""
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        prompt_path = os.path.join(script_dir, 'prompts', 'fever_self_reflection.json')
        with open(prompt_path, 'r') as f:
            prompts = json.load(f)
        return prompts['self_reflection_verification']
    except (FileNotFoundError, KeyError) as e:
        print(f"[WARNING] Could not load self-reflection prompt: {e}")
        return """You will be given a reasoning trace and its final answer. Analyze whether the answer is correct based on the evidence.
Respond with:
Verification: [CORRECT or INCORRECT]
Reasoning: [Brief explanation]
Final Answer: [The correct answer based on the trace]"""


def verify_and_correct_answer(trajectory_info, verification_prompt, to_print=False):
    """
    Verify if the answer in the trajectory is correct using LLM, and provide correction if needed.
    
    Args:
        trajectory_info: Dictionary with trajectory information
        verification_prompt: The verification prompt template
        to_print: Whether to print the verification
        
    Returns:
        Dictionary with verification results:
        - verification_status: 'CORRECT' or 'INCORRECT'
        - verification_reasoning: Explanation from the LLM
        - corrected_answer: The final answer (original if correct, corrected if incorrect)
    """
    # Extract trajectory string
    if isinstance(trajectory_info, dict):
        trajectory_str = trajectory_info.get('traj', '')
        if not trajectory_str:
            # Fallback: reconstruct from available info
            question = trajectory_info.get('question_text', '')
            answer = trajectory_info.get('answer', '')
            trajectory_str = f"Claim: {question}\n[Trajectory details not available]\nFinal Answer: {answer}"
    else:
        trajectory_str = str(trajectory_info)
    
    # Create full prompt for verification
    full_prompt = f"{verification_prompt}\n\nNow verify this reasoning trace:\n\n{trajectory_str}\n\nVerification:"
    
    # Generate verification using LLM
    try:
        verification_response, token_usage = llm(full_prompt, stop=[], num_traces=1)
        
        # Parse the verification response
        verification_status = "UNKNOWN"
        verification_reasoning = ""
        corrected_answer = trajectory_info.get('answer', 'NOT ENOUGH INFO')
        
        # Try to extract CORRECT or INCORRECT from the response
        if "CORRECT" in verification_response.upper():
            if "INCORRECT" in verification_response.upper():
                # Both appear, check which comes first after "Verification:"
                if verification_response.upper().find("VERIFICATION:") != -1:
                    after_label = verification_response[verification_response.upper().find("VERIFICATION:") + len("VERIFICATION:"):]
                    if after_label.strip().upper().startswith("CORRECT"):
                        verification_status = "CORRECT"
                    elif after_label.strip().upper().startswith("INCORRECT"):
                        verification_status = "INCORRECT"
            else:
                verification_status = "CORRECT"
        elif "INCORRECT" in verification_response.upper():
            verification_status = "INCORRECT"
        
        # Try to extract reasoning portion
        if "Reasoning:" in verification_response:
            reasoning_part = verification_response.split("Reasoning:")[-1]
            if "Final Answer:" in reasoning_part:
                verification_reasoning = reasoning_part.split("Final Answer:")[0].strip()
            else:
                verification_reasoning = reasoning_part.strip()
        
        # Try to extract corrected answer if verification says INCORRECT
        if "Final Answer:" in verification_response:
            final_answer_part = verification_response.split("Final Answer:")[-1].strip()
            # Extract SUPPORTS, REFUTES, or NOT ENOUGH INFO
            for valid_answer in ['SUPPORTS', 'REFUTES', 'NOT ENOUGH INFO']:
                if valid_answer in final_answer_part.upper():
                    corrected_answer = valid_answer
                    break
        
        if to_print:
            print(f"[VERIFICATION] Status: {verification_status}")
            print(f"[VERIFICATION] Reasoning: {verification_reasoning}")
            if verification_status == "INCORRECT":
                print(f"[VERIFICATION] Corrected Answer: {corrected_answer}")
        
        return {
            'verification_status': verification_status,
            'verification_reasoning': verification_reasoning,
            'corrected_answer': corrected_answer,
            'full_verification_response': verification_response,
            'input_tokens': token_usage['input_tokens'],
            'output_tokens': token_usage['output_tokens'],
            'total_tokens': token_usage['total_tokens']
        }
    except Exception as e:
        if to_print:
            print(f"[ERROR] Error during verification: {e}")
        return {
            'verification_status': 'ERROR',
            'verification_reasoning': f'Error during verification: {str(e)}',
            'corrected_answer': trajectory_info.get('answer', 'NOT ENOUGH INFO'),
            'full_verification_response': '',
            'input_tokens': 0,
            'output_tokens': 0,
            'total_tokens': 0
        }


def run_self_reflection(idx, prompt_template=None, to_print=True):
    """
    Run Self-Reflection agent with single trace and verification.
    
    Process:
    1. Run a single ReAct trace
    2. Verify the answer using LLM based on the trace
    3. If INCORRECT, use the corrected answer from verification
    4. If CORRECT, use the original answer
    
    Args:
        idx: Question index from FEVER dataset
        prompt_template: Custom prompt template (uses default if None)
        to_print: Whether to print progress during execution
        
    Returns:
        Tuple of (reward, info_dict) where:
        - reward: EM score of the final answer
        - info_dict: Dictionary with trace and verification results
    """
    if prompt_template is None:
        prompt_template = WEBTHINK_PROMPT_TEMPLATE
    
    # Load verification prompt
    verification_prompt = load_self_reflection_prompt()
    
    if to_print:
        print("="*60)
        print("[FRAMEWORK] Self-Reflection (Single Trace + Verification)")
        print("="*60)
    
    # --- Step 1: Run single ReAct trace ---
    if to_print:
        print("\n--- Step 1: Running ReAct Trace ---")
    
    trace_info = run_single_trace(
        idx=idx,
        initial_prompt_template=prompt_template,
        to_print=to_print,
        temperature=0.0
    )
    
    question_text = trace_info.get('question_text')
    gt_answer = trace_info.get('gt_answer')
    initial_answer = trace_info.get('answer')
    
    if to_print:
        print(f"[TRACE] Initial Answer: {initial_answer}")
    
    # --- Step 2: Verify and potentially correct the answer ---
    if to_print:
        print(f"\n{'='*60}")
        print("[VERIFICATION] Verifying answer based on trace...")
    
    verification_result = verify_and_correct_answer(trace_info, verification_prompt, to_print=to_print)
    
    # --- Step 3: Determine final answer ---
    if verification_result['verification_status'] == 'CORRECT':
        final_answer = initial_answer
        if to_print:
            print(f"[VERIFICATION] Answer verified as CORRECT")
    elif verification_result['verification_status'] == 'INCORRECT':
        final_answer = verification_result['corrected_answer']
        if to_print:
            print(f"[VERIFICATION] Answer was INCORRECT")
            print(f"[CORRECTION] Corrected Answer: {final_answer}")
    else:
        # If verification status is UNKNOWN or ERROR, use original answer
        final_answer = initial_answer
        if to_print:
            print(f"[WARNING] Verification status {verification_result['verification_status']}, using original answer")
    
    # Calculate metrics based on final answer
    em_score = 1.0 if final_answer == gt_answer else 0.0
    
    # Aggregate call counts and tokens (1 trace + 1 verification call)
    total_calls = trace_info.get('n_calls', 0) + 1  # +1 for verification
    total_badcalls = trace_info.get('n_badcalls', 0)
    total_input_tokens = trace_info.get('input_tokens', 0) + verification_result.get('input_tokens', 0)
    total_output_tokens = trace_info.get('output_tokens', 0) + verification_result.get('output_tokens', 0)
    
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
        'input_tokens': total_input_tokens,
        'output_tokens': total_output_tokens,
        'total_tokens': total_input_tokens + total_output_tokens,
        'initial_answer': initial_answer,
        'verification': verification_result,
        'trace': {
            'answer': initial_answer,
            'em': trace_info.get('em', 0.0),
            'n_calls': trace_info.get('n_calls', 0),
            'input_tokens': trace_info.get('input_tokens', 0),
            'output_tokens': trace_info.get('output_tokens', 0),
            'traj': trace_info.get('traj', '')
        },
        'framework': 'self_reflection'
    }
    
    if to_print:
        print("="*60)
        print(f"[FINAL] Answer: {final_answer} | GT: {gt_answer} | EM: {em_score}")
        print(f"[INITIAL] {initial_answer} -> [FINAL] {final_answer}")
        print(f"[VERIFICATION] {verification_result['verification_status']}")
        print("="*60)
    
    # Logging is handled by the experiment runner (run_fever_experiments.py)
    # framework_folder = os.path.join(os.path.dirname(__file__), '../../../results/fever/self_reflection')
    # run_name = get_next_run_number(framework_folder)
    # run_folder = os.path.join(framework_folder, run_name)
    # os.makedirs(run_folder, exist_ok=True)
    # log_file = os.path.join(run_folder, 'results.jsonl')
    # append_to_json(info_dict, log_file)
    
    return em_score, info_dict


if __name__ == '__main__':
    # Test with a sample FEVER example
    print("\n[TEST] Running Self-Reflection agent test\n")
    reward, info = run_self_reflection(idx=3687, to_print=True)
    print(f"\n[TEST RESULT] Reward: {reward}, Final Answer: {info['answer']}")
    print(f"[INITIAL ANSWER] {info['initial_answer']}")
    print(f"[VERIFICATION STATUS] {info['verification']['verification_status']}")
    print(f"[VERIFICATION REASONING] {info['verification']['verification_reasoning']}")
