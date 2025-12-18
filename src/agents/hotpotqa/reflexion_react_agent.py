"""
Reflexion ReAct Agent for HotPotQA

Runs a ReAct trace, then verifies the answer using LLM. If the verification
determines the answer is incorrect, runs a second trace with verification 
feedback to guide the reasoning.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import json

from hotpotqa_utils import (
    run_single_trace,
    llm,
    llm_judge_answer,
    get_hotpotqa_env,
    WEBTHINK_PROMPT_TEMPLATE
)


def load_verification_prompt():
    """Load the reflexion verification prompt from JSON file."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    prompt_path = os.path.join(script_dir, 'prompts', 'hotpotqa_reflexion.json')
    try:
        with open(prompt_path, 'r', encoding='utf-8') as f:
            prompts = json.load(f)
        return prompts.get('reflexion_verification', '')
    except Exception as e:
        print(f"[WARNING] Could not load verification prompt: {e}")
        return ""


def verify_answer(trajectory_info, verification_prompt, to_print=False):
    """
    Verify if the answer in the trajectory is correct using LLM.
    
    Args:
        trajectory_info: Dictionary with trajectory information
        verification_prompt: The verification prompt template
        to_print: Whether to print verification output
        
    Returns:
        Dictionary with:
        - verification_status: 'CORRECT' or 'INCORRECT'
        - verification_reasoning: Explanation and guidance from the LLM
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
    
    response = llm(full_prompt, stop=[], num_traces=1)
    
    # Parse the response
    verification_status = 'CORRECT'
    if 'INCORRECT' in response.upper():
        verification_status = 'INCORRECT'
    
    if to_print:
        print(f"\n[VERIFICATION] Status: {verification_status}")
        print(f"[VERIFICATION] Response: {response[:500]}...")
    
    return {
        'verification_status': verification_status,
        'verification_reasoning': response.strip()
    }


def run_reflexion_react(idx, prompt_template=None, to_print=True):
    """
    Run Reflexion ReAct with verification-driven second trace.
    
    Process:
    1. Run initial ReAct trace
    2. Verify the answer using LLM (reflexion verification)
    3. If INCORRECT, run second trace with verification guidance
    4. Return final answer (from trace 1 if CORRECT, or trace 2 if INCORRECT)
    
    Args:
        idx: Question index from HotPotQA dataset
        prompt_template: Custom prompt template (uses default if None)
        to_print: Whether to print progress during execution
        
    Returns:
        Tuple of (reward, info_dict) where:
        - reward: EM score of the final answer
        - info_dict: Dictionary with both traces and verification results
    """
    if prompt_template is None:
        prompt_template = WEBTHINK_PROMPT_TEMPLATE
    
    if to_print:
        print("="*60)
        print("[FRAMEWORK] Reflexion ReAct (Verification-Driven)")
        print("="*60)
    
    # Load verification prompt
    verification_prompt = load_verification_prompt()
    
    # Run initial trace
    if to_print:
        print("\n--- Trace 1 (Initial) ---")
    
    trace1_info = run_single_trace(
        idx=idx,
        initial_prompt_template=prompt_template,
        to_print=to_print,
        temperature=0.0  # Deterministic first trace
    )
    
    question_text = trace1_info.get('question_text', '')
    gt_answer = trace1_info.get('gt_answer', '')
    
    if to_print:
        print(f"\n[TRACE 1] Answer: {trace1_info.get('answer')}")
    
    # Verify the first trace answer
    if to_print:
        print("\n--- Verification Phase ---")
    
    verification_result = verify_answer(trace1_info, verification_prompt, to_print=to_print)
    
    # Determine if we need a second trace
    final_answer = trace1_info.get('answer')
    trace2_info = None
    used_trace = 1
    
    if verification_result['verification_status'] == 'INCORRECT':
        if to_print:
            print("\n--- Trace 2 (With Guidance) ---")
            print(f"[GUIDANCE] Using verification feedback to improve reasoning")
        
        # Build enhanced prompt with verification guidance
        guidance = verification_result['verification_reasoning']
        enhanced_prompt = prompt_template + f"""

Previous Attempt Feedback:
The previous attempt to answer this question was incorrect. Here is guidance for this attempt:
{guidance}

Use this feedback to avoid the same mistakes. Now answer the question:
"""
        
        trace2_info = run_single_trace(
            idx=idx,
            initial_prompt_template=enhanced_prompt,
            to_print=to_print,
            temperature=0.0
        )
        
        final_answer = trace2_info.get('answer')
        used_trace = 2
        
        if to_print:
            print(f"\n[TRACE 2] Answer: {final_answer}")
    
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
    
    # Calculate total calls
    total_calls = trace1_info.get('n_calls', 0)
    total_badcalls = trace1_info.get('n_badcalls', 0)
    if trace2_info:
        total_calls += trace2_info.get('n_calls', 0)
        total_badcalls += trace2_info.get('n_badcalls', 0)
    total_calls += 1  # Verification call
    
    info_dict = {
        'question_idx': idx,
        'question_text': question_text,
        'answer': final_answer,
        'gt_answer': gt_answer,
        'em': em_score,
        'f1': f1_score,
        'reward': em_score,
        'n_calls': total_calls,
        'n_badcalls': total_badcalls,
        'used_trace': used_trace,
        'trace_1': {
            'answer': trace1_info.get('answer'),
            'em': trace1_info.get('em', 0.0),
            'traj': trace1_info.get('traj', '')
        },
        'trace_2': {
            'answer': trace2_info.get('answer') if trace2_info else None,
            'em': trace2_info.get('em', 0.0) if trace2_info else None,
            'traj': trace2_info.get('traj', '') if trace2_info else None
        } if trace2_info else None,
        'verification': verification_result,
        'llm_correct': llm_eval['llm_correct'],
        'llm_explanation': llm_eval['llm_explanation'],
        'framework': 'reflexion'
    }
    
    if to_print:
        print("\n" + "="*60)
        print(f"[FINAL] Answer: {final_answer} | GT: {gt_answer} | EM: {em_score} | LLM: {llm_eval['llm_correct']}")
        print(f"[FINAL] Used Trace: {used_trace} | Verification: {verification_result['verification_status']}")
        print("="*60)
    
    return em_score, info_dict


if __name__ == '__main__':
    # Test with a sample HotPotQA example
    print("\n[TEST] Running Reflexion ReAct agent test\n")
    reward, info = run_reflexion_react(idx=0, to_print=True)
    print(f"\n[TEST RESULT] Reward: {reward}, Answer: {info['answer']}")
    print(f"[VERIFICATION STATUS] {info['verification']['verification_status']}")
