"""
Reflexion ReAct Agent for FEVER

Runs a ReAct trace, then verifies the answer using LLM. If the verification
determines the answer is incorrect, runs a second trace with verification 
feedback to guide the reasoning.
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


def load_verification_prompt():
    """Load the reflexion verification prompt from JSON file."""
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        prompt_path = os.path.join(script_dir, 'prompts', 'fever_reflexion.json')
        with open(prompt_path, 'r') as f:
            prompts = json.load(f)
        return prompts['reflexion_verification']
    except (FileNotFoundError, KeyError) as e:
        print(f"[WARNING] Could not load verification prompt: {e}")
        return """You will be given a reasoning trace and its final answer. Analyze whether the answer is correct based on the evidence.
Respond with:
Verification: [CORRECT or INCORRECT]
Reasoning: [Brief explanation]"""


def verify_answer(trajectory_info, verification_prompt, to_print=False):
    """
    Generate a reflection and plan based on past trajectory using LLM.
    
    Args:
        trajectory_info: Dictionary with trajectory information
        verification_prompt: The verification/reflection prompt template
        to_print: Whether to print the reflection
        
    Returns:
        Dictionary with reflection results:
        - plan: The LLM-generated plan for the next attempt
        - full_response: Full response from LLM
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
    
    # Create full prompt for reflection
    full_prompt = f"{verification_prompt}\n\nYour past attempt:\n{trajectory_str}"
    
    # Generate reflection using LLM
    try:
        reflection_response = llm(full_prompt, stop=[], num_traces=1)
        
        # Extract the plan from the response
        plan = reflection_response.strip()
        
        # Try to extract everything after "Plan:" for clarity
        if "Plan:" in reflection_response:
            plan = reflection_response.split("Plan:")[-1].strip()
        
        if to_print:
            print(f"[REFLECTION] Plan generated:")
            print(f"{plan}")
        
        return {
            'plan': plan,
            'full_response': reflection_response
        }
    except Exception as e:
        if to_print:
            print(f"[ERROR] Error during reflection: {e}")
        return {
            'plan': f'Error during reflection: {str(e)}',
            'full_response': ''
        }


def run_reflexion_react(idx, prompt_template=None, to_print=True, max_trials=7):
    """
    Run Reflexion ReAct with multi-trial verification-driven approach.
    
    Process:
    1. Run initial ReAct trace
    2. Verify the answer using LLM (reflexion verification)
    3. If INCORRECT, run another ReAct trace with verification feedback (repeat up to max_trials)
    4. Return final answer from the last trace
    
    Args:
        idx: Question index from FEVER dataset
        prompt_template: Custom prompt template (uses default if None)
        to_print: Whether to print progress during execution
        max_trials: Maximum number of trials to attempt (default: 7)
        
    Returns:
        Tuple of (reward, info_dict) where:
        - reward: EM score of the final answer
        - info_dict: Dictionary with all traces and verification results
    """
    if prompt_template is None:
        prompt_template = WEBTHINK_PROMPT_TEMPLATE
    
    # Load verification prompt
    verification_prompt = load_verification_prompt()
    
    if to_print:
        print("="*60)
        print(f"[FRAMEWORK] Reflexion ReAct (Up to {max_trials} Trials)")
        print("="*60)
    
    # Store all traces
    traces = {}
    reflections = {}
    num_traces_run = 0
    final_answer = None
    question_text = None
    gt_answer = None
    
    # Iterative trial loop
    for trial_num in range(1, max_trials + 1):
        if to_print:
            print(f"\n--- Trial {trial_num} ---")
        
        # Determine prompt for this trial
        if trial_num == 1:
            # First trial uses base prompt
            current_prompt = prompt_template
        else:
            # Build memory context with ONLY LAST 3 reflexions
            memory_context = ""
            start_trial = max(1, trial_num - 3)  # Only last 3 reflexions
            for t in range(start_trial, trial_num):
                plan_t = reflections.get(t, {}).get('plan', f'[Trial {t} plan not available]')
                memory_context += f"Trial #{t}: {plan_t}\n"
            
            # Generate new plan using reflection on the most recent failed attempt
            plan_prompt = f"""{verification_prompt}

Previous reflexions:
{memory_context}
"""
            reflection_result = verify_answer(traces[trial_num - 1], plan_prompt, to_print=to_print)
            new_plan = reflection_result.get('plan', '')
            
            # Store the reflection that generated this trial's plan
            reflections[trial_num - 1] = reflection_result
            
            # Incorporate plan into prompt as guidance
            current_prompt = f"""Based on this plan: {new_plan}

{prompt_template}"""
        
        # Run the trace
        trace_result = run_single_trace(
            idx=idx,
            initial_prompt_template=current_prompt,
            to_print=to_print,
            temperature=0.0
        )
        
        traces[trial_num] = trace_result
        num_traces_run = trial_num
        question_text = trace_result.get('question_text')
        gt_answer = trace_result.get('gt_answer')
        final_answer = trace_result.get('answer')
        
        if to_print:
            print(f"[TRIAL {trial_num}] Answer: {final_answer}")
        
        # Check if the answer is correct (exact match for FEVER: SUPPORTS/REFUTES/NOT ENOUGH INFO)
        is_correct = (final_answer == gt_answer)
        
        if to_print:
            print(f"[EVALUATION] Answer {'CORRECT' if is_correct else 'INCORRECT'} (GT: {gt_answer})")
        
        # If correct, stop - no need for more trials
        if is_correct:
            if to_print:
                print(f"[SUCCESS] Trial {trial_num} - Answer is correct! Stopping.")
            # Store a "success" reflection
            reflections[trial_num] = {
                'plan': 'No plan needed - answer was correct.',
                'full_response': 'Task completed successfully.',
                'success': True
            }
            break
        
        # If incorrect and not last trial, we'll reflect at the start of next iteration
        if trial_num < max_trials:
            if to_print:
                print(f"[INCORRECT] Trial {trial_num} - Will reflect and try again...")
        else:
            # Last trial - generate final reflection for records
            if to_print:
                print(f"[MAX TRIALS] Reached maximum of {max_trials} trials")
            final_reflection = verify_answer(trace_result, verification_prompt, to_print=to_print)
            reflections[trial_num] = final_reflection
    
    # Calculate metrics based on final answer
    em_score = 1.0 if final_answer == gt_answer else 0.0
    
    # Aggregate call counts across all traces
    total_calls = 0
    total_badcalls = 0
    for trial_num in range(1, num_traces_run + 1):
        total_calls += traces[trial_num].get('n_calls', 0)
        total_badcalls += traces[trial_num].get('n_badcalls', 0)
    
    # Add verification calls (one per trial)
    total_calls += num_traces_run
    
    # Build traces dict for info
    traces_info = {}
    for trial_num in range(1, num_traces_run + 1):
        traces_info[f'trace_{trial_num}'] = {
            'answer': traces[trial_num].get('answer'),
            'em': traces[trial_num].get('em', 0.0),
            'n_calls': traces[trial_num].get('n_calls', 0),
            'traj': traces[trial_num].get('traj', '')
        }
    
    # Build reflections/plans dict for info
    reflections_info = {}
    for trial_num in range(1, num_traces_run + 1):
        reflections_info[f'trial_{trial_num}'] = {
            'plan': reflections.get(trial_num, {}).get('plan', ''),
            'full_response': reflections.get(trial_num, {}).get('full_response', '')
        }
    
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
        'num_trials': num_traces_run,
        'max_trials': max_trials,
        'all_traces': traces_info,
        'all_reflections': reflections_info,
        'final_plan': reflections.get(num_traces_run, {}).get('plan', ''),
        'framework': 'reflexion_react'
    }
    
    if to_print:
        print("="*60)
        print(f"[FINAL] Answer: {final_answer} | GT: {gt_answer} | EM: {em_score}")
        print(f"[TRIALS] Ran {num_traces_run}/{max_trials} trials")
        print(f"[FINAL PLAN] {reflections.get(num_traces_run, {}).get('plan', 'N/A')[:100]}...")
        print("="*60)
    
    # Log the result to file with framework/run folder structure
    framework_folder = os.path.join(os.path.dirname(__file__), '../../../results/fever/reflexion')
    run_name = get_next_run_number(framework_folder)
    run_folder = os.path.join(framework_folder, run_name)
    os.makedirs(run_folder, exist_ok=True)
    log_file = os.path.join(run_folder, 'results.jsonl')
    append_to_json(info_dict, log_file)
    
    return em_score, info_dict


if __name__ == '__main__':
    # Test with a sample FEVER example
    print("\n[TEST] Running Reflexion ReAct agent test (max 7 trials)\n")
    reward, info = run_reflexion_react(idx=3687, to_print=True, max_trials=7)
    print(f"\n[TEST RESULT] Reward: {reward}, Final Answer: {info['answer']}")
    print(f"[TRACES RUN] {info['num_traces_run']}")
    print(f"[TRACE 1] Answer: {info['trace_1']['answer']}")
    if info['trace_2']:
        print(f"[TRACE 2] Answer: {info['trace_2']['answer']}")
    print(f"[VERIFICATION STATUS] {info['verification']['verification_status']}")
    print(f"[VERIFICATION REASONING] {info['verification']['verification_reasoning']}")
