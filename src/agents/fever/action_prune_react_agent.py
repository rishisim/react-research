"""
Action Prune ReAct Agent for FEVER

ReAct agent with added rules to prevent common failure modes:
- No repeated searches or lookups
- Maximum 2 consecutive searches before lookup required
- Specific entity searches (not generic words)
- Evidence-first finishing (only finish after seeing supporting facts)
- Multi-hop reasoning support
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fever_utils import (
    run_single_trace,
    get_fever_env,
    step as env_step
)

# Load base prompt
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    prompt_path = os.path.join(script_dir, 'prompts', 'fever.json')
    import json
    with open(prompt_path, 'r') as f:
        prompt_dict = json.load(f)
    webthink_examples = prompt_dict.get('webthink_short', prompt_dict.get('webthink_simple3', ""))
except:
    webthink_examples = ""

# Define simplified instruction (rules enforced by controller)
instruction = """Solve a question answering task with interleaving Thought, Action, Observation steps. Thought can reason about the current situation, and Action can be three types: (1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search. (2) Lookup[keyword], which returns the next sentence containing keyword in the current passage. (3) Finish[answer], which returns the answer and finishes the task.

### BEGIN EPISODE ###
"""

# Construct the full prompt template
ACTION_PRUNE_PROMPT_TEMPLATE = instruction + webthink_examples


def run_action_prune_react(idx, prompt_template=None, to_print=True):
    """
    Run Action Prune ReAct agent with a single trace and enhanced rules.
    
    Args:
        idx: Question index from FEVER dataset
        prompt_template: Custom prompt template (uses action prune template if None)
        to_print: Whether to print progress during execution
        
    Returns:
        Tuple of (reward, info_dict) where:
        - reward: EM score (0.0 or 1.0)
        - info_dict: Dictionary with keys:
            - question_idx: Index of the question
            - question_text: The claim text
            - answer: Agent's answer (SUPPORTS/REFUTES/NOT ENOUGH INFO)
            - gt_answer: Ground truth answer
            - em: Exact match score
            - f1: F1 score (same as EM for FEVER)
            - reward: Same as EM
            - n_calls: Number of LLM calls
            - n_badcalls: Number of failed LLM calls
            - traj: Full trajectory string
    """
    if prompt_template is None:
        prompt_template = ACTION_PRUNE_PROMPT_TEMPLATE
    
    if to_print:
        print("="*60)
        print("[FRAMEWORK] Action Prune ReAct (Single Trace with Rules)")
        print("="*60)
    
    # Run single trace with temperature=0.0 (deterministic)
    trace_info = run_single_trace(
        idx=idx,
        initial_prompt_template=prompt_template,
        to_print=to_print,
        temperature=0.0
    )
    
    # Calculate reward
    reward = trace_info.get('em', 0.0)
    
    # Prepare info dict with standardized format
    info_dict = {
        'question_idx': trace_info.get('question_idx'),
        'question_text': trace_info.get('question_text'),
        'answer': trace_info.get('answer'),
        'gt_answer': trace_info.get('gt_answer'),
        'em': trace_info.get('em', 0.0),
        'f1': trace_info.get('em', 0.0),  # For FEVER, F1 = EM
        'reward': reward,
        'n_calls': trace_info.get('n_calls', 0),
        'n_badcalls': trace_info.get('n_badcalls', 0),
        'input_tokens': trace_info.get('input_tokens', 0),
        'output_tokens': trace_info.get('output_tokens', 0),
        'total_tokens': trace_info.get('total_tokens', 0),
        'traj': trace_info.get('traj', ''),
        'framework': 'action_prune_react'
    }
    
    if to_print:
        print("="*60)
        print(f"[FINAL] Answer: {info_dict['answer']} | GT: {info_dict['gt_answer']} | EM: {info_dict['em']}")
        print("="*60)
    
    return reward, info_dict


if __name__ == '__main__':
    # Test with a sample FEVER example
    print("\n[TEST] Running Action Prune ReAct agent test\n")
    reward, info = run_action_prune_react(idx=3687, to_print=True)
    print(f"\n[TEST RESULT] Reward: {reward}, Answer: {info['answer']}")
