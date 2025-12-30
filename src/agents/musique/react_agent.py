"""
ReAct Agent for Musique
"""

import sys
import os
import json

# Add parent path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from musique_utils import (
    run_single_trace
)

# Load default prompt
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    prompt_path = os.path.join(script_dir, 'prompts', 'musique.json')
    with open(prompt_path, 'r') as f:
        prompts = json.load(f)
    WEBTHINK_PROMPT_TEMPLATE = prompts['webthink_musique_iterative']
except Exception as e:
    print(f"[ERROR] Loading prompts: {e}")
    WEBTHINK_PROMPT_TEMPLATE = "Question: {question}\n"


def run_react(idx, prompt_template=None, to_print=True):
    """
    Run standard ReAct agent on Musique.
    """
    if prompt_template is None:
        prompt_template = WEBTHINK_PROMPT_TEMPLATE
        
    if to_print:
        print("="*60)
        print("[FRAMEWORK] Musique ReAct (Iterative Search)")
        print("="*60)
        
    trace_info = run_single_trace(
        idx=idx,
        initial_prompt_template=prompt_template,
        to_print=to_print,
        temperature=0.0
    )
    
    return trace_info['em'], trace_info

if __name__ == '__main__':
    # Test on index 0
    run_react(0)
