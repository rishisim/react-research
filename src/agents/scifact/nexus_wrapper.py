"""
Nexus Agent Wrapper for SciFACT

Wraps the shared NexusAgent to work with SciFACT corpus-based search.
"""

import sys
import os

# Add local directory to path for relative imports if needed
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Add project root to path for nexus import
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)

from scifact_utils import get_scifact_env, llm, step

# Import Nexus Agent
from src.agents.nexus.nexus_agent import NexusAgent


class EnvWrapper:
    """Wraps the SciFACT env to use the robust step function from scifact_utils."""
    def __init__(self, env):
        self.env = env
        
    def step(self, action):
        return step(self.env, action)
    
    def reset(self, idx=None):
        return self.env.reset(idx=idx)


def run_nexus(idx, prompt_template=None, to_print=True):
    """
    Run Nexus agent on a single SciFACT claim.
    
    Args:
        idx: Claim index from SciFACT dataset
        prompt_template: Not used (Nexus has its own prompts), kept for API compatibility
        to_print: Whether to print progress during execution
        
    Returns:
        Tuple of (reward, info_dict)
    """
    if to_print:
        print("="*60)
        print("[FRAMEWORK] Nexus (Scout -> Architect -> Adjudicator)")
        print("="*60)

    # Setup Environment
    env = get_scifact_env()
    wrapped_env = EnvWrapper(env)
    
    # Reset env for this specific index
    claim = wrapped_env.reset(idx=idx)
    
    if to_print:
        print(f"[EX] Index: {idx}")
        print(f"[CLAIM] {claim}")

    # Initialize Agent with task_type for SciFACT (similar to FEVER)
    # We pass the shared 'llm' function from scifact_utils
    agent = NexusAgent(llm_func=llm, env=wrapped_env, task_type="fever")  # Use fever task_type for claim verification
    
    # Solve
    try:
        answer, debug_info = agent.solve(claim)
    except Exception as e:
        import traceback
        traceback.print_exc()
        answer = "NOT ENOUGH INFO"
        debug_info = {"traj": f"Error: {e}", "dossier": ""}

    # Normalize answer
    answer = _normalize_answer(answer)
    
    # Force lower-case finish for SciFACT env compatibility
    final_action = f"finish[{answer}]"
    if to_print:
        print(f"[FINAL ACTION] {final_action}")
        
    obs, reward, done, info = wrapped_env.step(final_action)
    
    # Construct info_dict
    info_dict = {
        'question_idx': idx,
        'question_text': claim,
        'answer': answer,
        'gt_answer': info.get('gt_answer', 'UNKNOWN'),
        'em': info.get('em', 0.0),
        'f1': info.get('f1', 0.0),
        'reward': reward,
        'n_calls': agent.n_calls,  # Use tracked LLM calls
        'n_badcalls': 0,
        'traj': debug_info.get('traj', ''),
        'framework': 'nexus'
    }
    
    if to_print:
        print(f"[RESULT] Answer: {answer} | GT: {info_dict['gt_answer']} | EM: {info_dict['em']}")
        print("="*60)
        
    return reward, info_dict


def _normalize_answer(answer: str) -> str:
    """
    Normalize answer to standard SciFACT format.
    Uses regex extraction to find labels in full reasoning text.
    """
    import re
    
    answer = answer.strip()
    
    # First, try to find Answer: pattern in the text
    answer_patterns = [
        r'Answer:\s*(SUPPORT(?:S)?|REFUTE(?:S)?|NOT ENOUGH INFO)',
        r'Final Answer:\s*(SUPPORT(?:S)?|REFUTE(?:S)?|NOT ENOUGH INFO)',
        r'\*\*(SUPPORT(?:S)?|REFUTE(?:S)?|NOT ENOUGH INFO)\*\*',
        r'\b(SUPPORTS|SUPPORT|REFUTES|REFUTE|NOT ENOUGH INFO)\b',
    ]
    
    for pattern in answer_patterns:
        match = re.search(pattern, answer, re.IGNORECASE)
        if match:
            label = match.group(1).upper()
            # Normalize
            if label in ["SUPPORT", "SUPPORTS"]:
                return "SUPPORT"
            elif label in ["REFUTE", "REFUTES"]:
                return "REFUTE"
            elif label in ["NOT ENOUGH INFO", "NEI", "NOINFO", "NOT_ENOUGH_INFO"]:
                return "NOT ENOUGH INFO"
    
    # Check for exact match (fallback)
    answer_upper = answer.upper()
    if answer_upper in ["SUPPORT", "SUPPORTS"]:
        return "SUPPORT"
    elif answer_upper in ["REFUTE", "REFUTES"]:
        return "REFUTE"
    elif answer_upper in ["NOT ENOUGH INFO", "NEI", "NOINFO", "NOT_ENOUGH_INFO"]:
        return "NOT ENOUGH INFO"
    
    # Default - couldn't extract a valid label
    print(f"[WARNING] Could not extract answer from: '{answer[:100]}...'")
    return "NOT ENOUGH INFO"


if __name__ == '__main__':
    # Test with a sample SciFACT claim
    print("\n[TEST] Running Nexus agent test\n")
    reward, info = run_nexus(idx=0, to_print=True)
    print(f"\n[TEST RESULT] Reward: {reward}, Answer: {info['answer']}")
