"""
ReWOO Agent Wrapper for FEVEROUS

Wraps the FEVEROUS-specific ReWOO agent (Planner -> Worker -> Solver)
with TableLookup support for structured table data.
"""

import sys
import os

# Add local directory to path for relative imports if needed
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from feverous_utils import get_feverous_env, llm, step

# Import FEVEROUS-specific ReWOO Agent
from feverous_rewoo_agent import FeverousReWOOAgent


class EnvWrapper:
    """Wraps the FeverousEnv to use the robust step function from feverous_utils."""
    def __init__(self, env):
        self.env = env
        
    def step(self, action):
        return step(self.env, action)
    
    def reset(self, idx=None):
        return self.env.reset(idx=idx)


def run_rewoo(idx, prompt_template=None, to_print=True):
    """
    Run ReWOO agent on a single FEVEROUS question.
    
    Args:
        idx: Question index from FEVEROUS dataset
        prompt_template: Not used (ReWOO has its own prompts)
        to_print: Whether to print progress
        
    Returns:
        Tuple of (reward, info_dict)
    """
    if to_print:
        print("="*60)
        print("[FRAMEWORK] FEVEROUS ReWOO (Planner -> Worker -> Solver)")
        print("="*60)

    # Setup Environment
    env = get_feverous_env()
    wrapped_env = EnvWrapper(env)
    
    # Reset env for this specific index
    question = wrapped_env.reset(idx=idx)
    
    if to_print:
        print(f"[EX] Index: {idx}")
        print(f"[Q] {question}")

    # Initialize FEVEROUS-specific ReWOO Agent
    agent = FeverousReWOOAgent(llm_func=llm, env=wrapped_env)
    
    # Solve
    try:
        answer, debug_info = agent.solve(question)
    except Exception as e:
        import traceback
        traceback.print_exc()
        answer = "NOT ENOUGH INFO"
        debug_info = {"traj": f"Error: {e}", "plan": "", "evidence": {}}

    # Force lower-case finish for env compatibility
    final_action = f"finish[{answer}]"
    if to_print:
        print(f"[FINAL ACTION] {final_action}")
        
    obs, reward, done, info = wrapped_env.step(final_action)
    
    # Construct info_dict
    info_dict = {
        'question_idx': idx,
        'question_text': question,
        'answer': answer,
        'gt_answer': info.get('gt_answer', 'UNKNOWN'),
        'em': info.get('em', 0.0),
        'f1': info.get('f1', 0.0),
        'reward': reward,
        'n_calls': agent.n_calls,  # LLM calls (should be 2: Planner + Solver)
        'n_badcalls': 0,
        'traj': debug_info.get('traj', ''),
        'framework': 'rewoo'
    }
    
    if to_print:
        print(f"[RESULT] Answer: {answer} | GT: {info_dict['gt_answer']} | EM: {info_dict['em']}")
        print("="*60)
        
    return reward, info_dict


if __name__ == '__main__':
    # Test with a sample FEVEROUS example
    print("\n[TEST] Running FEVEROUS ReWOO agent test\n")
    reward, info = run_rewoo(idx=0, to_print=True)
    print(f"\n[TEST RESULT] Reward: {reward}, Answer: {info['answer']}")
    print(f"[LLM CALLS] {info['n_calls']}")
    print("\n[TRACE]\n" + info['traj'])
