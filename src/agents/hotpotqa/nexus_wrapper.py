
import sys
import os

# Add local directory to path for relative imports if needed
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from hotpotqa_utils import get_hotpotqa_env, llm, step, llm_judge_answer

# Import Nexus Agent
# Assuming running from root, so src.agents... works.
from src.agents.nexus.nexus_agent import NexusAgent

class EnvWrapper:
    """Wraps the WikiEnv to use the robust step function from hotpotqa_utils."""
    def __init__(self, env):
        self.env = env
        
    def step(self, action):
        return step(self.env, action)
    
    def reset(self, idx=None):
        return self.env.reset(idx=idx)

def run_nexus(idx, prompt_template=None, to_print=True):
    """
    Run Nexus agent on a single HotPotQA question.
    """
    if to_print:
        print("="*60)
        print("[FRAMEWORK] Nexus (Scout -> Architect -> Adjudicator)")
        print("="*60)

    # Setup Environment
    env = get_hotpotqa_env()
    wrapped_env = EnvWrapper(env)
    
    # Reset env for this specific index
    question = wrapped_env.reset(idx=idx)
    
    if to_print:
        print(f"[EX] Index: {idx}")
        print(f"[Q] {question}")

    # Initialize Agent with task_type for correct prompting
    # We pass the shared 'llm' function from hotpotqa_utils
    agent = NexusAgent(llm_func=llm, env=wrapped_env, task_type="hotpotqa")
    
    # Solve
    try:
        answer, debug_info = agent.solve(question)
    except Exception as e:
        import traceback
        traceback.print_exc()
        answer = "null"
        debug_info = {"traj": f"Error: {e}", "dossier": ""}

    # Evaluate using the environment's validation via a "Finish" action
    # This ensures we get the exact same metrics as other frameworks
    first_char = answer[0].lower() if answer else ""
    # WikiEnv requires start with lowercase 'finish'
    final_action = f"finish[{answer}]"
    if to_print:
        print(f"[FINAL ACTION] {final_action}")
        
    obs, reward, done, info = wrapped_env.step(final_action)
    
    # Get GT and compute LLM Judge
    gt_answer = info.get('gt_answer', 'UNKNOWN')
    llm_eval = llm_judge_answer(question, answer, gt_answer)
    
    # Construct info_dict matched to run_single_trace output format
    info_dict = {
        'question_idx': idx,
        'question_text': question,
        'answer': answer,
        'gt_answer': gt_answer,
        'em': info.get('em', 0.0),
        'f1': info.get('f1', 0.0),
        'reward': reward,
        'n_calls': agent.n_calls,  # Use tracked LLM calls
        'n_badcalls': 0,
        'traj': debug_info.get('traj', ''),
        'framework': 'nexus',
        'llm_correct': llm_eval['llm_correct'],
        'llm_explanation': llm_eval['llm_explanation']
    }
    
    if to_print:
        print(f"[RESULT] Answer: {answer} | GT: {gt_answer} | EM: {info_dict['em']} | LLM: {info_dict['llm_correct']}")
        print("="*60)
        
    return reward, info_dict
