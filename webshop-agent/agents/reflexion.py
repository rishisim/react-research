from ..llm import call_llm
from .single_trace import run_single_trace

# --- Prompt Loading ---
with open('webshop-agent/prompts/webshop_reflexion_few_shot.txt', 'r') as f:
    REFLEXION_FEW_SHOT_PROMPT = f.read()

# --- Reflexion Agent Logic ---
def generate_reflection(instruction: str, trajectory_str: str, memory: str) -> str:
    """
    Generates a reflection on a failed trajectory using an LLM.
    This is the "Self-Reflection Model" (Msr).
    """
    prompt = f"{REFLEXION_FEW_SHOT_PROMPT}\n"
    if memory:
        prompt += f"You have failed on this task before. Here are your reflections from past trials to help you succeed now:\n{memory}\n\n"
    prompt += f"{trajectory_str}\nInstruction: {instruction}\n"

    reflection = call_llm(prompt, stop=['\n'])
    return reflection

def trajectory_to_str(trajectory, instruction):
    """Converts a trajectory list into a formatted string for prompts."""
    trajectory_str = f"Instruction:\n{instruction}\n[Search]\n\n"
    for step in trajectory:
        trajectory_str += f"Action: {step['action']}\nObservation:\n{step['observation']}\n\n"
    return trajectory_str.strip()

def run_reflexion_episode(env, session_id, instruction, max_traces=3, to_print=True, max_steps=15):
    """
    Runs an episode with the Reflexion agent, which reflects on failures and retries.
    """
    long_term_memory = ""
    all_traces_info = []
    final_reward = 0.0

    for trace_num in range(1, max_traces + 1):
        if to_print:
            print(f"\n{'='*20} REFLEXION TRACE {trace_num}/{max_traces} {'='*20}")

        reward, trajectory, n_calls = run_single_trace(
            env, session_id, instruction, long_term_memory, to_print, max_steps, num_traces=1
        )

        trace_info = {
            'trace_num': trace_num,
            'n_calls': n_calls,
            'trajectory': trajectory,
            'final_reward': reward,
            'reflection': None
        }

        final_reward = reward
        if reward == 1.0:
            if to_print:
                print(f"Reflexion trace {trace_num} SUCCEEDED with reward: {reward}")
            all_traces_info.append(trace_info)
            break  # Task solved

        if to_print:
            print(f"Reflexion trace {trace_num} FAILED with reward: {reward}. Generating reflection...")

        # Generate reflection for the next attempt
        fail_trajectory_str = trajectory_to_str(trajectory, instruction)
        reflection = generate_reflection(instruction, fail_trajectory_str, long_term_memory)
        long_term_memory += f"- {reflection}\n"

        trace_info['reflection'] = reflection
        all_traces_info.append(trace_info)

        if to_print:
            print(f"\n--- Reflection for Next Trace ---")
            print(reflection)

    return final_reward, {
        'num_traces_run': len(all_traces_info),
        'individual_traces': all_traces_info,
        'final_reward': final_reward,
    }
