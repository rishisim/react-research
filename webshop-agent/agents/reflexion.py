from llm import call_llm
from agents.single_trace import run_single_trace

# --- Prompt Loading ---
with open('prompts/webshop_reflexion_few_shot.txt', 'r') as f:
    REFLEXION_FEW_SHOT_PROMPT = f.read()

# --- Reflexion Agent Logic ---
def generate_reflection(instruction: str, failed_trajectory: str, task_reflections: str) -> str:
    """
    Generates a reflection on a failed trajectory to improve future attempts.
    Uses accumulated reflections from previous attempts within this task.
    """

    reflection_prompt = "You will be given the history of a past experience in which you were placed in an environment and given a task to complete. You were unsuccessful in completing the task. Do not summarize your environment, but rather think about the strategy and path you took to attempt to complete the task. Devise a concise, new plan of action that accounts for your mistake with reference to specific actions that you should have taken. There are two examples below." + REFLEXION_FEW_SHOT_PROMPT + "\n"
    
    # Plans from past attempts
    if task_reflections:
        reflection_prompt += f"You have failed on this task before. Your previous reflections didn't work, so analyze what went wrong in these attempts and devise a different strategy to complete the task:\n{task_reflections}\n\n"


    
    reflection_prompt += f"{failed_trajectory}\n\nInstruction: {instruction}\n\nReflection: "
    
    # Use appropriate stop tokens to end reflection without cutting off mid-paragraph
    # Use num_traces=2 to get higher temperature (0.7) for more varied reflections
    return call_llm(reflection_prompt, stop=["Action:"], num_traces=2)

def format_trajectory_for_reflection(trajectory, instruction):
    """Formats a failed trajectory for reflection generation."""
    formatted = f"Instruction:\n{instruction}\n[Search]\n\n"
    for step in trajectory:
        formatted += f"Action: {step['action']}\nObservation:\n{step['observation']}\n\n"
    return formatted.strip()

def run_reflexion_episode(env, session_id, instruction, max_traces=3, to_print=True, max_steps=15):
    """
    Runs multiple ReAct attempts with reflection after failures.
    Accumulates task-scoped reflections to improve subsequent attempts.
    """
    task_reflections = ""
    trace_results = []
    final_reward = 0.0

    for attempt_num in range(1, max_traces + 1):
        if to_print:
            print(f"\n{'='*20} REFLEXION ATTEMPT {attempt_num}/{max_traces} {'='*20}")

        # Run single ReAct trace with accumulated reflections
        reward, trajectory, llm_calls = run_single_trace(
            env, session_id, instruction, task_reflections, to_print, max_steps, num_traces=1
        )

        attempt_result = {
            'attempt_num': attempt_num,
            'llm_calls': llm_calls,
            'trajectory': trajectory,
            'reward': reward,
            'reflection': None
        }

        final_reward = reward
        
        if reward == 1.0:
            if to_print:
                print(f"Reflexion attempt {attempt_num} SUCCEEDED with reward: {reward}")
            trace_results.append(attempt_result)
            break  # Task completed successfully

        if to_print:
            print(f"Reflexion attempt {attempt_num} FAILED with reward: {reward}. Generating reflection...")

        # Generate reflection for next attempt
        failed_trajectory = format_trajectory_for_reflection(trajectory, instruction)
        reflection = generate_reflection(instruction, failed_trajectory, task_reflections)
        task_reflections += f"- {reflection}\n"

        attempt_result['reflection'] = reflection
        trace_results.append(attempt_result)

        if to_print:
            print(f"\n--- Reflection for Next Attempt ---")
            print(reflection)

    return final_reward, {
        'total_attempts': len(trace_results),
        'attempt_details': trace_results,
        'final_reward': final_reward,
    }
