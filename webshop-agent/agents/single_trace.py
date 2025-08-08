from llm import call_llm

# --- Prompt Loading ---
with open('prompts/react_few_shot.txt', 'r') as f:
    FEW_SHOT_PROMPT = f.read()

# --- ReAct Agent Logic ---
def run_single_trace(env, session_id, instruction, task_reflections: str = "", to_print=True, max_steps=15, num_traces=1):
    """
    Run a single ReAct reasoning trace, optionally guided by previous reflections from this task.
    """
    action = 'reset'

    # Build initial prompt with task reflections if available
    initial_prompt = FEW_SHOT_PROMPT + "\n"
    if task_reflections:
        initial_prompt += f"You have failed on this task before. Here are your reflections from past trials to help you succeed now:\n---BEGIN REFLECTIONS---\n{task_reflections}\n---END REFLECTIONS---\n\n"
    initial_prompt += f"Instruction: {instruction}\n[Search]\n"

    prompt_history = ''
    trajectory = []
    llm_calls = 0

    for i in range(max_steps):
        try:
            observation, reward, done = env.step(session_id, action)
        except AssertionError:
            observation, reward, done = 'Invalid action!', 0.0, False

        if action.startswith('think['): observation = 'OK.'

        trajectory.append({'step': i, 'action': action, 'observation': observation.strip()})
        if to_print:
            print(f"\n--- Step {i+1}/{max_steps} ---")
            print(f"Action: {action}")
            print(f"Observation:\n    " + "\n    ".join(observation.strip().split('\n')))

        if done:
            if to_print: print(f"Episode finished with reward: {reward}")
            return reward, trajectory, llm_calls

        if i == 0:
            prompt_history = f"Observation: {observation}\n\nAction:"
        else:
            prompt_history += f" {action}\nObservation: {observation}\n\nAction:"

        full_prompt = initial_prompt + prompt_history[-(6000 - len(initial_prompt)):]
        action = call_llm(full_prompt, stop=['\n'], num_traces=num_traces).strip()
        llm_calls += 1
        if not action: break

    # If max steps reached without completion
    if to_print: print(f"Max steps reached. Ending episode with reward: {reward}")
    return reward, trajectory, llm_calls
