from ..llm import call_llm

# --- Prompt Loading ---
with open('webshop-agent/prompts/react_few_shot.txt', 'r') as f:
    FEW_SHOT_PROMPT = f.read()

# --- ReAct Agent Logic ---
def run_single_trace(env, session_id, instruction, long_term_memory: str = "", to_print=True, max_steps=15, num_traces=1):
    """
    Run a single reasoning trace for a WebShop task, with optional long-term memory.
    """
    action = 'reset'

    # Construct the initial prompt, including long-term memory if available
    initial_prompt = f"{FEW_SHOT_PROMPT}\n"
    if long_term_memory:
        initial_prompt += f"You have failed on this task before. Here are your reflections from past trials to help you succeed now:\n---BEGIN REFLECTIONS---\n{long_term_memory}\n---END REFLECTIONS---\n\n"
    initial_prompt += f"Instruction: {instruction}\n[Search]\n"

    prompt_history = ''
    trajectory = []
    n_calls = 0

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
            return reward, trajectory, n_calls

        if i == 0:
            prompt_history = f"Observation: {observation}\n\nAction:"
        else:
            prompt_history += f" {action}\nObservation: {observation}\n\nAction:"

        full_prompt = initial_prompt + prompt_history[-(6000 - len(initial_prompt)):]
        action = call_llm(full_prompt, stop=['\n'], num_traces=num_traces).strip()
        n_calls += 1
        if not action: break

    # If the loop finishes without 'done', it implies max steps were reached.
    # The final reward is the one from the last successful env.step() call.
    # We need to ensure the environment is queried for the final state if it hasn't been.
    # However, the current loop structure already handles this.
    if to_print: print(f"Max steps reached. Ending episode with reward: {reward}")
    return reward, trajectory, n_calls
