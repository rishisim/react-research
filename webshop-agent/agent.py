from .llm import call_llm

# --- ReAct Agent Logic ---
with open('prompts/react_few_shot.txt', 'r') as f:
    FEW_SHOT_PROMPT = f.read()

def run_single_trace(env, session_id, instruction, to_print=True, max_steps=15, num_traces=1):
    """Run a single reasoning trace for WebShop task."""
    action = 'reset'
    initial_prompt = f"{FEW_SHOT_PROMPT}\nInstruction: {instruction}\n[Search]\n"
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

        if i == 0: prompt_history = f"Observation: {observation}\n\nAction:"
        else: prompt_history += f" {action}\nObservation: {observation}\n\nAction:"

        full_prompt = initial_prompt + prompt_history[-(6000 - len(initial_prompt)):]
        action = call_llm(full_prompt, stop=['\n'], num_traces=num_traces).strip()
        n_calls += 1
        if not action: break

    if to_print: print(f"Max steps reached. Ending episode with reward: {reward}")
    return reward, trajectory, n_calls

def run_single_episode(env, session_id, instruction, to_print=True, max_steps=15):
    """Backward compatibility wrapper for single episode execution."""
    reward, trajectory, _ = run_single_trace(env, session_id, instruction, to_print, max_steps, num_traces=1)
    return reward, trajectory
