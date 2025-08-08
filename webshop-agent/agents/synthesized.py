from agents.single_trace import run_single_trace

def synthesize_decision_deterministic(all_traces_info):
    """
    Deterministically selects the best trajectory based on reward and step count.
    Returns the trajectory number (1-indexed) of the best trajectory.
    """
    if not all_traces_info:
        return 1

    trajectory_metrics = [
        (
            trace.get('final_reward', 0.0),
            len(trace.get('trajectory', [])),
            i + 1
        )
        for i, trace in enumerate(all_traces_info)
    ]
    trajectory_metrics.sort(key=lambda x: (-x[0], x[1]))
    return trajectory_metrics[0][2]

def run_synthesized_episode(env, session_id, instruction, num_traces=3, to_print=True, max_steps=15):
    """
    Runs multiple reasoning traces and synthesizes the best one.
    """
    if num_traces <= 0:
        return 0.0, {'error': 'Invalid num_traces'}

    all_traces_info = []
    if to_print:
        print(f"Running {num_traces} trace(s) for session {session_id} (Synthesized Mode)")

    for i in range(num_traces):
        if num_traces > 1 and to_print:
            print(f"\n=== TRACE {i + 1}/{num_traces} ===")

        # We need to reset the environment for each trace to start from the same initial state
        env.reset(session_id)

        reward, trajectory, llm_calls = run_single_trace(
            env, session_id, instruction, to_print=to_print, max_steps=max_steps
        )
        all_traces_info.append({
            'attempt_num': i + 1,
            'llm_calls': llm_calls,
            'trajectory': trajectory,
            'reward': reward
        })
        if num_traces > 1 and to_print:
            print(f"Trace {i + 1} completed with reward: {reward}")

    if not all_traces_info:
        return 0.0, {'error': 'No traces completed'}

    if to_print:
        print(f"\n=== SYNTHESIZING {num_traces} TRACES ===")

    best_trajectory_num = synthesize_decision_deterministic(all_traces_info)
    best_trace_info = all_traces_info[best_trajectory_num - 1]
    final_reward = best_trace_info['reward']

    if to_print:
        print(f"Synthesis selected trajectory {best_trajectory_num} with reward: {final_reward}")

    return final_reward, {
        'total_attempts': num_traces,
        'synthesized_decision': best_trajectory_num,
        'attempt_details': all_traces_info,
        'final_reward': final_reward,
    }
