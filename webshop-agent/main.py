import argparse
import json
import os
import random
import re
from dotenv import load_dotenv

from agent import run_single_trace, run_reflexion_episode
from webshop_env import WebShopEnv, webshop_text
from utils import (
    append_to_json,
    get_all_processed_indices,
    get_processed_instructions,
    save_processed_instruction,
    OUTPUT_FILES,
)

# Load environment variables from .env file
load_dotenv()

# --- Synthesis Functions for Multi-Trace ---
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

# --- Core Webthink Logic ---
def webthink_webshop(env, session_id, instruction, num_traces=1, to_print=True, max_steps=15):
    """
    Main function for WebShop reasoning. Handles both standard (1 trace) and synthesized (>1 trace) ReAct.
    """
    if num_traces <= 0:
        return 0.0, {'error': 'Invalid num_traces'}

    all_traces_info = []
    print(f"Running {num_traces} trace(s) for session {session_id} (Synthesized Mode)")
    for i in range(num_traces):
        if num_traces > 1: print(f"\n=== TRACE {i + 1}/{num_traces} ===")
        reward, trajectory, n_calls = run_single_trace(env, session_id, instruction, to_print=to_print, max_steps=max_steps)
        all_traces_info.append({'trace_num': i + 1, 'n_calls': n_calls, 'trajectory': trajectory, 'final_reward': reward})
        if num_traces > 1: print(f"Trace {i + 1} completed with reward: {reward}")

    if not all_traces_info:
        return 0.0, {'error': 'No traces completed'}

    if num_traces == 1:
        return all_traces_info[0]['final_reward'], all_traces_info[0]
    else:
        print(f"\n=== SYNTHESIZING {num_traces} TRACES ===")
        best_trajectory_num = synthesize_decision_deterministic(all_traces_info)
        best_trace_info = all_traces_info[best_trajectory_num - 1]
        final_reward = best_trace_info['final_reward']
        print(f"Synthesis selected trajectory {best_trajectory_num} with reward: {final_reward}")
        return final_reward, {
            'num_traces_run': num_traces,
            'synthesized_decision': best_trajectory_num,
            'individual_traces': all_traces_info,
            'final_reward': final_reward,
        }

def run_task_with_all_modes(env, task_index, instruction):
    """Run a single task with all three modes: Standard, Synthesized, and Reflexion."""
    session_id = str(task_index)
    results = {}

    # 1. Standard ReAct (1 trace)
    print(f"\n[STANDARD REACT] Running for Session {session_id}")
    print("="*50)
    try:
        reward, info = webthink_webshop(env, session_id, instruction, num_traces=1, to_print=True)
        results['standard'] = {'reward': reward, 'info': info, 'success': True}
        print(f"Standard ReAct completed with reward: {reward}")
    except Exception as e:
        print(f"Standard ReAct failed: {e}")
        results['standard'] = {'reward': 0.0, 'info': {'trajectory': [{'error': str(e)}]}, 'success': False, 'error': str(e)}

    # 2. Synthesized ReAct (3 traces)
    print(f"\n[SYNTHESIZED REACT] Running for Session {session_id}")
    print("="*50)
    try:
        reward, info = webthink_webshop(env, session_id, instruction, num_traces=3, to_print=True)
        results['synthesized'] = {'reward': reward, 'info': info, 'success': True}
        print(f"Synthesized ReAct completed with reward: {reward}")
    except Exception as e:
        print(f"Synthesized ReAct failed: {e}")
        results['synthesized'] = {'reward': 0.0, 'info': {'individual_traces': [{'error': str(e)}]}, 'success': False, 'error': str(e)}

    # 3. Reflexion ReAct (up to 3 traces with reflection)
    print(f"\n[REFLEXION REACT] Running for Session {session_id}")
    print("="*50)
    try:
        reward, info = run_reflexion_episode(env, session_id, instruction, max_traces=3, to_print=True)
        results['reflexion'] = {'reward': reward, 'info': info, 'success': True}
        print(f"Reflexion ReAct completed with reward: {reward}")
    except Exception as e:
        print(f"Reflexion ReAct failed: {e}")
        results['reflexion'] = {'reward': 0.0, 'info': {'individual_traces': [{'error': str(e)}]}, 'success': False, 'error': str(e)}

    return results

def main():
    parser = argparse.ArgumentParser(description="Run ReAct agent variations on the WebShop environment.")
    parser.add_argument("--num_episodes", type=int, default=5, help="Number of tasks to attempt.")
    args = parser.parse_args()

    env = WebShopEnv()

    print("Running Standard, Synthesized, and Reflexion ReAct for each task.")

    MAX_WEBSHOP_TASKS = 699
    all_indices = list(range(MAX_WEBSHOP_TASKS))
    random.Random(42).shuffle(all_indices)

    processed_indices = get_all_processed_indices()
    print(f"Found {len(processed_indices)} previously completed tasks across all modes.")
    processed_instructions = get_processed_instructions()
    print(f"Found {len(processed_instructions)} previously used instructions.")

    remaining_indices = [idx for idx in all_indices if idx not in processed_indices]
    tasks_completed_this_session = 0
    duplicates_found_this_session = 0
    num_episodes_to_attempt = min(args.num_episodes, len(remaining_indices))

    print(f"\nAttempting to run up to {num_episodes_to_attempt} new tasks...")

    while tasks_completed_this_session < num_episodes_to_attempt:
        if not remaining_indices:
            print("No more available tasks to process. Stopping.")
            break

        task_index = remaining_indices.pop(0)
        session_id = str(task_index)

        try:
            obs, _ = webshop_text(session=session_id, page_type='init')
            instruction_match = re.search(r"Instruction:\s*(.*)", obs, re.DOTALL)
            if not instruction_match:
                print(f"Could not find instruction for session {session_id}. Skipping.")
                continue
            instruction = instruction_match.group(1).strip()
        except Exception as e:
            print(f"Critical error fetching instruction for {session_id}: {e}. Skipping.")
            continue

        if instruction in processed_instructions:
            duplicates_found_this_session += 1
            print(f"DUPLICATE INSTRUCTION: Task {task_index} is a duplicate. Skipping.")
            save_processed_instruction(instruction)
            continue

        print('\n' + '='*60)
        print(f"RUNNING TASK {tasks_completed_this_session + 1}/{num_episodes_to_attempt} (Session: {session_id})")
        print(f"Instruction: {instruction}")
        print('='*60)

        save_processed_instruction(instruction)
        processed_instructions.add(instruction)

        try:
            results = run_task_with_all_modes(env, task_index, instruction)

            # Save Standard ReAct results
            std_data = results.get('standard', {})
            std_info = std_data.get('info', {})
            std_episode_data = {
                'session_id_index': task_index, 'instruction': instruction,
                'final_reward': std_data.get('reward', 0.0),
                'trajectory': std_info.get('trajectory', [{'error': std_data.get('error', 'Unknown error')}]),
                'n_calls': std_info.get('n_calls', 0)
            }
            append_to_json(std_episode_data, OUTPUT_FILES['standard'])

            # Save Synthesized ReAct results
            synth_data = results.get('synthesized', {})
            synth_info = synth_data.get('info', {})
            synth_episode_data = {
                'session_id_index': task_index, 'instruction': instruction,
                'final_reward': synth_data.get('reward', 0.0),
                'num_traces_run': synth_info.get('num_traces_run', 3),
                'synthesized_decision': synth_info.get('synthesized_decision', 1),
                'individual_traces': synth_info.get('individual_traces', [{'error': synth_data.get('error', 'Unknown error')}])
            }
            append_to_json(synth_episode_data, OUTPUT_FILES['synthesized'])

            # Save Reflexion ReAct results
            reflexion_data = results.get('reflexion', {})
            reflexion_info = reflexion_data.get('info', {})
            reflexion_episode_data = {
                'session_id_index': task_index, 'instruction': instruction,
                'final_reward': reflexion_data.get('reward', 0.0),
                'num_traces_run': reflexion_info.get('num_traces_run', 0),
                'individual_traces': reflexion_info.get('individual_traces', [{'error': reflexion_data.get('error', 'Unknown error')}])
            }
            append_to_json(reflexion_episode_data, OUTPUT_FILES['reflexion'])

            print(f"\nResults for task {task_index} saved.")
            print(f"  - Standard Reward: {std_data.get('reward', 0.0)}")
            print(f"  - Synthesized Reward: {synth_data.get('reward', 0.0)}")
            print(f"  - Reflexion Reward: {reflexion_data.get('reward', 0.0)}")

            tasks_completed_this_session += 1
        except Exception as e:
            print(f"An unrecoverable error occurred during task execution for {session_id}: {e}")

    print("\n" + "="*60 + "\nRUN SUMMARY\n" + "="*60)
    print(f"Attempted {tasks_completed_this_session} new tasks.")
    print(f"Skipped {duplicates_found_this_session} duplicate instructions.")
    print("="*60)

if __name__ == "__main__":
    main()
