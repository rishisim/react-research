import argparse
import json
import os
import random
import re
import sys
from dotenv import load_dotenv

from agent import run_single_trace
from webshop_env import WebShopEnv, webshop_text
from utils import (
    append_to_json,
    get_processed_indices,
    get_processed_instructions,
    save_processed_instruction,
    get_output_filename,
)

# Load environment variables from .env file
load_dotenv()

# --- Synthesis Functions for Multi-Trace ---
def extract_trajectories_from_traces(all_traces_info):
    """Extract the full trajectory string from each trace in the all_traces_info list."""
    extracted_trajectories = []
    if not isinstance(all_traces_info, list):
        print("Warning: all_traces_info is not a list")
        return []

    for i, trace_info in enumerate(all_traces_info):
        if isinstance(trace_info, dict) and 'trajectory' in trace_info:
            trajectory = trace_info['trajectory']
            # Convert trajectory to string format
            trajectory_str = ""
            for step in trajectory:
                trajectory_str += f"Action: {step['action']}\nObservation: {step['observation']}\n"
            extracted_trajectories.append(trajectory_str.strip())
        else:
            print(f"Warning: trace_info {i} missing 'trajectory' key or not a dict")
            extracted_trajectories.append("")

    return extracted_trajectories

def synthesize_decision_deterministic(all_traces_info):
    """
    Deterministically selects the best trajectory based on reward and step count.
    Returns the trajectory number (1-indexed) of the best trajectory.

    Selection criteria:
    1. Highest reward
    2. If tied, fewest steps (most efficient)
    """
    if not all_traces_info:
        return 1

    # Extract reward and step count for each trajectory
    trajectory_metrics = []
    for i, trace_info in enumerate(all_traces_info):
        if isinstance(trace_info, dict):
            reward = trace_info.get('final_reward', 0.0)
            trajectory = trace_info.get('trajectory', [])
            step_count = len(trajectory) if trajectory else float('inf')
            trajectory_metrics.append((reward, step_count, i + 1))  # (reward, steps, trajectory_num)
        else:
            trajectory_metrics.append((0.0, float('inf'), i + 1))

    # Sort by reward (descending), then by step count (ascending)
    trajectory_metrics.sort(key=lambda x: (-x[0], x[1]))

    return trajectory_metrics[0][2]  # Return trajectory number

# --- Core Webthink Logic ---
def webthink_webshop(env, session_id, instruction, num_traces=1, to_print=True, max_steps=15):
    """
    Main function for WebShop reasoning with support for both single and multi-trace execution.

    Args:
        env: WebShop environment instance
        session_id: Session identifier
        instruction: Shopping instruction
        num_traces: Number of reasoning traces to run (1 for standard ReAct, >1 for synthesized)
        to_print: Whether to print trace details
        max_steps: Maximum steps per trace

    Returns:
        (reward, info_dict): Final reward and detailed information
    """
    if num_traces <= 0:
        print("Error: num_traces must be positive")
        return 0.0, {'error': 'Invalid num_traces'}

    all_traces_info = []

    print(f"Running {num_traces} trace(s) for session {session_id}")

    for trace_num in range(num_traces):
        if num_traces > 1:
            print(f"\n=== TRACE {trace_num + 1}/{num_traces} ===")

        # Run single trace
        reward, trajectory, n_calls = run_single_trace(env, session_id, instruction, to_print, max_steps, num_traces)

        # Store trace information
        trace_info = {
            'trace_num': trace_num + 1,
            'n_calls': n_calls,
            'trajectory': trajectory,
            'final_reward': reward
        }
        all_traces_info.append(trace_info)

        if num_traces > 1:
            print(f"Trace {trace_num + 1} completed with reward: {reward}")

    if not all_traces_info:
        return 0.0, {'error': 'No traces completed'}

    if num_traces == 1:
        # Single trace - return as before
        trace_info = all_traces_info[0]
        return trace_info['final_reward'], {
            'n_calls': trace_info['n_calls'],
            'trajectory': trace_info['trajectory'],
            'final_reward': trace_info['final_reward']
        }
    else:
        # Multi-trace synthesis
        print(f"\n=== SYNTHESIZING {num_traces} TRACES ===")

        # Synthesize best decision deterministically
        best_trajectory_num = synthesize_decision_deterministic(all_traces_info)

        # Use reward from best trajectory
        best_trace_info = all_traces_info[best_trajectory_num - 1]  # Convert to 0-indexed
        final_reward = best_trace_info['final_reward']

        print(f"Synthesis selected trajectory {best_trajectory_num} with reward: {final_reward}")

        return final_reward, {
            'num_traces_run': num_traces,
            'synthesized_decision': best_trajectory_num,
            'individual_traces': all_traces_info,
            'final_reward': final_reward,
            'synthesis_reasoning': ""
        }

def run_task_with_both_modes(env, task_index, instruction):
    """Run a single task with both standard and synthesized ReAct modes."""
    session_id = str(task_index)
    results = {}

    # Run Standard ReAct (1 trace)
    print(f"\n[STANDARD REACT] Running 1 trace for Session {session_id}")
    print("="*50)
    try:
        reward_std, info_std = webthink_webshop(env, session_id, instruction, num_traces=1, to_print=True)
        results['standard'] = {
            'reward': reward_std,
            'info': info_std,
            'success': True
        }
        print(f"Standard ReAct completed with reward: {reward_std}")
    except Exception as e:
        print(f"Standard ReAct failed: {e}")
        results['standard'] = {
            'reward': 0.0,
            'info': {'trajectory': [{'error': str(e)}]},
            'success': False,
            'error': str(e)
        }

    # Run Synthesized ReAct (3 traces)
    print(f"\n[SYNTHESIZED REACT] Running 3 traces for Session {session_id}")
    print("="*50)
    try:
        reward_synth, info_synth = webthink_webshop(env, session_id, instruction, num_traces=3, to_print=True)
        results['synthesized'] = {
            'reward': reward_synth,
            'info': info_synth,
            'success': True
        }
        print(f"Synthesized ReAct completed with reward: {reward_synth}")
    except Exception as e:
        print(f"Synthesized ReAct failed: {e}")
        results['synthesized'] = {
            'reward': 0.0,
            'info': {'individual_traces': [{'error': str(e)}]},
            'success': False,
            'error': str(e)
        }

    return results

def main():
    parser = argparse.ArgumentParser(description="Run a ReAct agent on the WebShop environment.")
    parser.add_argument("--num_episodes", type=int, default=5, help="Number of tasks to attempt to run.")
    parser.add_argument("--num_traces", type=int, default=None, help="Override to run only one mode: 1=standard ReAct only, 3=synthesized ReAct only")
    args = parser.parse_args()

    env = WebShopEnv()

    # Determine which modes to run
    run_both_modes = args.num_traces is None
    if run_both_modes:
        print("Running BOTH Standard ReAct and Synthesized ReAct for each task")
        standard_file = 'webshop_trajectories.json'
        synthesized_file = 'webshop_synthesized_trajectories.json'
    else:
        print(f"Running only {'Standard' if args.num_traces == 1 else 'Synthesized'} ReAct")
        output_file = get_output_filename(args.num_traces)

    MAX_WEBSHOP_TASKS = 699
    all_indices = list(range(MAX_WEBSHOP_TASKS))
    random.Random(42).shuffle(all_indices)

    # Load all previously completed indices and instructions
    if run_both_modes:
        processed_std = get_processed_indices(standard_file)
        processed_synth = get_processed_indices(synthesized_file)
        processed_indices = processed_std.union(processed_synth)
        print(f"Found {len(processed_std)} standard and {len(processed_synth)} synthesized completed tasks.")
    else:
        processed_indices = get_processed_indices(output_file)
        print(f"Found {len(processed_indices)} already completed tasks in {output_file}.")

    processed_instructions = get_processed_instructions()
    print(f"Found {len(processed_instructions)} previously used instructions.")

    # Get a list of sessions that have not yet been successfully run
    remaining_indices = [idx for idx in all_indices if idx not in processed_indices]

    # Counters for the final summary
    tasks_completed_this_session = 0
    duplicates_found_this_session = 0
    num_episodes_to_attempt = min(args.num_episodes, len(remaining_indices))

    print(f"\nAttempting to run up to {num_episodes_to_attempt} tasks...")

    # Loop until we have run the desired number of tasks OR run out of sessions
    while tasks_completed_this_session < num_episodes_to_attempt:
        if not remaining_indices:
            print("No more available tasks to process. Stopping.")
            break

        # 1. Pick a single, unprocessed session ID from the top of the list
        task_index = remaining_indices.pop(0)
        session_id = str(task_index)

        # 2. Make ONE API call to get its instruction
        try:
            obs, info = webshop_text(session=session_id, page_type='init')
            if 'error' in info:
                print(f"Error fetching page for session {session_id}. Skipping.")
                continue
            instruction_match = re.search(r"Instruction:\s*(.*)", obs, re.DOTALL)
            if not instruction_match:
                print(f"Could not find instruction for session {session_id}. Skipping.")
                continue
            instruction = instruction_match.group(1).strip()
        except Exception as e:
            print(f"Critical error fetching instruction for session {session_id}: {e}. Skipping.")
            continue

        # 3. Check if the instruction is a duplicate
        if instruction in processed_instructions:
            duplicates_found_this_session += 1
            print(f"DUPLICATE FOUND: Session {task_index} has a used instruction. ({duplicates_found_this_session} duplicates so far). Skipping.")
            save_processed_instruction(instruction) # Save to prevent re-checking
            continue

        # --- If we get here, the instruction is UNIQUE ---
        print('\n' + '='*60)
        print(f"RUNNING TASK {tasks_completed_this_session + 1}/{num_episodes_to_attempt} (Session: {session_id})")
        print(f"Instruction: {instruction}")
        print('='*60)

        # Mark this instruction as processed BEFORE running the agent
        save_processed_instruction(instruction)
        processed_instructions.add(instruction)

        try:
            if run_both_modes:
                results = run_task_with_both_modes(env, task_index, instruction)
                # Save standard ReAct results
                std_data = results.get('standard', {})
                std_episode_data = {
                    'session_id_index': task_index, 'instruction': instruction,
                    'final_reward': std_data.get('reward', 0.0),
                    'trajectory': std_data.get('info', {}).get('trajectory', [{'error': std_data.get('error', 'Unknown error')}])
                }
                append_to_json(std_episode_data, standard_file)
                # Save synthesized ReAct results
                synth_data = results.get('synthesized', {})
                synth_info = synth_data.get('info', {})
                synth_episode_data = {
                    'session_id_index': task_index, 'instruction': instruction,
                    'final_reward': synth_data.get('reward', 0.0),
                    'num_traces_run': synth_info.get('num_traces_run', 3),
                    'synthesized_decision': synth_info.get('synthesized_decision', 1),
                    'individual_traces': synth_info.get('individual_traces', [{'error': synth_data.get('error', 'Unknown error')}]),
                    'synthesis_reasoning': synth_info.get('synthesis_reasoning', "")
                }
                append_to_json(synth_episode_data, synthesized_file)
                print(f"\nResults saved (Standard Reward: {std_data.get('reward', 0.0)}, Synthesized Reward: {synth_data.get('reward', 0.0)})")
            else:
                reward, info_dict = webthink_webshop(env, session_id, instruction, num_traces=args.num_traces, to_print=True)
                if args.num_traces == 1:
                    episode_data = {'session_id_index': task_index, 'instruction': instruction, 'final_reward': reward, 'trajectory': info_dict.get('trajectory', [])}
                else:
                    episode_data = {'session_id_index': task_index, 'instruction': instruction, 'final_reward': reward, 'num_traces_run': info_dict.get('num_traces_run', args.num_traces), 'synthesized_decision': info_dict.get('synthesized_decision', 1), 'individual_traces': info_dict.get('individual_traces', []), 'synthesis_reasoning': info_dict.get('synthesis_reasoning', "")}
                append_to_json(episode_data, output_file)
                print(f"Result for session {session_id} saved to {output_file}")

            tasks_completed_this_session += 1
        except Exception as e:
            print(f"An unrecoverable error occurred during task execution for session {session_id}: {e}")
            # The task failed, but we attempted it, so we don't try it again. We just move on.

    # 4. Final summary after all attempts are finished
    print("\n" + "="*60)
    print("RUN SUMMARY")
    print(f"Completed {tasks_completed_this_session} / {num_episodes_to_attempt} attempted tasks in this session.")
    print(f"Found and skipped {duplicates_found_this_session} duplicate instructions.")
    print("="*60)

if __name__ == "__main__":
    main()
