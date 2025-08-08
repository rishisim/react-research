import argparse
import json
import random
import re
import os
from dotenv import load_dotenv

from agents.single_trace import run_single_trace
from agents.synthesized import run_synthesized_episode
from agents.reflexion import run_reflexion_episode
from webshop_env import WebShopEnv, webshop_text
from utils import (
    append_to_json,
    get_all_processed_indices,
    get_processed_instructions,
    save_processed_instruction,
    OUTPUT_FILES,
)

# Load environment variables (now handled in llm.py)
# load_dotenv is called in llm.py before client initialization

def run_task_with_all_modes(env, task_index, instruction):
    """Run a single task with all three modes: Standard, Synthesized, and Reflexion."""
    session_id = str(task_index)
    results = {}
    max_steps = 15

    # 1. Standard ReAct (1 trace)
    print(f"\n[STANDARD REACT] Running for Session {session_id}")
    print("="*50)
    try:
        # For single trace, we call it directly, not through synthesized agent
        reward, trajectory, llm_calls = run_single_trace(env, session_id, instruction, to_print=True, max_steps=max_steps)
        info = {'trajectory': trajectory, 'llm_calls': llm_calls}
        results['standard'] = {'reward': reward, 'info': info, 'success': True}
        print(f"Standard ReAct completed with reward: {reward}")
    except Exception as e:
        print(f"Standard ReAct failed: {e}")
        results['standard'] = {'reward': 0.0, 'info': {'trajectory': [{'error': str(e)}]}, 'success': False, 'error': str(e)}

    # 2. Synthesized ReAct (3 traces)
    print(f"\n[SYNTHESIZED REACT] Running for Session {session_id}")
    print("="*50)
    try:
        reward, info = run_synthesized_episode(env, session_id, instruction, num_traces=3, to_print=True, max_steps=max_steps)
        results['synthesized'] = {'reward': reward, 'info': info, 'success': True}
        print(f"Synthesized ReAct completed with reward: {reward}")
    except Exception as e:
        print(f"Synthesized ReAct failed: {e}")
        results['synthesized'] = {'reward': 0.0, 'info': {'attempt_details': [{'error': str(e)}]}, 'success': False, 'error': str(e)}

    # 3. Reflexion ReAct (up to 3 traces with reflection)
    print(f"\n[REFLEXION REACT] Running for Session {session_id}")
    print("="*50)
    try:
        reward, info = run_reflexion_episode(env, session_id, instruction, max_traces=3, to_print=True, max_steps=max_steps)
        results['reflexion'] = {'reward': reward, 'info': info, 'success': True}
        print(f"Reflexion ReAct completed with reward: {reward}")
    except Exception as e:
        print(f"Reflexion ReAct failed: {e}")
        results['reflexion'] = {'reward': 0.0, 'info': {'attempt_details': [{'error': str(e)}]}, 'success': False, 'error': str(e)}

    return results

def run_task_reflexion_only(env, task_index, instruction):
    """Run a single task with only Reflexion mode."""
    session_id = str(task_index)
    results = {}
    max_steps = 15

    # Only Reflexion ReAct (up to 3 traces with reflection)
    print(f"\n[REFLEXION REACT] Running for Session {session_id}")
    print("="*50)
    try:
        reward, info = run_reflexion_episode(env, session_id, instruction, max_traces=3, to_print=True, max_steps=max_steps)
        results['reflexion'] = {'reward': reward, 'info': info, 'success': True}
        print(f"Reflexion ReAct completed with reward: {reward}")
    except Exception as e:
        print(f"Reflexion ReAct failed: {e}")
        results['reflexion'] = {'reward': 0.0, 'info': {'attempt_details': [{'error': str(e)}]}, 'success': False, 'error': str(e)}

    return results

def main():
    parser = argparse.ArgumentParser(description="Run ReAct agent variations on the WebShop environment.")
    parser.add_argument("--num_episodes", type=int, default=5, help="Number of tasks to attempt.")
    parser.add_argument("--reflexion_only", action="store_true", help="Run only reflexion agent (skip standard and synthesized)")
    args = parser.parse_args()

    env = WebShopEnv()

    if args.reflexion_only:
        print("Running ONLY Reflexion ReAct for each task.")
    else:
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
            if args.reflexion_only:
                results = run_task_reflexion_only(env, task_index, instruction)
            else:
                results = run_task_with_all_modes(env, task_index, instruction)

            # Save results based on mode
            if args.reflexion_only:
                # Save only Reflexion ReAct results
                reflexion_data = results.get('reflexion', {})
                reflexion_info = reflexion_data.get('info', {})
                reflexion_episode_data = {
                    'session_id_index': task_index, 'instruction': instruction,
                    'final_reward': reflexion_data.get('reward', 0.0),
                    'total_attempts': reflexion_info.get('total_attempts', 0),
                    'attempt_details': reflexion_info.get('attempt_details', [{'error': reflexion_data.get('error', 'Unknown error')}])
                }
                append_to_json(reflexion_episode_data, OUTPUT_FILES['reflexion'])
                
                print(f"\nResults for task {task_index} saved.")
                print(f"  - Reflexion Reward: {reflexion_data.get('reward', 0.0)}")
            else:
                # Save Standard ReAct results
                std_data = results.get('standard', {})
                std_info = std_data.get('info', {})
                std_episode_data = {
                    'session_id_index': task_index, 'instruction': instruction,
                    'final_reward': std_data.get('reward', 0.0),
                    'trajectory': std_info.get('trajectory', [{'error': std_data.get('error', 'Unknown error')}]),
                    'llm_calls': std_info.get('llm_calls', 0)
                }
                append_to_json(std_episode_data, OUTPUT_FILES['standard'])

                # Save Synthesized ReAct results
                synth_data = results.get('synthesized', {})
                synth_info = synth_data.get('info', {})
                synth_episode_data = {
                    'session_id_index': task_index, 'instruction': instruction,
                    'final_reward': synth_data.get('reward', 0.0),
                    'total_attempts': synth_info.get('total_attempts', 3),
                    'synthesized_decision': synth_info.get('synthesized_decision', 1),
                    'attempt_details': synth_info.get('attempt_details', [{'error': synth_data.get('error', 'Unknown error')}])
                }
                append_to_json(synth_episode_data, OUTPUT_FILES['synthesized'])

                # Save Reflexion ReAct results
                reflexion_data = results.get('reflexion', {})
                reflexion_info = reflexion_data.get('info', {})
                reflexion_episode_data = {
                    'session_id_index': task_index, 'instruction': instruction,
                    'final_reward': reflexion_data.get('reward', 0.0),
                    'total_attempts': reflexion_info.get('total_attempts', 0),
                    'attempt_details': reflexion_info.get('attempt_details', [{'error': reflexion_data.get('error', 'Unknown error')}])
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
