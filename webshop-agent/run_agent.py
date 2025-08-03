import argparse
import json
import os
import random
import re
import sys
import time
import requests
from bs4 import BeautifulSoup, Comment
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load environment variables from .env file
load_dotenv()

# --- Gemini API Configuration ---
try:
    client = genai.Client()
except Exception as e:
    print(f"ERROR: Failed to initialize Gemini client: {e}")
    sys.exit(1)

def llm(prompt, stop=None, num_traces=1):
    if stop is None: stop = ["\n"]
    time.sleep(10)
    temperature_setting = 0.0 if num_traces == 1 else 0.7
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_budget=0),
                stop_sequences=stop,
                temperature=temperature_setting,
                max_output_tokens=300,
            )
        )
        return response.text
    except Exception as e:
        print(f"Error calling LLM: {e}")
        return ""

# --- WebShop Environment Interaction ---
WEBSHOP_URL = "http://localhost:3000"
ACTION_TO_TEMPLATE = {
    'Description': 'description_page.html', 'Features': 'features_page.html',
    'Reviews': 'review_page.html', 'Attributes': 'attributes_page.html',
}

def clean_str(p):
    return p.encode().decode("unicode-escape").encode("latin1").decode("utf-8")

def tag_visible(element):
    ignore = {'style', 'script', 'head', 'title', 'meta', '[document]'}
    return element.parent.name not in ignore and not isinstance(element, Comment)

def webshop_text(session, page_type, **kwargs):
    options = kwargs.get('options', {})
    try:
        url_map = {
            'init': f'{WEBSHOP_URL}/{session}',
            'search': f'{WEBSHOP_URL}/search_results/{session}/{kwargs.get("query_string", "")}/{kwargs.get("page_num", 1)}',
            'item': f'{WEBSHOP_URL}/item_page/{session}/{kwargs.get("asin", "")}/{kwargs.get("query_string", "")}/{kwargs.get("page_num", 1)}/{options}',
            'item_sub': f'{WEBSHOP_URL}/item_sub_page/{session}/{kwargs.get("asin", "")}/{kwargs.get("query_string", "")}/{kwargs.get("page_num", 1)}/{kwargs.get("subpage", "")}/{options}',
            'end': f'{WEBSHOP_URL}/done/{session}/{kwargs.get("asin", "")}/{options}'
        }
        url = url_map.get(page_type)
        if not url: raise ValueError(f"Invalid page_type: {page_type}")

        html = requests.get(url).text
        html_obj = BeautifulSoup(html, 'html.parser')
        texts = html_obj.find_all(string=True)
        visible_texts = list(filter(tag_visible, texts))
        
        # Match notebook formatting exactly
        observation = ''
        option_type = ''
        page_options = {}
        asins = []
        cnt = 0
        prod_cnt = 0
        just_prod = 0
        
        for t in visible_texts:
            if t == '\n': continue
            if t.replace('\n', '').replace('\\n', '').replace(' ', '') == '': continue
            
            if t.parent.name == 'button':  # button
                processed_t = f'\n[{t}] '
            elif t.parent.name == 'label':  # options
                if f"'{t}'" in url:
                    processed_t = f'[[{t}]]'
                else:
                    processed_t = f'[{t}]'
                page_options[str(t)] = option_type
            elif t.parent.get('class') == ["product-link"]: # product asins
                processed_t = f'\n[{t}] '
                if prod_cnt >= 3:
                    processed_t = ''
                prod_cnt += 1
                asins.append(str(t))
                just_prod = 0
            else: # regular, unclickable text
                processed_t = '\n' + str(t) + ' '
                if cnt < 2 and page_type != 'init': processed_t = ''
                if just_prod <= 2 and prod_cnt >= 4: processed_t = ''
                option_type = str(t)
                cnt += 1
            just_prod += 1
            observation += processed_t
        
        info = {'asins': asins, 'option_types': page_options}
        if 'Your score (min 0.0, max 1.0)' in visible_texts:
            idx = visible_texts.index('Your score (min 0.0, max 1.0)')
            info['reward'] = float(visible_texts[idx + 1])
            observation = 'Your score (min 0.0, max 1.0): ' + (visible_texts[idx + 1])
        return clean_str(observation), info
    except requests.exceptions.RequestException as e:
        return f"Error connecting to WebShop: {e}", {'error': str(e)}

class WebShopEnv:
    def __init__(self):
        self.sessions = {}

    def step(self, session, action):
        done = False
        observation_ = None
        action_type = action.split('[')[0]

        if action_type == 'reset':
            self.sessions[session] = {'session': session, 'page_type': 'init'}  # Match notebook format
        elif action_type == 'think': pass
        elif action_type == 'search':
            assert self.sessions[session]['page_type'] == 'init'
            query = action[7:-1]
            self.sessions[session] = {'session': session, 'page_type': 'search', 'query_string': query, 'page_num': 1}
        elif action_type == 'click':
            button = action[6:-1]
            page_type = self.sessions[session]['page_type']
            if button == 'Buy Now':
                assert page_type == 'item'
                self.sessions[session]['page_type'] = 'end'
                done = True
            elif button == 'Back to Search':
                assert page_type in ['search', 'item_sub', 'item']
                self.sessions[session] = {'session': session, 'page_type': 'init'}
            elif button == '< Prev':
                assert page_type in ['search', 'item_sub', 'item']
                if page_type == 'item_sub': 
                    self.sessions[session]['page_type'] = 'item'
                elif page_type == 'item': 
                    self.sessions[session]['page_type'] = 'search'
                    self.sessions[session]['options'] = {}  # Clear options when going back
            elif button == 'Next >':
                assert page_type == 'search'
                self.sessions[session]['page_num'] += 1
            elif button in ACTION_TO_TEMPLATE:
                assert page_type == 'item'  # Only from main item page
                self.sessions[session]['page_type'] = 'item_sub'
                self.sessions[session]['subpage'] = button
            else:
                if page_type == 'search':
                    assert button in self.sessions[session].get('asins', [])  # must be asins
                    self.sessions[session]['page_type'] = 'item'
                    self.sessions[session]['asin'] = button
                elif page_type == 'item':
                    assert 'option_types' in self.sessions[session]
                    assert button in self.sessions[session]['option_types'], (button, self.sessions[session]['option_types'])  # must be options
                    option_type = self.sessions[session]['option_types'][button]
                    if 'options' not in self.sessions[session]: 
                        self.sessions[session]['options'] = {}
                    self.sessions[session]['options'][option_type] = button
                    observation_ = f'You have clicked {button}.'
        else:
            assert False, f"Invalid action format: {action}"

        observation, info = webshop_text(session=session, **{k:v for k,v in self.sessions[session].items() if k != 'session'})
        if 'error' in info: return observation, 0.0, True
        if observation_: observation = observation_
        self.sessions[session].update(info)
        return observation, info.get('reward', 0.0), done

# --- ReAct Agent Logic ---
FEW_SHOT_PROMPT = """Webshop
Instruction: i would like a 3 ounce bottle of bright citrus deodorant for sensitive skin, and price lower than 50.00 dollars
[Search]

Action: search[3 ounce bright citrus deodorant sensitive skin]
Observation:
[Back to Search]
Page 1 (Total results: 50)
[B078GWRC1J]
Bright Citrus Deodorant by Earth Mama | Natural and Safe for Sensitive Skin...
$10.99
Action: think[B078GWRC1J is a bright citrus deodorant that is less then 50 dollars. I can check it.]
Observation: OK.
Action: click[B078GWRC1J]
Observation:
[Back to Search]
scent [assorted scents][bright citrus]
size [3 ounce (pack of 1)]
Price: $10.99
[Buy Now]
Action: think[The item has options 'bright citrus' and '3 ounce (pack of 1)' and seems good to buy.]
Observation: OK.
Action: click[bright citrus]
Observation: You have clicked bright citrus.
Action: click[3 ounce (pack of 1)]
Observation: You have clicked 3 ounce (pack of 1).
Action: click[Buy Now]

----

Webshop
Instruction: i need a 1 gallon jug of unscented laundry detergent, and price lower than 25.00 dollars
[Search]

Action: search[1 gallon unscented laundry detergent]
Observation:
[Back to Search]
Page 1 (Total results: 50)
[B07FF6259C]
Molly's Suds Unscented Laundry Detergent Powder... 120 Loads
$21.99
[B08149V812]
ECOS Hypoallergenic Laundry Detergent, Unscented, 1 Gallon
$18.99

Action: think[B07FF6259C is unscented and under $25, but it is a powder, not a 1 gallon jug. I will check it anyway to see the options.]
Observation: OK.
Action: click[B07FF6259C]
Observation:
[Back to Search]
Molly's Suds Unscented Laundry Detergent Powder...
Price: $21.99
[Description]
This is a super-concentrated powder laundry detergent.
[Buy Now]

Action: think[This is a powder detergent, which is not what the instruction asked for. I need to go back to the search results to find a liquid 1-gallon jug.]
Observation: OK.
Action: click[Back to Search]
Observation:
[Back to Search]
Page 1 (Total results: 50)
[B07FF6259C]
Molly's Suds Unscented Laundry Detergent Powder... 120 Loads
$21.99
[B08149V812]
ECOS Hypoallergenic Laundry Detergent, Unscented, 1 Gallon
$18.99

Action: think[B08149V812 is a 1 gallon, unscented detergent and the price is good. This looks like the right item.]
Observation: OK.
Action: click[B08149V812]
Observation:
[Back to Search]
ECOS Hypoallergenic Laundry Detergent, Unscented, 1 Gallon
Price: $18.99
[Buy Now]

Action: click[Buy Now]
"""

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
        action = llm(full_prompt, stop=['\n'], num_traces=num_traces).strip()
        n_calls += 1
        if not action: break
    
    if to_print: print(f"Max steps reached. Ending episode with reward: {reward}")
    return reward, trajectory, n_calls

def run_single_episode(env, session_id, instruction, to_print=True, max_steps=15):
    """Backward compatibility wrapper for single episode execution."""
    reward, trajectory, _ = run_single_trace(env, session_id, instruction, to_print, max_steps, num_traces=1)
    return reward, trajectory

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

# --- File Utilities & Experiment Runner ---
def append_to_json(data, filename):
    if os.path.exists(filename) and os.path.getsize(filename) > 0:
        with open(filename, 'r+', encoding='utf-8') as f:
            try: file_data = json.load(f)
            except json.JSONDecodeError: file_data = []
            if isinstance(file_data, list): file_data.append(data)
            else: file_data = [data]
            f.seek(0)
            json.dump(file_data, f, indent=2)
            f.truncate()
    else:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump([data], f, indent=2)

def get_processed_indices(output_file):
    processed_indices = set()
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                for entry in data:
                    if 'session_id_index' in entry:
                        processed_indices.add(entry['session_id_index'])
            except (json.JSONDecodeError, AttributeError): pass
    return processed_indices

def get_processed_instructions(instruction_file='used_instructions.json'):
    """
    Get all instructions that have been processed so far.
    Similar to get_processed_indices but for instructions.
    Loads from a persistent file that tracks used instructions.
    """
    processed_instructions = set()
    
    # Load from persistent instruction tracking file
    if os.path.exists(instruction_file):
        try:
            with open(instruction_file, 'r', encoding='utf-8') as f:
                instruction_list = json.load(f)
                processed_instructions = set(instruction_list)
        except (json.JSONDecodeError, AttributeError):
            pass
    
    # Also load from existing trajectory files (in case file doesn't exist yet)
    trajectory_files = ['webshop_trajectories.json', 'webshop_synthesized_trajectories.json']
    for filename in trajectory_files:
        if os.path.exists(filename):
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for entry in data:
                        if 'instruction' in entry:
                            processed_instructions.add(entry['instruction'])
            except (json.JSONDecodeError, AttributeError):
                pass
    
    return processed_instructions

def save_processed_instruction(instruction, instruction_file='used_instructions.json'):
    """
    Save a newly processed instruction to the persistent tracking file.
    Similar to how session indices are tracked, but for instructions.
    """
    processed_instructions = get_processed_instructions(instruction_file)
    processed_instructions.add(instruction)
    
    # Save back to file
    try:
        with open(instruction_file, 'w', encoding='utf-8') as f:
            json.dump(list(processed_instructions), f, indent=2)
    except Exception as e:
        print(f"Warning: Could not save instruction to {instruction_file}: {e}")

def get_output_filename(num_traces):
    """Get appropriate filename based on whether using synthesis or not."""
    if num_traces == 1:
        return 'webshop_trajectories.json'  # Standard ReAct
    else:
        return 'webshop_synthesized_trajectories.json'  # Synthesized ReAct

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
