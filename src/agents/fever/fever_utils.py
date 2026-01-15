"""
Shared utilities for FEVER agents.

This module contains common functions used by all FEVER agent frameworks:
- LLM interaction
- Environment setup and interaction
- Prompt loading
- Answer extraction and synthesis
"""

import os
import time
import re
import json
import sys
import requests
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Add shared directory to path for wikienv and wrappers
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../shared')))
import wikienv
import wrappers

# --- LLM Configuration and Interaction ---
from google import genai
from google.genai import types

# The client gets the API key from the environment variable `GEMINI_API_KEY`.
client = genai.Client()


def llm(prompt, stop=["\n"], num_traces=1):
    """
    Call the language model with the given prompt.
    
    Args:
        prompt: The input prompt string
        stop: List of stop sequences
        num_traces: Number of traces (affects temperature setting)
        
    Returns:
        String response from the LLM
    """
    # This delay ensures we don't exceed API rate limits (3 seconds between calls).
    time.sleep(3.0)


    temperature_setting = 0.0 if num_traces == 1 else 0.7
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=0),  # Disables thinking
            stop_sequences=stop,
            temperature=temperature_setting,
            max_output_tokens=512,
            top_p=1.0
        )
    )
    return response.text


# --- Environment Setup ---
env = None


def get_fever_env():
    """Get or initialize the FEVER environment."""
    global env
    if env is None:
        env = wikienv.WikiEnv()
        env = wrappers.FeverWrapper(env, split="dev")
        env = wrappers.LoggingWrapper(env)
    return env


def step(current_env, action):
    """
    Execute a step in the environment with retry logic for timeouts.
    
    Args:
        current_env: The FEVER environment
        action: Action string to execute
        
    Returns:
        Tuple of (observation, reward, done, info)
    """
    attempts = 0
    while attempts < 10:
        try:
            return current_env.step(action)
        except requests.exceptions.Timeout:
            print(f"[WARNING] Timeout during env.step attempt {attempts+1} for action: {action}")
            attempts += 1
            time.sleep(2)  # Wait before retrying
    
    print(f"[ERROR] Failed to execute step after 10 attempts due to timeout for action: {action}")
    return "Timeout after 10 attempts", 0, False, {"error": "API Timeout"}


# --- Prompt Loading ---
PROMPT_FILE_PATH = './prompts/fever.json'
WEBTHINK_PROMPT_TEMPLATE = ""

try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    prompt_path = os.path.join(script_dir, 'prompts', 'fever.json')
    with open(prompt_path, 'r') as f:
        prompt_dict = json.load(f)
    WEBTHINK_PROMPT_TEMPLATE = prompt_dict['webthink_simple3']
except FileNotFoundError:
    print(f"[ERROR] Prompt file not found at {PROMPT_FILE_PATH}")
    print("Please ensure 'fever.json' is in the 'prompts/' directory.")
    WEBTHINK_PROMPT_TEMPLATE = """Claim: {claim}
Thought 1: I need to assess the claim.
Action 1: Search[entity related to claim]
Observation 1: [Search results]
Thought 2: [Reasoning based on observation]
Action 2: Finish[SUPPORTS/REFUTES/NOT ENOUGH INFO]
"""
except KeyError:
    print(f"[ERROR] 'webthink_simple3' key not found in {PROMPT_FILE_PATH}")
    WEBTHINK_PROMPT_TEMPLATE = "Claim: {claim}\n"


# --- Answer Extraction ---
def extract_final_answer_from_trace_string(trace_trajectory_string):
    """
    Extract the final answer from a trajectory string.
    
    Args:
        trace_trajectory_string: Full trajectory text
        
    Returns:
        Answer string (SUPPORTS/REFUTES/NOT ENOUGH INFO) or None
    """
    pattern = re.compile(r"^Action \d+: Finish\[(SUPPORTS|REFUTES|NOT ENOUGH INFO)\]\s*$", re.MULTILINE)
    matches = pattern.findall(trace_trajectory_string)
    if matches:
        return matches[-1].strip()
    return None


def extract_answers_from_traces(all_traces_info):
    """
    Extract answers from a list of trace info dictionaries.
    
    Args:
        all_traces_info: List of trace info dicts
        
    Returns:
        List of answer strings (filtering out None values)
    """
    extracted_answers = []
    if not isinstance(all_traces_info, list):
        print(f"[WARNING] extract_answers_from_traces expected a list, got {type(all_traces_info)}")
        return extracted_answers

    for i, trace_info in enumerate(all_traces_info):
        trajectory = trace_info.get('traj', '')
        answer_from_traj = extract_final_answer_from_trace_string(trajectory)

        if answer_from_traj is not None:
            extracted_answers.append(answer_from_traj)
        else:
            env_answer = trace_info.get('answer')
            if env_answer in ['SUPPORTS', 'REFUTES', 'NOT ENOUGH INFO']:
                extracted_answers.append(env_answer)
            else:
                extracted_answers.append(None)

    return [ans for ans in extracted_answers if ans is not None]


def extract_trajectories_from_traces(all_traces_info):
    """
    Extract the full trajectory string from each trace.
    
    Args:
        all_traces_info: List of trace info dicts
        
    Returns:
        List of trajectory strings
    """
    extracted_trajectories = []
    if not isinstance(all_traces_info, list):
        print(f"[WARNING] extract_trajectories_from_traces expected a list, got {type(all_traces_info)}")
        return extracted_trajectories

    for i, trace_info in enumerate(all_traces_info):
        trajectory = trace_info.get('traj', '')
        if trajectory:
            extracted_trajectories.append(trajectory)
        else:
            print(f"[WARNING] Missing trajectory in trace_info for trace {i}")
            extracted_trajectories.append(f"Trace {i+1} was empty or missing.")

    return extracted_trajectories


# --- Answer Synthesis ---
def synthesize_answer_with_llm(list_of_trajectories, claim_for_context=""):
    """
    Synthesize a final verdict using an LLM to review multiple trajectories.
    
    Args:
        list_of_trajectories: List of trajectory strings
        claim_for_context: The original claim being verified
        
    Returns:
        Synthesized answer (SUPPORTS/REFUTES/NOT ENOUGH INFO)
    """
    if not list_of_trajectories:
        return "NOT ENOUGH INFO"

    valid_trajectories = [str(t).strip() for t in list_of_trajectories if str(t).strip()]
    if not valid_trajectories:
        return "NOT ENOUGH INFO"

    prompt_template = """You are an answer aggregation assistant. Your task is to review the reasoning trajectories below, each of which ends with a verdict about the claim.

Your goal is to find the best overall answer by considering the conclusions reached in the trajectories. If most trajectories agree, that is a strong signal, but also consider the quality and justification of the reasoning in each trajectory. If a minority trajectory provides much stronger evidence or reasoning than the majority, you may select its answer instead. Clearly prefer the majority answer unless there is a compelling, well-justified reason to do otherwise.

Claim:
"{claim_context}"

Reasoning Trajectories:
{formatted_trajectories}

Based on your review, what is the best overall answer? Choose one of: SUPPORTS, REFUTES, or NOT ENOUGH INFO.

Final Answer (SUPPORTS / REFUTES / NOT ENOUGH INFO):"""

    formatted_trajectories = ""
    for i, traj in enumerate(valid_trajectories):
        formatted_trajectories += f"--- Trajectory {i+1} ---\n{traj}\n--- End of Trajectory {i+1} ---\n\n"
    formatted_trajectories = formatted_trajectories.strip()

    synthesizer_prompt = prompt_template.format(
        claim_context=claim_for_context,
        formatted_trajectories=formatted_trajectories
    )

    # LLM call to get the synthesized verdict - allow full response without stopping
    llm_response = llm(synthesizer_prompt, stop=[], num_traces=1)
    
    # Extract the answer using pattern matching
    # Look for SUPPORTS, REFUTES, or NOT ENOUGH INFO in the response
    # Prioritize uppercase versions and check for common patterns
    
    # First, try to find the answer in common formats
    patterns_to_try = [
        r'\b(SUPPORTS|REFUTES|NOT ENOUGH INFO)\b',  # Uppercase standalone
        r'\*\*(SUPPORTS|REFUTES|NOT ENOUGH INFO)\*\*',  # Markdown bold
        r'Final Answer:?\s*(SUPPORTS|REFUTES|NOT ENOUGH INFO)',  # After "Final Answer"
        r'Answer:?\s*(SUPPORTS|REFUTES|NOT ENOUGH INFO)',  # After "Answer"
        r'\b(supports|refutes|not enough info)\b',  # Lowercase versions
        r'Final Answer:?\s*(SUP|REF|NOT)',  # Truncated versions
    ]
    
    for pattern in patterns_to_try:
        matches = re.findall(pattern, llm_response, re.IGNORECASE)
        if matches:
            # Return the last match found, converted to uppercase
            match_str = matches[-1].upper()
            
            # Handle truncated/partial matches
            if match_str.startswith("SUP"):
                return "SUPPORTS"
            elif match_str.startswith("REF"):
                return "REFUTES"
            elif match_str.startswith("NOT"):
                return "NOT ENOUGH INFO"
                
            if match_str in ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]:
                return match_str
    
    # If no match found, log the response and default to NOT ENOUGH INFO
    print(f"[WARNING] LLM returned unexpected verdict: '{llm_response}'. Defaulting to NOT ENOUGH INFO.")
    return "NOT ENOUGH INFO"



# --- Core Single Trace Execution ---
def run_single_trace(idx, initial_prompt_template, to_print=True, temperature=None):
    """
    Execute a single ReAct trace for a FEVER claim.
    
    Args:
        idx: Question index
        initial_prompt_template: The prompt template to use
        to_print: Whether to print progress
        temperature: Override temperature (if None, uses default based on single trace)
        
    Returns:
        Dictionary with trace information including:
        - question_idx, question_text, answer, gt_answer
        - em, f1, reward
        - n_calls, n_badcalls
        - traj (full trajectory string)
    """
    fever_env = get_fever_env()
    
    question = fever_env.reset(idx=idx)
    current_prompt = initial_prompt_template + question + "\n"
    
    if to_print:
        print(f"[TRACE] Index: {idx}")
        print(f"[CLAIM] {question}")
    
    n_calls, n_badcalls = 0, 0
    current_trace_steps = []
    
    # Determine num_traces parameter for LLM (affects temperature)
    num_traces_param = 1 if temperature is None or temperature == 0.0 else 3
    
    for i in range(1, 8):  # Max 7 steps per trace
        n_calls += 1
        thought_action = llm(current_prompt + f"Thought {i}:", stop=[f"\nObservation {i}:"], num_traces=num_traces_param)
        
        try:
            thought, action = thought_action.strip().split(f"\nAction {i}: ")
        except:
            if to_print:
                print(f"[ERROR] Parsing thought/action: '{thought_action}'")
            n_badcalls += 1
            thought = thought_action.strip().split('\n')[0] if thought_action else "Error in thought generation"
            action_prompt = current_prompt + f"Thought {i}: {thought}\nAction {i}:"
            action = llm(action_prompt, stop=["\n"], num_traces=num_traces_param).strip()
            if not action or ("Finish[" not in action and "Search[" not in action and "Lookup[" not in action):
                action = "Finish[NOT ENOUGH INFO]"
                if to_print:
                    print(f"[RECOVERY] Using default action: {action}")
        
        # Ensure action is a string
        if not isinstance(action, str):
            action = str(action)
        
        # Lowercase first character of action (e.g., Search -> search)
        # This is required by the WikiEnv - see FEVER.ipynb line 95
        action_lowercase = action[0].lower() + action[1:] if action else action
        obs, r, done, info = step(fever_env, action_lowercase)
        obs = obs.replace('\\n', '') if isinstance(obs, str) else str(obs)
        
        step_str = f"Thought {i}: {thought}\nAction {i}: {action}\nObservation {i}: {obs}\n"
        current_prompt += step_str
        current_trace_steps.append(step_str)
        
        if to_print:
            print(step_str)
        
        if done:
            break
    
    if not isinstance(info, dict):
        info = {}
    
    if not done:
        if to_print:
            print(f"[WARNING] Agent did not finish in {i} steps. Forcing Finish[NOT ENOUGH INFO].")
        obs_finish, r_finish, done_finish, info_finish = step(fever_env, "finish[NOT ENOUGH INFO]")
        info.update(info_finish)
        if 'answer' not in info or not info['answer']:
            info['answer'] = 'NOT ENOUGH INFO'
        forced_step_str = f"Thought {i+1}: Agent did not finish. Forcing.\nAction {i+1}: Finish[NOT ENOUGH INFO]\nObservation {i+1}: {obs_finish}\n"
        current_trace_steps.append(forced_step_str)
    
    trace_info = info.copy()
    trace_info.update({
        'n_calls': n_calls,
        'n_badcalls': n_badcalls,
        'traj': initial_prompt_template + question + "\n" + "".join(current_trace_steps),
        'question_idx': idx,
        'question_text': question,
        'answer': info.get('answer', 'NOT ENOUGH INFO' if not done else '[ERROR_NO_ANSWER]')
    })
    
    if to_print:
        print(f"[RESULT] Answer: {trace_info['answer']} | GT: {trace_info.get('gt_answer', 'UNKNOWN')} | EM: {trace_info.get('em', 0.0)}\n")
    
    return trace_info


# --- Utility for JSON appending ---
def append_to_json(data, filename):
    """Append data to a JSON file that stores a list of JSON objects."""
    if os.path.exists(filename):
        with open(filename, 'r+') as f:
            try:
                file_data = json.load(f)
            except json.JSONDecodeError:
                file_data = []
            
            if isinstance(file_data, list):
                file_data.append(data)
            else:
                file_data = [data]
            
            f.seek(0)
            json.dump(file_data, f, indent=4)
            f.truncate()
    else:
        with open(filename, 'w') as f:
            json.dump([data], f, indent=4)


def get_next_run_number(framework_folder):
    """
    Get the next run number for a framework folder.
    
    Looks for existing run_XXX folders and returns the next number.
    
    Args:
        framework_folder: Path to the framework folder (e.g., results/fever/reflexion)
        
    Returns:
        String like 'run_001', 'run_002', etc.
    """
    import re
    
    if not os.path.exists(framework_folder):
        return 'run_001'
    
    existing_runs = []
    for item in os.listdir(framework_folder):
        match = re.match(r'run_(\d+)', item)
        if match:
            existing_runs.append(int(match.group(1)))
    
    if not existing_runs:
        return 'run_001'
    
    next_num = max(existing_runs) + 1
    return f'run_{next_num:03d}'
