"""
Shared utilities for SciFACT agents.

This module contains common functions used by all SciFACT agent frameworks:
- LLM interaction (Gemini API with rate limiting)
- Environment setup and interaction
- Prompt loading
- Answer extraction and synthesis
"""

import os
import time
import re
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# --- LLM Configuration and Interaction ---
from google import genai
from google.genai import types

# The client gets the API key from the environment variable `GEMINI_API_KEY`.
client = genai.Client()

# Model configuration
LLM_MODEL = "gemini-2.5-flash"
LLM_DELAY_SECONDS = 0.5  # Rate limiting delay


def llm(prompt: str, stop: List[str] = ["\n"], temperature: float = None, num_traces: int = 1) -> str:
    """
    Call the language model with the given prompt.
    
    Args:
        prompt: The input prompt string
        stop: List of stop sequences
        temperature: Override temperature (if None, uses default based on num_traces)
        num_traces: Number of traces (affects default temperature setting)
        
    Returns:
        String response from the LLM
    """
    # Set temperature: 0.0 for single trace (deterministic), 0.7 for multi-trace
    if temperature is None:
        temperature = 0.0 if num_traces == 1 else 0.7
    
    time.sleep(LLM_DELAY_SECONDS)
    
    try:
        response = client.models.generate_content(
            model=LLM_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=temperature,
                stop_sequences=stop,
                max_output_tokens=256
            )
        )
        return response.text.strip()
    except Exception as e:
        print(f"[LLM ERROR] {e}")
        return ""


# --- Environment Setup ---
env = None


def get_scifact_env():
    """Get or initialize the SciFACT environment."""
    global env
    if env is None:
        from scifact_env import SciFACTEnv
        env = SciFACTEnv()
    return env


def step(current_env, action: str) -> Tuple[str, float, bool, Dict]:
    """
    Execute a step in the environment.
    
    Args:
        current_env: The SciFACT environment
        action: Action string to execute
        
    Returns:
        Tuple of (observation, reward, done, info)
    """
    # Normalize action case for SciFACT (lowercase action name)
    action = action.strip()
    if action.startswith("Search"):
        action = "s" + action[1:]
    elif action.startswith("Lookup"):
        action = "l" + action[1:]
    elif action.startswith("Finish"):
        action = "f" + action[1:]
    
    return current_env.step(action)


# --- Prompt Loading ---
PROMPT_FILE_PATH = './prompts/scifact.json'
WEBTHINK_PROMPT_TEMPLATE = ""

try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    prompt_path = os.path.join(script_dir, 'prompts', 'scifact.json')
    
    with open(prompt_path, 'r', encoding='utf-8') as f:
        prompts = json.load(f)
        WEBTHINK_PROMPT_TEMPLATE = prompts.get('webthink_simple3', '')
except FileNotFoundError:
    print(f"[ERROR] Prompt file not found: {PROMPT_FILE_PATH}")
    WEBTHINK_PROMPT_TEMPLATE = """
Determine if the given Claim is SUPPORT, REFUTE, or NOT ENOUGH INFO.

Claim: {claim}
"""
except KeyError:
    print(f"[ERROR] 'webthink_simple3' key not found in {PROMPT_FILE_PATH}")
    WEBTHINK_PROMPT_TEMPLATE = "Claim: {claim}\n"


# --- Answer Extraction ---

def extract_final_answer_from_trace_string(trace_trajectory_string: str) -> Optional[str]:
    """
    Extract the final answer from a trajectory string.
    
    Args:
        trace_trajectory_string: Full trajectory text
        
    Returns:
        Answer string (SUPPORT/REFUTE/NOT ENOUGH INFO) or None
    """
    match = re.search(r'[Ff]inish\[([^\]]+)\]', trace_trajectory_string)
    if match:
        answer = match.group(1).strip().upper()
        # Normalize
        if answer in ["SUPPORT", "SUPPORTS"]:
            return "SUPPORT"
        elif answer in ["REFUTE", "REFUTES"]:
            return "REFUTE"
        elif answer in ["NOT ENOUGH INFO", "NEI", "NOINFO"]:
            return "NOT ENOUGH INFO"
        return answer
    return None


def extract_answers_from_traces(all_traces_info: List[Dict]) -> List[str]:
    """
    Extract answers from a list of trace info dictionaries.
    
    Args:
        all_traces_info: List of trace info dicts
        
    Returns:
        List of answer strings (filtering out None values)
    """
    answers = []
    for trace_info in all_traces_info:
        traj = trace_info.get('traj', '')
        answer = extract_final_answer_from_trace_string(traj)
        if answer:
            answers.append(answer)
    return answers


def extract_trajectories_from_traces(all_traces_info: List[Dict]) -> List[str]:
    """
    Extract the full trajectory string from each trace.
    
    Args:
        all_traces_info: List of trace info dicts
        
    Returns:
        List of trajectory strings
    """
    trajectories = []
    for trace_info in all_traces_info:
        traj = trace_info.get('traj', '')
        if traj:
            trajectories.append(traj)
    return trajectories


# --- Answer Synthesis ---

def synthesize_answer_with_llm(list_of_trajectories: List[str], claim_for_context: str = "") -> str:
    """
    Synthesize a final verdict using an LLM to review multiple trajectories.
    
    Args:
        list_of_trajectories: List of trajectory strings
        claim_for_context: The original claim being verified
        
    Returns:
        Synthesized answer (SUPPORT/REFUTE/NOT ENOUGH INFO)
    """
    # Load synthesis prompt
    script_dir = os.path.dirname(os.path.abspath(__file__))
    prompt_path = os.path.join(script_dir, 'prompts', 'scifact.json')
    
    try:
        with open(prompt_path, 'r', encoding='utf-8') as f:
            prompts = json.load(f)
            synthesis_template = prompts.get('synthesis_prompt', '')
    except:
        synthesis_template = "Based on the traces, what is the verdict? Answer: "
    
    # Build prompt
    traces_text = ""
    for i, traj in enumerate(list_of_trajectories[:3]):  # Max 3 traces
        traces_text += f"\n\nTrace {i+1}:\n{traj}"
    
    prompt = synthesis_template.format(
        claim=claim_for_context,
        trace_1=list_of_trajectories[0] if len(list_of_trajectories) > 0 else "",
        trace_2=list_of_trajectories[1] if len(list_of_trajectories) > 1 else "",
        trace_3=list_of_trajectories[2] if len(list_of_trajectories) > 2 else ""
    )
    
    response = llm(prompt, stop=["\n"], temperature=0.0)
    
    # Extract answer
    answer = response.strip().upper()
    if "SUPPORT" in answer:
        return "SUPPORT"
    elif "REFUTE" in answer:
        return "REFUTE"
    else:
        return "NOT ENOUGH INFO"


# --- Core Single Trace Execution ---

def run_single_trace(
    idx: int,
    initial_prompt_template: str,
    to_print: bool = True,
    temperature: float = None
) -> Dict[str, Any]:
    """
    Execute a single ReAct trace for a SciFACT claim.
    Uses FEVER-style robust parsing: combined thought+action call with fallback.
    
    Args:
        idx: Claim index
        initial_prompt_template: The prompt template to use
        to_print: Whether to print progress
        temperature: Temperature for LLM sampling
        
    Returns:
        Dictionary with trace info including:
        - question_idx, question_text, answer, gt_answer
        - em, f1, reward
        - n_calls, n_badcalls
        - traj (full trajectory string)
    """
    current_env = get_scifact_env()
    claim = current_env.reset(idx=idx)
    
    if to_print:
        print(f"[CLAIM] {claim}")
    
    # Build initial prompt - append claim with "Claim:" prefix
    current_prompt = initial_prompt_template + f"Claim: {claim}\n"
    
    n_calls = 0
    n_badcalls = 0
    done = False
    max_steps = 7
    current_trace_steps = []
    
    # Determine num_traces parameter for LLM (affects temperature)
    num_traces_param = 1 if temperature is None or temperature == 0.0 else 3
    
    for i in range(1, max_steps + 1):
        n_calls += 1
        
        # FEVER-style: Get thought and action together, stop at Observation
        thought_action = llm(
            current_prompt + f"Thought {i}:", 
            stop=[f"\nObservation {i}:"], 
            temperature=temperature
        )
        
        # Try to parse thought and action from combined response
        try:
            thought, action = thought_action.strip().split(f"\nAction {i}: ")
        except ValueError:
            # Parsing failed - try to extract action from anywhere in the response
            if to_print:
                print(f"[ERROR] Parsing thought/action: '{thought_action[:100]}...'")
            n_badcalls += 1
            
            # First, try to find an action (Search/Lookup/Finish) anywhere in the response
            action_match = re.search(r'(Search\[[^\]]+\]|Lookup\[[^\]]+\]|Finish\[[^\]]+\])', thought_action)
            
            if action_match:
                # Found an action in the response!
                action = action_match.group(1)
                # Extract thought as everything before the action
                thought = thought_action[:action_match.start()].strip()
                # Remove "Action X:" prefix from thought if present
                thought = re.sub(r'Action \d+:\s*$', '', thought).strip()
                if not thought:
                    thought = "I need to investigate this claim."
                if to_print:
                    print(f"[RECOVERY] Extracted action: {action}")
            else:
                # No valid action found - use default
                thought = thought_action.strip().split('\n')[0] if thought_action else "I need more information."
                action = "Finish[NOT ENOUGH INFO]"
                if to_print:
                    print(f"[RECOVERY] Using default action: {action}")
        
        # Ensure action is a string
        if not isinstance(action, str):
            action = str(action)
        
        # Clean action - remove any prefix like "Action X:" if LLM repeated it
        action_clean_match = re.search(r'(Search\[[^\]]+\]|Lookup\[[^\]]+\]|Finish\[[^\]]+\])', action)
        if action_clean_match:
            action = action_clean_match.group(1)
        
        # Lowercase first character of action for environment
        action_lowercase = action[0].lower() + action[1:] if action else action
        obs, r, done, info = step(current_env, action_lowercase)
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
    
    # If not done, force finish
    if not done:
        if to_print:
            print(f"[WARNING] Agent did not finish in {i} steps. Forcing Finish[NOT ENOUGH INFO].")
        obs_finish, r_finish, done_finish, info_finish = step(current_env, "finish[NOT ENOUGH INFO]")
        info.update(info_finish)
        if 'answer' not in info or not info['answer']:
            info['answer'] = 'NOT ENOUGH INFO'
        forced_step_str = f"Thought {i+1}: Agent did not finish. Forcing.\nAction {i+1}: Finish[NOT ENOUGH INFO]\nObservation {i+1}: {obs_finish}\n"
        current_trace_steps.append(forced_step_str)
    
    # Build result
    result = {
        'question_idx': idx,
        'question_text': claim,
        'answer': info.get('answer', 'NOT ENOUGH INFO'),
        'gt_answer': info.get('gt_answer', 'UNKNOWN'),
        'em': info.get('em', 0.0),
        'f1': info.get('f1', 0.0),
        'reward': r if done else 0.0,
        'n_calls': n_calls,
        'n_badcalls': n_badcalls,
        'traj': initial_prompt_template + f"Claim: {claim}\n" + "".join(current_trace_steps)
    }
    
    return result


# --- Utility for JSON appending ---

def append_to_json(data: Dict, filename: str):
    """Append data to a JSON file that stores a list of JSON objects."""
    filepath = Path(filename)
    
    if filepath.exists():
        with open(filepath, 'r', encoding='utf-8') as f:
            existing = json.load(f)
    else:
        existing = []
    
    existing.append(data)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)
