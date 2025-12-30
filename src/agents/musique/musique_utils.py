"""
Shared utilities for Musique agents.

This module contains common functions used by Musique agent frameworks:
- LLM interaction
- Environment setup
- Single trace execution
- Evaluation metrics with decomposition analysis
"""

import os
import time
import re
import json
import sys
import collections
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Add shared directory to path for wikienv and wrappers
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../shared')))

# Import Musique Environment
# Assumes this file is in src/agents/musique/
from musique_env import MusiqueEnv

# --- LLM Configuration and Interaction ---
from google import genai
from google.genai import types

client = genai.Client()

def llm(prompt, stop=["\n"], num_traces=1):
    """
    Call the language model with the given prompt.
    """
    time.sleep(1.0) # Rate limit protection

    temperature_setting = 0.0 if num_traces == 1 else 0.7
    model_name = os.environ.get("MODEL_NAME", "gemini-2.5-flash")
    
    try:
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_budget=0),
                stop_sequences=stop,
                temperature=temperature_setting,
                max_output_tokens=512,
                top_p=1.0
            )
        )
        return response.text
    except Exception as e:
        print(f"[LLM ERROR] {e}")
        return ""

# --- Environment Setup ---
env = None

def get_musique_env():
    """Get or initialize the Musique environment."""
    global env
    if env is None:
        env = MusiqueEnv()
    return env

# --- Core Single Trace Execution ---
def run_single_trace(idx, initial_prompt_template, to_print=True, temperature=None):
    """
    Execute a single ReAct trace for a Musique question.
    """
    musique_env = get_musique_env()
    
    # Reset environment
    question_text = musique_env.reset(idx=idx)
    current_prompt = initial_prompt_template + question_text + "\n"
    
    if to_print:
        print(f"[TRACE] Index: {idx}")
        print(f"[QUESTION] {question_text}")
    
    n_calls = 0
    current_trace_steps = []
    
    num_traces_param = 1 if temperature is None or temperature == 0.0 else 3
    
    # Run ReAct loop (max 7 steps)
    for i in range(1, 8):
        n_calls += 1
        # Generate Thought + Action
        thought_action = llm(current_prompt + f"Thought {i}:", stop=[f"\nObservation {i}:"], num_traces=num_traces_param)
        
        try:
            parts = thought_action.strip().split(f"\nAction {i}: ")
            thought = parts[0]
            action = parts[1] if len(parts) > 1 else ""
        except:
             thought = thought_action
             action = ""
        
        # If action is empty or malformed, try to recover or force finish
        if not action or ("Finish[" not in action and "Search[" not in action and "Select[" not in action and "Lookup[" not in action):
             # Try one more time to generate just action
             action_prompt = current_prompt + f"Thought {i}: {thought}\nAction {i}:"
             action = llm(action_prompt, stop=["\n"], num_traces=num_traces_param).strip()
             
             if not action:
                 action = "Finish[NOT ENOUGH INFO]"

        # Execute step
        obs, r, done, info = musique_env.step(action)
        
        # Format step for prompt
        step_str = f"Thought {i}: {thought}\nAction {i}: {action}\nObservation {i}: {obs}\n"
        current_prompt += step_str
        current_trace_steps.append(step_str)
        
        if to_print:
            print(step_str)
            
        if done:
            break
            
    # Trace info
    info = musique_env._get_info()
    trace_info = info.copy()
    trace_info.update({
        'n_calls': n_calls,
        'traj': initial_prompt_template + question_text + "\n" + "".join(current_trace_steps),
        'em': r  # reward is EM
    })
    
    if to_print:
        print(f"[RESULT] Answer: {trace_info['answer']} | GT: {trace_info['gt_answer']} | EM: {trace_info['em']}")
        
    return trace_info

# --- Analysis Tools ---
def analyze_decomposition_performance(results):
    """
    Analyze results performance by hop count and composition type.
    
    Args:
        results: List of trace result dicts
        
    Returns:
        Summary string of performance breakdown
    """
    breakdown = collections.defaultdict(list)
    
    for res in results:
        # Determine hop count from decomposition length
        decomp = res.get('decomposition', [])
        hop_count = f"{len(decomp)}-hop" if decomp else "unknown"
        
        em = res.get('em', 0.0)
        breakdown[hop_count].append(em)
        
    summary = "Performance by Hop Count:\n"
    for hop, scores in sorted(breakdown.items()):
        avg_em = sum(scores) / len(scores) if scores else 0.0
        summary += f"  {hop}: EM = {avg_em:.3f} ({len(scores)} examples)\n"
        
    return summary

def append_to_json(data, filename):
    """Append data to a JSON file."""
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
