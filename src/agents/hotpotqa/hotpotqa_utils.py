"""
Shared utilities for HotPotQA agents.

This module contains common functions used by all HotPotQA agent frameworks:
- LLM interaction (3s delay, gemini-2.5-flash)
- Environment setup and interaction (HotPotQAWrapper)
- Prompt loading
- Answer extraction and synthesis
- LLM-as-judge evaluation
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


def llm(prompt, stop=["\n"], temperature=None, num_traces=1):
    """
    Call the language model with the given prompt.
    
    Args:
        prompt: The input prompt string
        stop: List of stop sequences
        temperature: Override temperature (if None, uses default based on num_traces)
        num_traces: Number of traces (affects default temperature setting)
        
    Returns:
        Tuple of (text_response, token_usage_dict) where token_usage_dict has:
        - input_tokens: Number of prompt tokens
        - output_tokens: Number of completion tokens
        - total_tokens: Sum of input and output tokens
    """
    # 3 second delay for paid tier rate limits
    time.sleep(3.0)

    if temperature is None:
        temperature_setting = 0.0 if num_traces == 1 else 0.7
    else:
        temperature_setting = temperature
    
    # Default token usage for error cases
    default_token_usage = {'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0}
        
    max_retries = 3
    for attempt in range(max_retries):
        try:
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
            if response and response.text:
                # Extract token usage from response metadata
                token_usage = default_token_usage.copy()
                if hasattr(response, 'usage_metadata') and response.usage_metadata:
                    token_usage = {
                        'input_tokens': getattr(response.usage_metadata, 'prompt_token_count', 0) or 0,
                        'output_tokens': getattr(response.usage_metadata, 'candidates_token_count', 0) or 0,
                        'total_tokens': getattr(response.usage_metadata, 'total_token_count', 0) or 0
                    }
                return response.text, token_usage
            time.sleep(2)  # Wait before retry if we got an empty response
        except Exception as e:
            print(f"LLM call failed (attempt {attempt + 1}/{max_retries}): {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                raise
    
    return "I need to finish now.\nFinish[Unable to proceed due to API error]", default_token_usage


# --- Environment Setup ---
env = None


def get_hotpotqa_env():
    """Get or initialize the HotPotQA environment."""
    global env
    if env is None:
        env = wikienv.WikiEnv()
        env = wrappers.HotPotQAWrapper(env, split="dev")
        env = wrappers.LoggingWrapper(env)
    return env


def step(current_env, action):
    """
    Execute a step in the environment with retry logic for timeouts.
    
    Args:
        current_env: The HotPotQA environment
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
            time.sleep(2)
    
    print(f"[ERROR] Failed to execute step after 10 attempts due to timeout for action: {action}")
    return "Timeout after 10 attempts", 0, False, {"error": "API Timeout"}


# --- Prompt Loading ---
PROMPT_FILE_PATH = './prompts_naive.json'
WEBTHINK_PROMPT_TEMPLATE = ""

try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    prompt_path = os.path.join(script_dir, 'prompts_naive.json')
    with open(prompt_path, 'r') as f:
        prompt_dict = json.load(f)
    webthink_examples = prompt_dict['webthink_simple6']
    instruction = """Solve a question answering task with interleaving Thought, Action, Observation steps. Thought can reason about the current situation, and Action can be three types: 
(1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.
(2) Lookup[keyword], which returns the next sentence containing keyword in the current passage.
(3) Finish[answer], which returns the answer and finishes the task.
Here are some examples.
If the question is based on a false premise or the information needed to answer is not available, answer "null"."""
    WEBTHINK_PROMPT_TEMPLATE = instruction + webthink_examples
except FileNotFoundError:
    print(f"[ERROR] Prompt file not found at {PROMPT_FILE_PATH}")
    WEBTHINK_PROMPT_TEMPLATE = "Question: "
except KeyError:
    print(f"[ERROR] 'webthink_simple6' key not found in {PROMPT_FILE_PATH}")
    WEBTHINK_PROMPT_TEMPLATE = "Question: "


# --- Answer Extraction ---
def extract_final_answer_from_trace_string(trace_trajectory_string):
    """
    Extract the final answer from a trajectory string.
    
    Args:
        trace_trajectory_string: Full trajectory text
        
    Returns:
        Answer string or None
    """
    pattern = re.compile(r"^Action \d+: Finish\[(.*?)\]\s*$", re.MULTILINE)
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
            if env_answer:
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
def synthesize_answer_with_llm(list_of_trajectories, question_for_context=""):
    """
    Synthesize a final answer using an LLM to review multiple trajectories.
    
    Args:
        list_of_trajectories: List of trajectory strings
        question_for_context: The original question being answered
        
    Returns:
        Tuple of (synthesized_answer, token_usage)
    """
    default_token_usage = {'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0}
    
    if not list_of_trajectories:
        return "null", default_token_usage

    valid_trajectories = [str(t).strip() for t in list_of_trajectories if str(t).strip()]
    if not valid_trajectories:
        return "null", default_token_usage

    prompt_template = """You are an expert analyst. Your task is to determine the single best answer to the question, based on the reasoning trajectories provided below.

Each trajectory represents a separate attempt to answer the same question, including the reasoning steps and final answer.

Carefully review all trajectories and evaluate the logical soundness, factual accuracy, relevance to the question, and completeness of each. Then, identify the answer that is best supported by reasoning and evidence.

Question: {question_context}

Reasoning Trajectories:
{formatted_trajectories}

Based on your analysis of all the reasoning trajectories, what is the single best answer to the question? Output ONLY the final answer, nothing else.
Final Answer:"""

    formatted_trajectories = ""
    for i, traj in enumerate(valid_trajectories):
        formatted_trajectories += f"--- Trajectory {i+1} ---\n{traj}\n--- End of Trajectory {i+1} ---\n\n"
    formatted_trajectories = formatted_trajectories.strip()

    synthesizer_prompt = prompt_template.format(
        question_context=question_for_context,
        formatted_trajectories=formatted_trajectories
    )

    llm_response, token_usage = llm(synthesizer_prompt, stop=["\n"], num_traces=1)
    return llm_response.strip(), token_usage


# --- Majority Voting (Semantic) ---
def majority_vote_semantic(answers, question):
    """
    Use LLM to find majority answer considering semantic equivalence.
    
    Args:
        answers: List of answer strings (typically 3)
        question: The original question for context
        
    Returns:
        Tuple of (majority_answer, token_usage)
    """
    default_token_usage = {'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0}
    
    if not answers:
        return "null", default_token_usage
    
    if len(answers) == 1:
        return answers[0], default_token_usage
    
    # Build numbered list of answers
    answer_list = "\n".join(f"{i+1}. {ans}" for i, ans in enumerate(answers))
    
    prompt = f"""Question: {question}

Three reasoning attempts produced these answers:
{answer_list}

Task: Identify which answers mean the same thing (semantic equivalence), then select the most common answer.

Consider that different phrasings can mean the same thing (e.g., "NYC" = "New York City" = "New York").

If there's no clear majority (all different), pick the most reasonable answer based on the question.

Output ONLY the majority answer, nothing else.
Majority Answer:"""
    
    response, token_usage = llm(prompt, stop=["\n"], num_traces=1)
    return response.strip(), token_usage


# --- LLM-as-Judge Evaluation ---
def llm_judge_answer(question, prediction, ground_truth):
    """
    Use LLM as a judge to evaluate semantic equivalence of answer.
    
    Args:
        question: The original question
        prediction: The agent's predicted answer
        ground_truth: The correct answer
        
    Returns:
        Dictionary with 'llm_correct' (bool), 'llm_explanation' (str), and token_usage fields
    """
    prompt = f"""You are an expert evaluator for a Question Answering system.

Question: {question}
Ground Truth Answer: {ground_truth}
Predicted Answer: {prediction}

Task: Determine if the "Predicted Answer" is semantically equivalent to the "Ground Truth Answer".

Guidelines:
- Ignore minor formatting differences (punctuation, capitalization)
- Allow for synonyms or different entity aliases (e.g., "JFK" = "John F. Kennedy")
- If the prediction adds extra correct context, it is ACCEPTABLE
- If the prediction is vague or missing key information, it is INCORRECT

Output:
Explanation: [Brief reasoning]
Label: [CORRECT or INCORRECT]"""
    
    response, token_usage = llm(prompt, stop=[], num_traces=1)
    
    # Parse response
    is_correct = "CORRECT" in response.upper() and "INCORRECT" not in response.upper()
    
    # Extract explanation
    explanation = response.strip()
    if "Explanation:" in response:
        parts = response.split("Label:")
        explanation = parts[0].replace("Explanation:", "").strip() if parts else response.strip()
    
    return {
        'llm_correct': is_correct,
        'llm_explanation': explanation,
        'judge_input_tokens': token_usage['input_tokens'],
        'judge_output_tokens': token_usage['output_tokens'],
        'judge_total_tokens': token_usage['total_tokens']
    }


# --- Core Single Trace Execution ---
def run_single_trace(idx, initial_prompt_template, to_print=True, temperature=None):
    """
    Execute a single ReAct trace for a HotPotQA question.
    
    Args:
        idx: Question index
        initial_prompt_template: The prompt template to use
        to_print: Whether to print progress
        temperature: Override temperature (if None, uses default)
        
    Returns:
        Dictionary with trace information including:
        - question_idx, question_text, answer, gt_answer
        - em, f1, reward
        - n_calls, n_badcalls
        - input_tokens, output_tokens, total_tokens (NEW)
        - traj (full trajectory string)
        - llm_correct, llm_explanation (LLM-judge results)
    """
    hotpotqa_env = get_hotpotqa_env()
    
    question = hotpotqa_env.reset(idx=idx)
    current_prompt = initial_prompt_template + question + "\n"
    
    if to_print:
        print(f"[TRACE] Index: {idx}")
        print(f"[QUESTION] {question}")
    
    n_calls, n_badcalls = 0, 0
    total_input_tokens, total_output_tokens = 0, 0  # Token accumulators
    llm_calls = []  # Per-call token logging
    current_trace_steps = []
    
    # Determine temperature for diversity
    num_traces_param = 1 if temperature is None or temperature == 0.0 else 3
    
    for i in range(1, 8):  # Max 7 steps per trace
        n_calls += 1
        thought_action, token_usage = llm(current_prompt + f"Thought {i}:", stop=[f"\nObservation {i}:"], num_traces=num_traces_param)
        total_input_tokens += token_usage['input_tokens']
        total_output_tokens += token_usage['output_tokens']
        llm_calls.append({
            'call_num': n_calls,
            'step': i,
            'type': 'thought_action',
            'input_tokens': token_usage['input_tokens'],
            'output_tokens': token_usage['output_tokens'],
            'total_tokens': token_usage['total_tokens']
        })
        
        try:
            thought, action = thought_action.strip().split(f"\nAction {i}: ")
        except:
            if to_print:
                print(f"[ERROR] Parsing thought/action: '{thought_action}'")
            n_badcalls += 1
            thought = thought_action.strip().split('\n')[0] if thought_action else "Error in thought generation"
            action_prompt = current_prompt + f"Thought {i}: {thought}\nAction {i}:"
            action, recovery_token_usage = llm(action_prompt, stop=["\n"], num_traces=num_traces_param)
            action = action.strip()
            total_input_tokens += recovery_token_usage['input_tokens']
            total_output_tokens += recovery_token_usage['output_tokens']
            n_calls += 1
            llm_calls.append({
                'call_num': n_calls,
                'step': i,
                'type': 'action_recovery',
                'input_tokens': recovery_token_usage['input_tokens'],
                'output_tokens': recovery_token_usage['output_tokens'],
                'total_tokens': recovery_token_usage['total_tokens']
            })
            if not action or ("Finish[" not in action and "Search[" not in action and "Lookup[" not in action):
                action = "Finish[null]"
                if to_print:
                    print(f"[RECOVERY] Using default action: {action}")
        
        # Ensure action is a string
        if not isinstance(action, str):
            action = str(action)
        
        # Lowercase first character of action (required by WikiEnv)
        action_lowercase = action[0].lower() + action[1:] if action else action
        obs, r, done, info = step(hotpotqa_env, action_lowercase)
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
            print(f"[WARNING] Agent did not finish in {i} steps. Forcing Finish[null].")
        obs_finish, r_finish, done_finish, info_finish = step(hotpotqa_env, "finish[null]")
        info.update(info_finish)
        if 'answer' not in info or not info['answer']:
            info['answer'] = 'null'
        forced_step_str = f"Thought {i+1}: Agent did not finish. Forcing.\nAction {i+1}: Finish[null]\nObservation {i+1}: {obs_finish}\n"
        current_trace_steps.append(forced_step_str)
    
    # Get answer and ground truth
    answer = info.get('answer', 'null')
    gt_answer = info.get('gt_answer', '')
    question_text = question
    
    # LLM-as-judge evaluation
    llm_eval = llm_judge_answer(question_text, answer, gt_answer)
    # Add judge tokens to totals and call log
    total_input_tokens += llm_eval.get('judge_input_tokens', 0)
    total_output_tokens += llm_eval.get('judge_output_tokens', 0)
    llm_calls.append({
        'call_num': n_calls + 1,
        'step': 'judge',
        'type': 'llm_judge',
        'input_tokens': llm_eval.get('judge_input_tokens', 0),
        'output_tokens': llm_eval.get('judge_output_tokens', 0),
        'total_tokens': llm_eval.get('judge_total_tokens', 0)
    })
    
    trace_info = info.copy()
    trace_info.update({
        'n_calls': n_calls,
        'n_badcalls': n_badcalls,
        'input_tokens': total_input_tokens,
        'output_tokens': total_output_tokens,
        'total_tokens': total_input_tokens + total_output_tokens,
        'llm_calls': llm_calls,  # Per-call token breakdown
        'traj': initial_prompt_template + question + "\n" + "".join(current_trace_steps),
        'question_idx': idx,
        'question_text': question_text,
        'answer': answer,
        'llm_correct': llm_eval['llm_correct'],
        'llm_explanation': llm_eval['llm_explanation']
    })
    
    if to_print:
        print(f"[RESULT] Answer: {trace_info['answer']} | GT: {trace_info.get('gt_answer', 'UNKNOWN')} | EM: {trace_info.get('em', 0.0)} | LLM: {llm_eval['llm_correct']}\n")
    
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
        framework_folder: Path to the framework folder (e.g., results/hotpotqa/reflexion)
        
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
