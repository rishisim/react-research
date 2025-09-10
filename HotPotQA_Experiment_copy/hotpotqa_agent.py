import os
import json
import time
import requests
from google import genai
from google.genai import types
import wikienv, wrappers

# --- Environment Setup ---
env = wikienv.WikiEnv()
env = wrappers.HotPotQAWrapper(env, split="dev")
env = wrappers.LoggingWrapper(env)

def step(env, action):
    attempts = 0
    while attempts < 10:
        try:
            return env.step(action)
        except requests.exceptions.Timeout:
            attempts += 1

# --- LLM and Helper Functions ---
client = genai.Client() # Assuming API key is in env
# client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"]) # for scripts

def llm(prompt, stop=["\n"], num_traces=1):
  # This delay handles the 15 RPM limit by waiting ~4 seconds per call.
  time.sleep(4.1)

  temperature_setting = 0.0 if num_traces == 1 else 0.7
  max_retries = 3
  for attempt in range(max_retries):
    try:
      response = client.models.generate_content(
        model="gemini-2.5-flash-lite-preview-06-17",
        contents=prompt,
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=0), # Disables thinking
            stop_sequences=stop,
            temperature=temperature_setting,
            max_output_tokens=100,
            top_p=1.0
        )
      )
      if response and response.text:
        return response.text
      time.sleep(2)  # Wait before retry if we got an empty response
    except Exception as e:
      print(f"LLM call failed (attempt {attempt + 1}/{max_retries}): {str(e)}")
      if attempt < max_retries - 1:
        time.sleep(2)  # Wait before retry
      else:
        raise  # Re-raise the last exception if we're out of retries
  
  return "I need to finish now.\nFinish[Unable to proceed due to API error]"  # Fallback response if all retries failed

import re

def extract_final_answer_from_trace_string(trace_trajectory_string):
    """
    Extracts the final answer from a ReAct trace trajectory string.
    Looks for the last occurrence of 'Action X: Finish[answer]'.
    """
    pattern = re.compile(r"^Action \d+: Finish\[(.*?)\]\s*$", re.MULTILINE)
    matches = pattern.findall(trace_trajectory_string)

    if matches:
        # The last match in the string is the one we want
        return matches[-1].strip()

    return None

def extract_answers_from_traces(all_traces_info):
    """
    Extracts the final answer from each trace in the all_traces_info list.
    """
    extracted_answers = []
    if not isinstance(all_traces_info, list):
        print(f"Warning: extract_answers_from_traces expected a list, got {type(all_traces_info)}")
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
    Extracts the full trajectory string from each trace in the all_traces_info list.
    """
    extracted_trajectories = []
    if not isinstance(all_traces_info, list):
        print(f"Warning: extract_trajectories_from_traces expected a list, got {type(all_traces_info)}")
        return extracted_trajectories

    for i, trace_info in enumerate(all_traces_info):
        trajectory = trace_info.get('traj', '')
        if trajectory:
            extracted_trajectories.append(trajectory)
        else:
            # Fallback or logging if a trajectory is empty/missing, though 'traj' should ideally always be there.
            print(f"Warning: Missing trajectory in trace_info for trace {i}")
            extracted_trajectories.append(f"Trace {i+1} was empty or missing.") # Placeholder for missing traj

    return extracted_trajectories


def synthesize_answer_with_llm(list_of_trajectories, question_for_context=""):
    """
    Synthesizes a single best answer from a list of full reasoning trajectories using an LLM.
    Includes the original question for better context.
    """
    if not list_of_trajectories:
        return "Error: No trajectories provided to synthesize."

    # Filter out any empty or placeholder trajectories if necessary, though ideally all should be valid.
    valid_trajectories = [str(t).strip() for t in list_of_trajectories if str(t).strip()]
    if not valid_trajectories:
        return "Error: No valid trajectories found after filtering to synthesize."

    # If only one trajectory, we might still want the LLM to extract the answer from it,
    # or we could attempt to parse its Finish[] action. For now, let LLM handle it.

    prompt_template = """
    
You are an expert analyst. Your task is to determine the single best answer to the question, based on the reasoning trajectories provided below.

Each trajectory represents a separate attempt to answer the same question, including the reasoning steps and final answer.

Carefully review all trajectories and evaluate the logical soundness, factual accuracy, relevance to the question, and completeness of each. Then, identify the answer that is best supported by reasoning and evidence.

Question Context:
{question_context}

Reasoning Trajectories:
{formatted_trajectories}

Based on your analysis of all the reasoning trajectories, what is the single best answer to the question? Simply output the final answer without any additional commentary or explanation.
Final Answer:"""

    question_context_str = ""
    if question_for_context:
        question_context_str = f"The question asked was: \"{question_for_context}\"\n\n"

    formatted_trajectories = ""
    for i, traj in enumerate(valid_trajectories):
        formatted_trajectories += f"--- Trajectory {i+1} ---\n{traj}\n--- End of Trajectory {i+1} ---\n\n"
    formatted_trajectories = formatted_trajectories.strip()

    synthesizer_prompt = prompt_template.format(
        question_context=question_context_str,
        formatted_trajectories=formatted_trajectories
    )
    # print(f"DEBUG: Synthesizer prompt for HotPotQA:\n{synthesizer_prompt}") # For debugging

    final_answer = llm(synthesizer_prompt, stop=["\n"], num_traces=1) # Stop at newline for cleaner answer
    return final_answer.strip()

def append_to_json(data_dict, json_file_path):
    if os.path.exists(json_file_path) and os.path.getsize(json_file_path) > 0:
        with open(json_file_path, 'r') as f:
            try:
                results_list = json.load(f)
            except json.JSONDecodeError:
                results_list = []
    else:
        results_list = []

    results_list.append(data_dict)

    with open(json_file_path, 'w') as f:
        json.dump(results_list, f, indent=4)

# --- Webthink Agent ---
# Load prompts
prompt_file_path = os.path.join(os.path.dirname(__file__), 'prompts_naive.json')

try:
    with open(prompt_file_path, 'r') as f:
        prompt_dict = json.load(f)
    webthink_examples = prompt_dict['webthink_simple6']
    instruction = """Solve a question answering task with interleaving Thought, Action, Observation steps. Thought can reason about the current situation, and Action can be three types: \n(1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.\n(2) Lookup[keyword], which returns the next sentence containing keyword in the current passage.\n(3) Finish[answer], which returns the answer and finishes the task.\nHere are some examples.\n If the question is based on a false premise or the information needed to answer is not available, answer "null"."""
    WEBTHINK_PROMPT_TEMPLATE = instruction + webthink_examples
except FileNotFoundError:
    print(f"ERROR: Prompt file {prompt_file_path} not found. Webthink might not work correctly.")
    WEBTHINK_PROMPT_TEMPLATE = "ERROR_PROMPT_FILE_NOT_FOUND" # Fallback
except KeyError:
    print(f"ERROR: Key 'webthink_simple6' not found in {prompt_file_path}. Webthink might not work correctly.")
    WEBTHINK_PROMPT_TEMPLATE = "ERROR_PROMPT_KEY_NOT_FOUND" # Fallback


def webthink(idx=None, initial_prompt_template=WEBTHINK_PROMPT_TEMPLATE, to_print=True, num_traces=1):
    all_traces_info = []
    question_for_synthesis = "" # Define outside loop to store it

    if num_traces <= 0:
        if to_print:
            print(f"Warning: webthink called with num_traces = {num_traces}. Must be > 0.")
        return "[INVALID_NUM_TRACES]", []

    for trace_num in range(num_traces):
        question = env.reset(idx=idx) # Reset environment for each trace
        if trace_num == 0: # Capture question on first trace for synthesizer
            question_for_synthesis = question

        current_prompt = initial_prompt_template + question + "\n"

        if to_print:
            print(f"--- Trace {trace_num + 1}/{num_traces} ---")
            print(idx, question)

        n_calls, n_badcalls = 0, 0

        for i in range(1, 8): # Max 7 steps per trace
            n_calls += 1
            try:
                thought_action = llm(current_prompt + f"Thought {i}:", stop=[f"\nObservation {i}:"], num_traces=1 if num_traces == 1 else 0.7)
                if not thought_action:
                    raise ValueError("Empty response from LLM")
                    
                # Try to split into thought and action
                try:
                    thought, action = thought_action.strip().split(f"\nAction {i}: ")
                except ValueError:
                    # If we can't split properly, try to salvage what we can
                    parts = thought_action.strip().split('\n')
                    thought = parts[0] if parts else "I need to finish now."
                    
                    # Make a separate call for the action
                    action = llm(current_prompt + f"Thought {i}: {thought}\nAction {i}:", 
                               stop=[f"\n"], 
                               num_traces=1 if num_traces == 1 else 0.7)
                    
                    if not action:
                        action = "Finish[Unable to determine next action]"
                    else:
                        action = action.strip()
                    
                    n_badcalls += 1
                    n_calls += 1  # Count the extra LLM call
            except Exception as e:
                print(f"Error in step {i}: {str(e)}")
                thought = "I need to finish now."
                action = "Finish[Error occurred while processing]"
                n_badcalls += 1

            obs, r, done, info = step(env, action[0].lower() + action[1:])
            obs = obs.replace('\\n', '')

            step_str = f"Thought {i}: {thought}\nAction {i}: {action}\nObservation {i}: {obs}\n"
            current_prompt += step_str

            if to_print:
                print(step_str)

            if done:
                break

        if not done: # If loop finished without 'Finish' action
            obs, r, done, info = step(env, "finish[]") # Default finish
            if not isinstance(info, dict):
                info = {}
            info['finish_action_obs'] = obs
            # Add standard keys if missing (for consistency)
            for key in ["answer", "gt_answer", "reward", "em", "f1"]:
                if key not in info:
                    info[key] = None


        trace_info_updates = info.copy()
        trace_info_updates.update({
            'n_calls': n_calls,
            'n_badcalls': n_badcalls,
            'traj': current_prompt,
            'question_idx': idx,
            'question_text': question,
            'trace_num': trace_num + 1
        })
        # Ensure all keys from info are preserved, and new ones are added/updated
        # The 'info' from env.step() can overwrite keys if not handled carefully.
        # Let's assume 'info' from env.step() is the base and we update it.
        # However, the original code did info.copy() then updated.
        # Let's stick to the original logic for now: info from the last step (or finish[]) is the base.

        # Let's refine how trace_info is constructed to ensure all original info fields are kept
        # and our specific trace metadata is added.
        final_trace_info = info # Start with the info from the last env.step
        final_trace_info.update(trace_info_updates) # Add/overwrite with our collected data

        # Standardize output keys for baseline output
        expected_keys = [
            "steps", "answer", "gt_answer", "question_idx", "reward",
            "em", "f1", "n_calls", "n_badcalls", "traj", "question_text", "trace_num",
            "finish_action_obs"
        ]
        for key in expected_keys:
            if key not in final_trace_info:
                # For 'steps', try to infer from traj if possible
                if key == "steps":
                    # Count 'Thought' occurrences as steps
                    traj_str = final_trace_info.get("traj", "")
                    final_trace_info["steps"] = traj_str.count("Thought ")
                else:
                    final_trace_info[key] = None

        all_traces_info.append(final_trace_info)

        if to_print:
            print(f"(Trace {trace_num + 1}) Info: {final_trace_info}\n")
            if num_traces > 1 and trace_num < num_traces - 1:
                print(f"--- End of Trace {trace_num + 1} ---\n")

    if not all_traces_info:
        if to_print:
            print("Warning: No traces were generated despite num_traces > 0.")
        return "[NO_TRACE_GENERATED]", []

    if num_traces == 1:
        # For single trace, the 'reward' is directly from the trace's info.
        # The original notebook code returned info.get('reward', 0.0) which seems problematic if info is not always a dict with 'reward'
        # Let's assume all_traces_info[0] is the dict we need.
        final_r = all_traces_info[0].get('reward', 0.0) # Default to 0.0 if 'reward' not found
        return final_r, all_traces_info[0]

    else: # num_traces > 1
        if to_print:
            print("\n--- Starting Answer Synthesis ---")

        # MODIFIED: Extract full trajectories instead of just answers
        extracted_trajectories = extract_trajectories_from_traces(all_traces_info)

        if to_print:
            print(f"Extracted Trajectories for Synthesis: {len(extracted_trajectories)} trajectories")
            # for i, traj in enumerate(extracted_trajectories):
            #     print(f"Trajectory {i+1}:\n{traj[:300]}...\n") # Print start of each trajectory for brevity

        if not extracted_trajectories:
            if to_print:
                print("Warning: No trajectories extracted. Cannot synthesize.")
            # Return all trace details even if synthesis fails
            return "[SYNTHESIS_FAILED_NO_EXTRACTED_TRAJECTORIES]", all_traces_info

        synthesized_answer = synthesize_answer_with_llm(extracted_trajectories, question_for_synthesis)

        if to_print:
            print(f"Synthesized Answer: {synthesized_answer}")
            print("--- End of Answer Synthesis ---\n")

        # For multiple traces, the "final answer" is the synthesized one.
        # The second element returned should be the list of all trace infos.
        return synthesized_answer, all_traces_info
