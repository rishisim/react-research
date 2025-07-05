import os
import time
import re
import json
import sys
import requests # Ensure requests is imported

# Assuming wikienv and wrappers are in the same directory or accessible in PYTHONPATH
import wikienv
import wrappers

# --- LLM Configuration and Interaction ---
from google import genai # Ensuring this matches the notebook
from google.genai import types # Ensuring this matches the notebook

# The client gets the API key from the environment variable `GEMINI_API_KEY`.
_client = None

def get_genai_client():
    global _client
    if _client is None:
        try:
            _client = genai.Client() # As per notebook
        except Exception as e:
            print(f"Error initializing Google GenAI Client: {e}")
            print("Please ensure the GEMINI_API_KEY environment variable is set correctly.")
            # Allow execution to continue, llm calls will fail gracefully
    return _client

def llm(prompt, stop=["\n"], num_traces=1):
    client = get_genai_client()
    if not client:
        print("GenAI Client not initialized. Cannot make LLM call.")
        return "[GENAI_CLIENT_NOT_INITIALIZED]"

    time.sleep(4.1)
    temperature_setting = 0.0 if num_traces == 1 else 0.7

    # Model name from the notebook
    model_name_from_notebook = "gemini-2.5-flash-lite-preview-06-17"

    try:
        # Configuration as per the notebook:
        # Using types.GenerateContentConfig for the 'config' parameter
        # of client.models.generate_content.
        llm_config = types.GenerateContentConfig(
            stop_sequences=stop,
            temperature=temperature_setting,
            max_output_tokens=100,
            top_p=1.0
        )
        # Add thinking_config if it was in the notebook's GenerateContentConfig
        # The notebook shows: thinking_config=types.ThinkingConfig(thinking_budget=0)
        if hasattr(types, 'ThinkingConfig'):
            try:
                # Try to initialize with thinking_config if it's a constructor argument
                llm_config = types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                    stop_sequences=stop,
                    temperature=temperature_setting,
                    max_output_tokens=100,
                    top_p=1.0
                )
            except TypeError:
                # If not a constructor arg, try setting as an attribute if it exists
                if hasattr(llm_config, 'thinking_config'):
                    llm_config.thinking_config = types.ThinkingConfig(thinking_budget=0)
                else:
                    # If cannot be set, proceed without it but log a warning or note
                    if to_print: # (Assuming to_print is available or passed to llm)
                         print("Note: ThinkingConfig could not be applied as expected.")


        # LLM call structure as per the notebook: client.models.generate_content
        response = client.models.generate_content(
            model=model_name_from_notebook,
            contents=prompt,
            config=llm_config # Pass the GenerateContentConfig instance here
        )
        return response.text
    except Exception as e:
        print(f"Error during LLM call: {e}")
        return f"[LLM_ERROR: {e}]"

# --- Environment Setup ---
# (This will be initialized when webthink is called, or can be global)
env = None

def get_fever_env():
    global env
    if env is None:
        env = wikienv.WikiEnv()
        env = wrappers.FeverWrapper(env, split="dev") # Ensure FeverWrapper is used
        env = wrappers.LoggingWrapper(env)
    return env

def step(current_env, action):
    attempts = 0
    while attempts < 10:
        try:
            return current_env.step(action)
        except requests.exceptions.Timeout:
            print(f"Timeout during env.step attempt {attempts+1} for action: {action}")
            attempts += 1
            time.sleep(2) # Wait a bit before retrying
    print(f"Failed to execute step after 10 attempts due to timeout for action: {action}")
    # Fallback if all attempts fail
    return "Timeout after 10 attempts", 0, False, {"error": "API Timeout"}


# --- Prompt Loading ---
PROMPT_FILE_PATH = './prompts/fever.json' # Relative to FEVER_Experiment folder later
WEBTHINK_PROMPT_TEMPLATE = ""

try:
    # Adjust path for loading within the FEVER_Experiment directory later if needed
    # For now, assume it can be loaded if fever.json is copied there
    with open(PROMPT_FILE_PATH, 'r') as f:
        prompt_dict = json.load(f)
    WEBTHINK_PROMPT_TEMPLATE = prompt_dict['webthink_simple3']
except FileNotFoundError:
    print(f"Error: Prompt file not found at {PROMPT_FILE_PATH}")
    print("Please ensure 'fever.json' is in the 'FEVER_Experiment/prompts/' directory.")
    # Provide a default fallback template if file not found
    WEBTHINK_PROMPT_TEMPLATE = """Claim: {claim}
Thought 1: I need to assess the claim.
Action 1: Search[entity related to claim]
Observation 1: [Search results]
Thought 2: [Reasoning based on observation]
Action 2: Finish[SUPPORTS/REFUTES/NOT ENOUGH INFO]
"""
except KeyError:
    print(f"Error: 'webthink_simple3' key not found in {PROMPT_FILE_PATH}")
    # Fallback if key is missing
    WEBTHINK_PROMPT_TEMPLATE = "Claim: {claim}\n" # Minimalist fallback


# --- Answer Extraction and Synthesis ---
def extract_final_answer_from_trace_string(trace_trajectory_string):
    pattern = re.compile(r"^Action \d+: Finish\[(SUPPORTS|REFUTES|NOT ENOUGH INFO)\]\s*$", re.MULTILINE)
    matches = pattern.findall(trace_trajectory_string)
    if matches:
        return matches[-1].strip()
    return None

def extract_answers_from_traces(all_traces_info):
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
            if env_answer in ['SUPPORTS', 'REFUTES', 'NOT ENOUGH INFO']:
                extracted_answers.append(env_answer)
            else:
                extracted_answers.append(None)

    return [ans for ans in extracted_answers if ans is not None]

def synthesize_answer_with_llm(list_of_answers, question_for_context=""):
    if not list_of_answers:
        return "NOT ENOUGH INFO"

    counts = {'SUPPORTS': 0, 'REFUTES': 0, 'NOT ENOUGH INFO': 0}
    valid_answers_found = False
    for ans in list_of_answers:
        if ans in counts:
            counts[ans] += 1
            valid_answers_found = True

    if not valid_answers_found:
        return "NOT ENOUGH INFO"

    # Majority vote logic for FEVER
    if counts['SUPPORTS'] > counts['REFUTES'] and counts['SUPPORTS'] > counts['NOT ENOUGH INFO']:
        return 'SUPPORTS'
    elif counts['REFUTES'] > counts['SUPPORTS'] and counts['REFUTES'] > counts['NOT ENOUGH INFO']:
        return 'REFUTES'
    elif counts['NOT ENOUGH INFO'] > counts['SUPPORTS'] and counts['NOT ENOUGH INFO'] > counts['REFUTES']:
        return 'NOT ENOUGH INFO'
    else: # Ties or ambiguous cases default to NOT ENOUGH INFO
        return "NOT ENOUGH INFO"

# --- Core Webthink Logic ---
def webthink(idx=None, initial_prompt_template=None, to_print=True, num_traces=1):
    fever_env = get_fever_env() # Get initialized environment

    if initial_prompt_template is None:
        initial_prompt_template = WEBTHINK_PROMPT_TEMPLATE


    all_traces_info = []
    question_for_synthesis = ""

    if num_traces <= 0:
        if to_print:
            print(f"Warning: webthink called with num_traces = {num_traces}. Must be > 0.")
        # Consistent return type: (reward, info_dict)
        return 0, {'error': 'num_traces must be > 0', 'traces': [], 'question_idx': idx, 'answer': '[INVALID_NUM_TRACES]', 'em':0, 'f1':0, 'reward':0}

    for trace_num in range(num_traces):
        question = fever_env.reset(idx=idx) # Reset environment for each trace
        if trace_num == 0:
            question_for_synthesis = question

        # The prompt template for FEVER expects the claim directly.
        # The original FEVER notebook's prompt_dict['webthink_simple3'] is just the prefix.
        # The question/claim is appended after it.
        current_prompt = initial_prompt_template + question + "\n"


        if to_print:
            print(f"--- Trace {trace_num + 1}/{num_traces} ---")
            print(f"Index: {idx}, Claim: {question}")

        n_calls, n_badcalls = 0, 0
        current_trace_steps = []

        for i in range(1, 8): # Max 7 steps per trace
            n_calls += 1
            thought_action = llm(current_prompt + f"Thought {i}:", stop=[f"\nObservation {i}:"], num_traces=num_traces)

            try:
                thought, action = thought_action.strip().split(f"\nAction {i}: ")
            except:
                if to_print:
                    print(f"Error parsing thought/action: '{thought_action}'")
                n_badcalls += 1
                # Attempt to recover or use a default action if parsing fails
                thought = thought_action.strip().split('\n')[0] if thought_action else "Error in thought generation"
                # Let LLM try to generate action again, or use a default if that also fails
                action_prompt = current_prompt + f"Thought {i}: {thought}\nAction {i}:"
                action = llm(action_prompt, stop=[f"\n"], num_traces=num_traces).strip()
                if not action or "Finish[" not in action and "Search[" not in action and "Lookup[" not in action : # FEVER specific actions
                    action = "Finish[NOT ENOUGH INFO]" # Default recovery action for FEVER
                    if to_print:
                         print(f"Recovered with action: {action}")

            # Ensure action is a string before lowercasing
            if not isinstance(action, str):
                action = str(action) # Convert if it's not (e.g. due to LLM error)

            obs, r, done, info = step(fever_env, action[0].lower() + action[1:]) # Ensure action is not empty
            obs = obs.replace('\\n', '') if isinstance(obs, str) else str(obs)

            step_str = f"Thought {i}: {thought}\nAction {i}: {action}\nObservation {i}: {obs}\n"
            current_prompt += step_str
            current_trace_steps.append(step_str)

            if to_print:
                print(step_str)

            if done:
                break

        if not isinstance(info, dict): # Ensure info is a dict
            info = {}

        if not done:
            if to_print:
                print(f"Agent did not finish in {i} steps. Forcing Finish[NOT ENOUGH INFO].")
            # Force a finish action for FEVER if not naturally concluded
            obs_finish, r_finish, done_finish, info_finish = step(fever_env, "finish[NOT ENOUGH INFO]")
            info.update(info_finish)
            if 'answer' not in info or not info['answer']: # Ensure answer is set
                 info['answer'] = 'NOT ENOUGH INFO'
            # Append the forced finish step to the trajectory for completeness
            forced_step_str = f"Thought {i+1}: Agent did not finish. Forcing.\nAction {i+1}: Finish[NOT ENOUGH INFO]\nObservation {i+1}: {obs_finish}\n"
            current_trace_steps.append(forced_step_str)


        trace_info_package = info.copy() # Start with env info
        trace_info_package.update({
            'n_calls': n_calls,
            'n_badcalls': n_badcalls,
            'traj': initial_prompt_template + question + "\n" + "".join(current_trace_steps),
            'question_idx': idx,
            'question_text': question, # This is the claim for FEVER
            'trace_num': trace_num + 1,
            # Ensure 'answer' from info is present, default if somehow missing
            'answer': info.get('answer', 'NOT ENOUGH INFO' if not done else '[ERROR_NO_ANSWER_IN_INFO]')
        })
        all_traces_info.append(trace_info_package)

        if to_print:
            print(f"(Trace {trace_num + 1}) Info: {trace_info_package}\n")
            if num_traces > 1 and trace_num < num_traces - 1:
                print(f"--- End of Trace {trace_num + 1} ---\n")

    if not all_traces_info:
        if to_print:
            print("Warning: No traces were generated.")
        return 0, {'question_idx': idx, 'question_text': question_for_synthesis, 'answer': '[NO_TRACE_GENERATED]', 'em':0, 'f1':0, 'reward':0, 'traces': [], 'n_calls':0, 'n_badcalls':0}

    if num_traces == 1:
        final_info_package = all_traces_info[0]
        final_reward = final_info_package.get('em', 0.0) # 'em' is the reward for FEVER
        return final_reward, final_info_package

    else: # num_traces > 1
        if to_print:
            print("\n--- Starting Answer Synthesis for FEVER ---")

        extracted_answers = extract_answers_from_traces(all_traces_info)

        if to_print:
            print(f"Extracted Answers for Synthesis: {extracted_answers}")

        if not extracted_answers:
            if to_print:
                print("Warning: No answers extracted from traces. Defaulting to NOT ENOUGH INFO.")
            synthesized_answer = "NOT ENOUGH INFO"
        else:
            synthesized_answer = synthesize_answer_with_llm(extracted_answers, question_for_synthesis)

        if to_print:
            print(f"Synthesized Answer: {synthesized_answer}")
            print("--- End of Answer Synthesis ---\n")

        gt_answer = all_traces_info[0].get('gt_answer', 'UNKNOWN_GT_ANSWER')
        em_score = 1.0 if synthesized_answer == gt_answer else 0.0

        total_calls = sum(t.get('n_calls', 0) for t in all_traces_info)
        total_badcalls = sum(t.get('n_badcalls', 0) for t in all_traces_info)

        synthesized_info = {
            'question_idx': idx,
            'question_text': question_for_synthesis,
            'answer': synthesized_answer,
            'gt_answer': gt_answer,
            'em': em_score,
            'f1': em_score, # For FEVER, EM and F1 are the same for SUPPORTS/REFUTES/NEI
            'reward': em_score,
            'n_calls': total_calls,
            'n_badcalls': total_badcalls,
            'num_traces_run': num_traces,
            'individual_traces': all_traces_info
        }
        return em_score, synthesized_info

# --- Utility for run_experiments.py ---
def append_to_json(data, filename):
    """Appends data to a JSON file that stores a list of JSON objects."""
    if os.path.exists(filename):
        with open(filename, 'r+') as f:
            try:
                file_data = json.load(f)
            except json.JSONDecodeError: # Handle empty or malformed file
                file_data = []

            if isinstance(file_data, list):
                file_data.append(data)
            else: # If existing data is not a list, start a new list
                file_data = [data]

            f.seek(0)
            json.dump(file_data, f, indent=4)
            f.truncate()
    else:
        with open(filename, 'w') as f:
            json.dump([data], f, indent=4)

if __name__ == '__main__':
    # This is a simple test block, similar to the FEVER.ipynb demonstration cells
    print("--- Running Self-Test for fever_agent.py ---")

    # Ensure GEMINI_API_KEY is set before running this test
    if not os.getenv('GEMINI_API_KEY'):
        print("CRITICAL: GEMINI_API_KEY environment variable not set. Aborting test.")
        sys.exit(1)

    print("\n--- Running standard ReAct (num_traces=1) for one FEVER example ---")
    # FEVER dev set has many examples, let's pick one, e.g., index 0 or a random one
    # For consistency with notebook, let's try to use a known index if possible,
    # but the dataset isn't loaded here directly, it's via env.reset(idx=...)

    # Initialize env for test
    test_env = get_fever_env()
    # There are 19998 claims in the paper_dev.jsonl (which FeverWrapper uses by default for "dev")
    # The notebook uses idxs = list(range(7405)) - this might be from a different dataset version or split.
    # Let's use a low index that should exist.
    example_idx = 3687 # From the notebook's example output

    # Make sure prompt is loaded
    if not WEBTHINK_PROMPT_TEMPLATE or "{" not in WEBTHINK_PROMPT_TEMPLATE : # Basic check
        print("CRITICAL: WEBTHINK_PROMPT_TEMPLATE not loaded correctly. Aborting test.")
        # Attempt to reload it here, assuming fever.json is in FEVER_Experiment/prompts/
        try:
            with open('prompts/fever.json', 'r') as f_retry: # Path relative to FEVER_Experiment
                prompt_dict_retry = json.load(f_retry)
            WEBTHINK_PROMPT_TEMPLATE = prompt_dict_retry['webthink_simple3']
            print("Successfully reloaded prompt template for test.")
        except Exception as e_prompt_retry:
            print(f"Failed to reload prompt template: {e_prompt_retry}")
            sys.exit(1)


    print(f"Using FEVER example with index: {example_idx}\n")

    reward_single, info_single = webthink(idx=example_idx, to_print=True, num_traces=1)

    print("\n--- Standard ReAct (num_traces=1) Summary ---")
    print(f"Question Index: {info_single.get('question_idx')}")
    print(f"Claim: {info_single.get('question_text')}")
    print(f"Agent's Answer: {info_single.get('answer')}")
    print(f"Ground Truth: {info_single.get('gt_answer')}")
    print(f"EM Score (Reward): {info_single.get('em')}")
    print(f"LLM Calls: {info_single.get('n_calls')}")
    # print(f"Trajectory:\n{info_single.get('traj')}")


    print("\n--- Running updated ReAct (num_traces=3) for one FEVER example ---")
    # Using the same example_idx for consistency
    print(f"Using FEVER example with index: {example_idx} for multi-trace run\n")

    synthesized_reward, synthesized_info = webthink(idx=example_idx, to_print=True, num_traces=3)

    print("\n--- Final Synthesized Summary (across all traces) ---")
    print(f"Question Index: {synthesized_info.get('question_idx')}")
    print(f"Claim: {synthesized_info.get('question_text')}")
    print(f"Synthesized Answer: {synthesized_info.get('answer')}")
    print(f"Ground Truth: {synthesized_info.get('gt_answer')}")
    print(f"Synthesized EM Score (Reward): {synthesized_info.get('em')}")
    print(f"Total LLM Calls (all traces): {synthesized_info.get('n_calls')}")
    print(f"Number of Traces Run: {synthesized_info.get('num_traces_run')}")

    # Detailed individual traces if needed
    # for i, trace_detail in enumerate(synthesized_info.get('individual_traces', [])):
    #     print(f"\nTrace {i+1} Details:")
    #     print(f"  Answer: {trace_detail.get('answer')}, EM: {trace_detail.get('em')}")
    #     # print(f"  Trajectory:\n{trace_detail.get('traj')}")

    print("\n--- fever_agent.py Self-Test Complete ---")
