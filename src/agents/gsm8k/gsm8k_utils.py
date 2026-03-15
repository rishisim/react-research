"""
Shared utilities for GSM8K agents.
"""

import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

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

_gemini_client = None


def get_gemini_client():
    """
    Build (and cache) a Gemini client for GSM8K runs.
    Prefers GEMINI_API_KEY_GSM8K while preserving existing fallback behavior.
    """
    global _gemini_client
    if _gemini_client is not None:
        return _gemini_client

    api_key = (
        os.environ.get("GEMINI_API_KEY_GSM8K")
        or os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
    )
    _gemini_client = genai.Client(api_key=api_key) if api_key else genai.Client()
    return _gemini_client


def llm(prompt, stop=None, temperature=None, num_traces=1):
    """
    Call the language model with retry and model routing.
    """
    import random

    if stop is None:
        stop = ["\n"]

    time.sleep(0.2)
    if temperature is None:
        temperature_setting = 0.0 if num_traces == 1 else 0.7
    else:
        temperature_setting = temperature

    default_token_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    active_model = os.environ.get("ACTIVE_MODEL", "gemini-2.5-flash")

    # Qwen / local-model path via OpenAI-compatible endpoint.
    if "qwen" in active_model.lower():
        import openai

        qwen_base_url = (
            os.environ.get("QWEN_OPENAI_BASE_URL")
            or os.environ.get("OPENAI_BASE_URL")
            or "http://127.0.0.1:1234/v1"
        )
        qwen_api_key = os.environ.get("QWEN_OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY") or "lm-studio"
        qwen_max_retries = int(os.environ.get("QWEN_MAX_RETRIES", "3"))
        qwen_retry_delay = float(os.environ.get("QWEN_RETRY_DELAY", "1.0"))

        oai_client = openai.OpenAI(base_url=qwen_base_url, api_key=qwen_api_key)
        for attempt in range(qwen_max_retries):
            try:
                oai_response = oai_client.chat.completions.create(
                    model=active_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature_setting,
                    max_tokens=512,
                    top_p=1.0,
                    stop=stop,
                )
                text_response = oai_response.choices[0].message.content
                token_usage = {
                    "input_tokens": oai_response.usage.prompt_tokens if hasattr(oai_response.usage, "prompt_tokens") else 0,
                    "output_tokens": oai_response.usage.completion_tokens if hasattr(oai_response.usage, "completion_tokens") else 0,
                    "total_tokens": oai_response.usage.total_tokens if hasattr(oai_response.usage, "total_tokens") else 0,
                }
                return text_response, token_usage
            except Exception as e:
                print(
                    f"[ERROR] Qwen LLM call failed (attempt {attempt + 1}/{qwen_max_retries}) "
                    f"against {qwen_base_url}: {e}"
                )
                if attempt < qwen_max_retries - 1:
                    time.sleep(qwen_retry_delay)

        return "Finish[null]", default_token_usage

    request_timeout_seconds = 25
    max_retries = 3
    base_delay = 0.4

    def make_api_call():
        return get_gemini_client().models.generate_content(
            model=active_model,
            contents=prompt,
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_budget=0),
                stop_sequences=stop,
                temperature=temperature_setting,
                max_output_tokens=512,
                top_p=1.0,
            ),
        )

    for attempt in range(max_retries):
        executor = ThreadPoolExecutor(max_workers=1)
        try:
            future = executor.submit(make_api_call)
            try:
                response = future.result(timeout=request_timeout_seconds)
            except FuturesTimeoutError:
                print(
                    f"[TIMEOUT] Gemini call timed out after {request_timeout_seconds}s "
                    f"(attempt {attempt + 1}/{max_retries})"
                )
                raise TimeoutError("Gemini API timeout")

            if response and response.text:
                token_usage = default_token_usage.copy()
                if hasattr(response, "usage_metadata") and response.usage_metadata:
                    token_usage = {
                        "input_tokens": getattr(response.usage_metadata, "prompt_token_count", 0) or 0,
                        "output_tokens": getattr(response.usage_metadata, "candidates_token_count", 0) or 0,
                        "total_tokens": getattr(response.usage_metadata, "total_token_count", 0) or 0,
                    }
                return response.text, token_usage

            print(f"[WARNING] Empty Gemini response (attempt {attempt + 1}/{max_retries})")
        except Exception as e:
            print(f"[WARNING] Gemini call failed (attempt {attempt + 1}/{max_retries}): {str(e)[:160]}")
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

        if attempt < max_retries - 1:
            delay = base_delay * (2 ** attempt) + random.uniform(0, 0.2)
            time.sleep(delay)

    print("[FATAL] All Gemini retries failed, using fallback answer.")
    return "Finish[null]", default_token_usage


# --- Environment Setup ---
_env_by_split = {}


def _get_target_split():
    return os.environ.get("GSM8K_SPLIT", "test")


def get_gsm8k_env(split=None):
    """Get or initialize GSM8K environment for a split."""
    target_split = split or _get_target_split()
    if target_split not in _env_by_split:
        env = wikienv.WikiEnv()
        env = wrappers.GSM8KWrapper(env, split=target_split)
        env = wrappers.LoggingWrapper(env)
        _env_by_split[target_split] = env
    return _env_by_split[target_split]


def step(current_env, action):
    """Execute a step with retry logic."""
    attempts = 0
    while attempts < 10:
        try:
            return current_env.step(action)
        except requests.exceptions.Timeout:
            print(f"[WARNING] Timeout during env.step attempt {attempts + 1} for action: {action}")
            attempts += 1
            time.sleep(0.5)

    print(f"[ERROR] Failed step after 10 attempts for action: {action}")
    return "Timeout after 10 attempts", 0, False, {"error": "API Timeout"}


# --- Prompt Loading ---
PROMPT_FILE_PATH = "./prompts/gsm8k.json"
WEBTHINK_PROMPT_TEMPLATE = ""

try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    prompt_path = os.path.join(script_dir, "prompts", "gsm8k.json")
    with open(prompt_path, "r", encoding="utf-8") as f:
        prompt_dict = json.load(f)
    WEBTHINK_PROMPT_TEMPLATE = prompt_dict["webthink_simple_math"]
except Exception as e:
    print(f"[WARNING] Could not load GSM8K prompt at {PROMPT_FILE_PATH}: {e}")
    WEBTHINK_PROMPT_TEMPLATE = (
        "Solve the math question with Thought/Action/Observation steps.\n"
        "Use Finish[answer] with only the final numeric answer.\n\n"
    )


# --- Answer extraction / synthesis ---
def extract_final_answer_from_trace_string(trace_trajectory_string):
    pattern = re.compile(r"^Action \d+: Finish\[(.*?)\]\s*$", re.MULTILINE)
    matches = pattern.findall(trace_trajectory_string)
    if matches:
        return matches[-1].strip()
    return None


def extract_answers_from_traces(all_traces_info):
    extracted_answers = []
    if not isinstance(all_traces_info, list):
        return extracted_answers

    for trace_info in all_traces_info:
        trajectory = trace_info.get("traj", "")
        answer_from_traj = extract_final_answer_from_trace_string(trajectory)
        if answer_from_traj is not None:
            extracted_answers.append(answer_from_traj)
        else:
            env_answer = trace_info.get("answer")
            extracted_answers.append(env_answer if env_answer else None)

    return [ans for ans in extracted_answers if ans is not None]


def extract_trajectories_from_traces(all_traces_info):
    extracted_trajectories = []
    if not isinstance(all_traces_info, list):
        return extracted_trajectories

    for i, trace_info in enumerate(all_traces_info):
        trajectory = trace_info.get("traj", "")
        if trajectory:
            extracted_trajectories.append(trajectory)
        else:
            extracted_trajectories.append(f"Trace {i + 1} was empty or missing.")

    return extracted_trajectories


def synthesize_answer_with_llm(list_of_trajectories, question_for_context=""):
    default_token_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    if not list_of_trajectories:
        return "null", default_token_usage

    valid_trajectories = [str(t).strip() for t in list_of_trajectories if str(t).strip()]
    if not valid_trajectories:
        return "null", default_token_usage

    prompt_template = """You are an expert math answer aggregator.
Review the trajectories and return the single best final numeric answer.

Question: {question_context}

Reasoning Trajectories:
{formatted_trajectories}

Output ONLY the final numeric answer, without explanation.
Final Answer:"""

    formatted_trajectories = ""
    for i, traj in enumerate(valid_trajectories):
        formatted_trajectories += f"--- Trajectory {i + 1} ---\\n{traj}\\n--- End of Trajectory {i + 1} ---\\n\\n"
    formatted_trajectories = formatted_trajectories.strip()

    synthesizer_prompt = prompt_template.format(
        question_context=question_for_context,
        formatted_trajectories=formatted_trajectories,
    )

    llm_response, token_usage = llm(synthesizer_prompt, stop=["\n"], num_traces=1)
    return llm_response.strip(), token_usage


def majority_vote_semantic(answers, question):
    default_token_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    if not answers:
        return "null", default_token_usage

    if len(answers) == 1:
        return answers[0], default_token_usage

    answer_list = "\n".join(f"{i + 1}. {ans}" for i, ans in enumerate(answers))

    prompt = f"""Question: {question}

Three solving attempts produced these answers:
{answer_list}

Task: Select the most likely correct final numeric answer.
Treat formatting variants as equivalent (e.g., 1,234 and 1234).
If no clear majority exists, choose the most plausible one.

Output ONLY the final numeric answer.
Majority Answer:"""

    response, token_usage = llm(prompt, stop=["\n"], num_traces=1)
    return response.strip(), token_usage


# --- LLM-as-judge evaluation ---
def llm_judge_answer(question, prediction, ground_truth):
    pred_norm = wrappers.parse_gsm8k_prediction_answer(prediction)
    gt_norm = wrappers.parse_gsm8k_prediction_answer(ground_truth)

    if pred_norm is not None and gt_norm is not None and pred_norm == gt_norm:
        return {
            "llm_correct": True,
            "llm_explanation": "Normalized numeric answers match exactly.",
            "judge_input_tokens": 0,
            "judge_output_tokens": 0,
            "judge_total_tokens": 0,
        }

    prompt = f"""You are an evaluator for GSM8K math answers.

Question: {question}
Ground Truth Answer: {ground_truth}
Predicted Answer: {prediction}

Return whether the prediction is mathematically equivalent to ground truth.
Equivalent numeric formatting (comma separators, trailing zeros) should be considered CORRECT.

Output format:
Explanation: <brief reason>
Label: CORRECT or INCORRECT"""

    response, token_usage = llm(prompt, stop=[], num_traces=1)

    upper = response.upper()
    is_correct = "CORRECT" in upper and "INCORRECT" not in upper

    explanation = response.strip()
    if "Explanation:" in response:
        parts = response.split("Label:")
        explanation = parts[0].replace("Explanation:", "").strip() if parts else response.strip()

    return {
        "llm_correct": is_correct,
        "llm_explanation": explanation,
        "judge_input_tokens": token_usage["input_tokens"],
        "judge_output_tokens": token_usage["output_tokens"],
        "judge_total_tokens": token_usage["total_tokens"],
    }


# --- Core trace execution ---
def run_single_trace(idx, initial_prompt_template, to_print=True, temperature=None, split=None):
    gsm8k_env = get_gsm8k_env(split=split)

    question = gsm8k_env.reset(idx=idx)
    current_prompt = initial_prompt_template + question + "\n"

    if to_print:
        print(f"[TRACE] Index: {idx}")
        print(f"[QUESTION] {question}")

    n_calls, n_badcalls = 0, 0
    total_input_tokens, total_output_tokens = 0, 0
    llm_calls = []
    current_trace_steps = []

    num_traces_param = 1 if temperature is None or temperature == 0.0 else 3

    for i in range(1, 8):
        n_calls += 1
        thought_action, token_usage = llm(
            current_prompt + f"Thought {i}:",
            stop=[f"\nObservation {i}:"],
            num_traces=num_traces_param,
        )
        total_input_tokens += token_usage["input_tokens"]
        total_output_tokens += token_usage["output_tokens"]
        llm_calls.append(
            {
                "call_num": n_calls,
                "step": i,
                "type": "thought_action",
                "input_tokens": token_usage["input_tokens"],
                "output_tokens": token_usage["output_tokens"],
                "total_tokens": token_usage["total_tokens"],
            }
        )

        try:
            thought, action = thought_action.strip().split(f"\nAction {i}: ")
        except Exception:
            if to_print:
                print(f"[ERROR] Parsing thought/action: '{thought_action}'")
            n_badcalls += 1
            thought = thought_action.strip().split("\n")[0] if thought_action else "Error in thought generation"
            action_prompt = current_prompt + f"Thought {i}: {thought}\nAction {i}:"
            action, recovery_usage = llm(action_prompt, stop=["\n"], num_traces=num_traces_param)
            action = action.strip()
            total_input_tokens += recovery_usage["input_tokens"]
            total_output_tokens += recovery_usage["output_tokens"]
            n_calls += 1
            llm_calls.append(
                {
                    "call_num": n_calls,
                    "step": i,
                    "type": "action_recovery",
                    "input_tokens": recovery_usage["input_tokens"],
                    "output_tokens": recovery_usage["output_tokens"],
                    "total_tokens": recovery_usage["total_tokens"],
                }
            )
            if not action or not any(k in action for k in ["Finish[", "Search[", "Lookup[", "Think["]):
                action = "Finish[null]"

        if not isinstance(action, str):
            action = str(action)

        action_lowercase = action[0].lower() + action[1:] if action else action
        obs, r, done, info = step(gsm8k_env, action_lowercase)
        obs = obs.replace("\\n", "") if isinstance(obs, str) else str(obs)

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
        obs_finish, _, _, info_finish = step(gsm8k_env, "finish[null]")
        info.update(info_finish)
        if "answer" not in info or not info["answer"]:
            info["answer"] = "null"
        forced_step_str = (
            f"Thought {i + 1}: Agent did not finish. Forcing.\n"
            f"Action {i + 1}: Finish[null]\n"
            f"Observation {i + 1}: {obs_finish}\n"
        )
        current_trace_steps.append(forced_step_str)

    answer = info.get("answer", "null")
    gt_answer = info.get("gt_answer", "")
    question_text = question

    llm_eval = llm_judge_answer(question_text, answer, gt_answer)
    total_input_tokens += llm_eval.get("judge_input_tokens", 0)
    total_output_tokens += llm_eval.get("judge_output_tokens", 0)
    llm_calls.append(
        {
            "call_num": n_calls + 1,
            "step": "judge",
            "type": "llm_judge",
            "input_tokens": llm_eval.get("judge_input_tokens", 0),
            "output_tokens": llm_eval.get("judge_output_tokens", 0),
            "total_tokens": llm_eval.get("judge_total_tokens", 0),
        }
    )

    trace_info = info.copy()
    trace_info.update(
        {
            "n_calls": n_calls,
            "n_badcalls": n_badcalls,
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
            "total_tokens": total_input_tokens + total_output_tokens,
            "llm_calls": llm_calls,
            "traj": initial_prompt_template + question + "\n" + "".join(current_trace_steps),
            "question_idx": idx,
            "question_text": question_text,
            "answer": answer,
            "llm_correct": llm_eval["llm_correct"],
            "llm_explanation": llm_eval["llm_explanation"],
        }
    )

    if to_print:
        print(
            f"[RESULT] Answer: {trace_info['answer']} | GT: {trace_info.get('gt_answer', 'UNKNOWN')} "
            f"| EM: {trace_info.get('em', 0.0)} | LLM: {llm_eval['llm_correct']}\n"
        )

    return trace_info


# --- Result logging helpers ---
def append_to_json(data, filename):
    if os.path.exists(filename):
        with open(filename, "r+", encoding="utf-8") as f:
            try:
                file_data = json.load(f)
            except json.JSONDecodeError:
                file_data = []

            if isinstance(file_data, list):
                file_data.append(data)
            else:
                file_data = [data]

            f.seek(0)
            json.dump(file_data, f, indent=2, ensure_ascii=False)
            f.truncate()
    else:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump([data], f, indent=2, ensure_ascii=False)


def get_next_run_number(framework_folder):
    if not os.path.exists(framework_folder):
        return "run_001"

    existing_runs = []
    for item in os.listdir(framework_folder):
        match = re.match(r"run_(\d+)", item)
        if match:
            existing_runs.append(int(match.group(1)))

    if not existing_runs:
        return "run_001"

    next_num = max(existing_runs) + 1
    return f"run_{next_num:03d}"


def get_framework_log_folder(framework):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))
    active_model = os.environ.get("ACTIVE_MODEL", "gemini-2.5-flash")
    model_bucket = "qwen" if "qwen" in active_model.lower() else "gemini"
    return os.path.join(project_root, "results", "gsm8k", model_bucket, framework)
