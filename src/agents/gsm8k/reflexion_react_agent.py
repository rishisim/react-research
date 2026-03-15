"""Reflexion ReAct Agent for GSM8K."""

import json
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from gsm8k_utils import (
    WEBTHINK_PROMPT_TEMPLATE,
    append_to_json,
    get_framework_log_folder,
    get_gsm8k_env,
    get_next_run_number,
    llm,
    llm_judge_answer,
    run_single_trace,
)


def load_verification_prompt():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    prompt_path = os.path.join(script_dir, "prompts", "gsm8k_reflexion.json")
    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            prompts = json.load(f)
        return prompts.get("reflexion_verification", "")
    except Exception as e:
        print(f"[WARNING] Could not load reflexion prompt: {e}")
        return ""


def verify_answer(trajectory_info, verification_prompt, to_print=False):
    question = trajectory_info.get("question_text", "")
    trajectory = trajectory_info.get("traj", "")
    answer = trajectory_info.get("answer", "")

    full_prompt = f"""{verification_prompt}

Question: \"{question}\"

{trajectory}

The agent's final answer was: {answer}"""

    response, token_usage = llm(full_prompt, stop=[], num_traces=1)
    plan = response.strip()
    if "Plan:" in response:
        plan = response.split("Plan:")[-1].strip()

    if to_print:
        print("\n[REFLECTION] Plan generated:")
        print(f"{plan[:300]}...")

    return {
        "plan": plan,
        "full_response": response.strip(),
        "input_tokens": token_usage["input_tokens"],
        "output_tokens": token_usage["output_tokens"],
        "total_tokens": token_usage["total_tokens"],
    }


def run_reflexion_react(idx, prompt_template=None, to_print=True, max_trials=3, split=None):
    if prompt_template is None:
        prompt_template = WEBTHINK_PROMPT_TEMPLATE

    if to_print:
        print("=" * 60)
        print(f"[FRAMEWORK] Reflexion ReAct (Up to {max_trials} Trials)")
        print("=" * 60)

    verification_prompt = load_verification_prompt()

    traces = {}
    reflections = {}
    num_trials_run = 0
    final_answer = None
    question_text = None
    gt_answer = None

    for trial_num in range(1, max_trials + 1):
        if to_print:
            print(f"\n--- Trial {trial_num} ---")

        if trial_num == 1:
            current_prompt = prompt_template
        else:
            memory_context = ""
            start_trial = max(1, trial_num - 3)
            for t in range(start_trial, trial_num):
                plan_t = reflections.get(t, {}).get("plan", f"[Trial {t} plan not available]")
                memory_context += f"Trial #{t}: {plan_t}\n"

            plan_prompt = f"""{verification_prompt}

Previous reflexions:
{memory_context}
"""
            reflection_result = verify_answer(traces[trial_num - 1], plan_prompt, to_print=to_print)
            new_plan = reflection_result.get("plan", "")
            reflections[trial_num - 1] = reflection_result
            current_prompt = f"Based on this plan: {new_plan}\n\n{prompt_template}"

        trace_result = run_single_trace(
            idx=idx,
            initial_prompt_template=current_prompt,
            to_print=to_print,
            temperature=0.0,
            split=split,
        )

        traces[trial_num] = trace_result
        num_trials_run = trial_num
        question_text = trace_result.get("question_text", "")
        gt_answer = trace_result.get("gt_answer", "")
        final_answer = trace_result.get("answer")

        llm_eval = llm_judge_answer(question_text, final_answer, gt_answer)
        is_correct = llm_eval["llm_correct"]

        if to_print:
            print(f"[TRIAL {trial_num}] Answer: {final_answer}")
            print(f"[EVALUATION] LLM Judge: {'CORRECT' if is_correct else 'INCORRECT'}")

        if is_correct:
            reflections[trial_num] = {
                "plan": "No plan needed - answer was correct.",
                "full_response": "Task completed successfully.",
                "success": True,
            }
            break

        if trial_num == max_trials:
            reflections[trial_num] = verify_answer(trace_result, verification_prompt, to_print=to_print)

    gsm8k_env = get_gsm8k_env(split=split)
    inner_env = gsm8k_env
    while hasattr(inner_env, "env") and not hasattr(inner_env, "get_metrics"):
        inner_env = inner_env.env

    metrics = inner_env.get_metrics({"answer": final_answer}) if hasattr(inner_env, "get_metrics") else {
        "em": 0,
        "f1": 0,
        "reward": 0,
    }

    llm_eval = llm_judge_answer(question_text, final_answer, gt_answer)

    total_calls = 0
    total_badcalls = 0
    total_input_tokens = 0
    total_output_tokens = 0

    for trial_num in range(1, num_trials_run + 1):
        total_calls += traces[trial_num].get("n_calls", 0)
        total_badcalls += traces[trial_num].get("n_badcalls", 0)
        total_input_tokens += traces[trial_num].get("input_tokens", 0)
        total_output_tokens += traces[trial_num].get("output_tokens", 0)

    for reflection in reflections.values():
        if isinstance(reflection, dict):
            total_input_tokens += reflection.get("input_tokens", 0)
            total_output_tokens += reflection.get("output_tokens", 0)

    total_input_tokens += llm_eval.get("judge_input_tokens", 0)
    total_output_tokens += llm_eval.get("judge_output_tokens", 0)
    total_calls += num_trials_run

    traces_info = {}
    for trial_num in range(1, num_trials_run + 1):
        traces_info[f"trace_{trial_num}"] = {
            "answer": traces[trial_num].get("answer"),
            "em": traces[trial_num].get("em", 0.0),
            "n_calls": traces[trial_num].get("n_calls", 0),
            "input_tokens": traces[trial_num].get("input_tokens", 0),
            "output_tokens": traces[trial_num].get("output_tokens", 0),
            "traj": traces[trial_num].get("traj", ""),
        }

    info_dict = {
        "question_idx": idx,
        "question_text": question_text,
        "answer": final_answer,
        "gt_answer": gt_answer,
        "em": metrics.get("em", 0.0),
        "f1": metrics.get("f1", 0.0),
        "reward": metrics.get("reward", 0.0),
        "n_calls": total_calls,
        "n_badcalls": total_badcalls,
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
        "total_tokens": total_input_tokens + total_output_tokens,
        "num_trials": num_trials_run,
        "max_trials": max_trials,
        "all_traces": traces_info,
        "all_reflections": reflections,
        "final_plan": reflections.get(num_trials_run, {}).get("plan", ""),
        "llm_correct": llm_eval["llm_correct"],
        "llm_explanation": llm_eval["llm_explanation"],
        "framework": "reflexion",
    }

    if to_print:
        print("\n" + "=" * 60)
        print(
            f"[FINAL] Answer: {final_answer} | GT: {gt_answer} | "
            f"EM: {info_dict['em']} | LLM: {info_dict['llm_correct']}"
        )
        print(f"[TRIALS] Ran {num_trials_run}/{max_trials} trials")
        print("=" * 60)

    framework_folder = get_framework_log_folder("reflexion")
    run_name = get_next_run_number(framework_folder)
    run_folder = os.path.join(framework_folder, run_name)
    os.makedirs(run_folder, exist_ok=True)
    append_to_json(info_dict, os.path.join(run_folder, "results.jsonl"))

    return info_dict["reward"], info_dict


if __name__ == "__main__":
    print("\n[TEST] Running Reflexion ReAct agent test\n")
    reward, info = run_reflexion_react(idx=0, to_print=True, max_trials=3, split="test")
    print(f"\n[TEST RESULT] Reward: {reward}, Answer: {info['answer']}")
