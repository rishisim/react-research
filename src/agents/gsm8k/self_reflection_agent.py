"""Self-Reflection Agent for GSM8K."""

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


def load_self_reflection_prompt():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    prompt_path = os.path.join(script_dir, "prompts", "gsm8k_self_reflection.json")
    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            prompts = json.load(f)
        return prompts.get("self_reflection_verification", "")
    except Exception as e:
        print(f"[WARNING] Could not load self-reflection prompt: {e}")
        return ""


def verify_and_correct_answer(trajectory_info, verification_prompt, to_print=False):
    question = trajectory_info.get("question_text", "")
    trajectory = trajectory_info.get("traj", "")
    answer = trajectory_info.get("answer", "")

    full_prompt = f"""{verification_prompt}

Now verify this trajectory:

Question: \"{question}\"

{trajectory}

The agent's final answer is: {answer}

Provide your verification:"""

    response, token_usage = llm(full_prompt, stop=[], num_traces=1)

    verification_status = "CORRECT"
    if "INCORRECT" in response.upper():
        verification_status = "INCORRECT"

    corrected_answer = answer
    if verification_status == "INCORRECT" and "Final Answer:" in response:
        parts = response.split("Final Answer:")
        if len(parts) > 1:
            corrected_answer = parts[-1].strip().split("\n")[0].strip()

    if to_print:
        print(f"\n[SELF-REFLECTION] Status: {verification_status}")
        print(f"[SELF-REFLECTION] Original: {answer}")
        if verification_status == "INCORRECT":
            print(f"[SELF-REFLECTION] Corrected: {corrected_answer}")

    return {
        "verification_status": verification_status,
        "verification_reasoning": response.strip(),
        "corrected_answer": corrected_answer,
        "input_tokens": token_usage["input_tokens"],
        "output_tokens": token_usage["output_tokens"],
        "total_tokens": token_usage["total_tokens"],
    }


def run_self_reflection(idx, prompt_template=None, to_print=True, split=None):
    if prompt_template is None:
        prompt_template = WEBTHINK_PROMPT_TEMPLATE

    if to_print:
        print("=" * 60)
        print("[FRAMEWORK] Self-Reflection (Single Trace + Verification)")
        print("=" * 60)

    verification_prompt = load_self_reflection_prompt()

    trace_info = run_single_trace(
        idx=idx,
        initial_prompt_template=prompt_template,
        to_print=to_print,
        temperature=0.0,
        split=split,
    )

    question_text = trace_info.get("question_text", "")
    gt_answer = trace_info.get("gt_answer", "")
    initial_answer = trace_info.get("answer", "")

    verification_result = verify_and_correct_answer(trace_info, verification_prompt, to_print=to_print)
    final_answer = verification_result["corrected_answer"]
    answer_was_corrected = verification_result["verification_status"] == "INCORRECT"

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

    total_calls = trace_info.get("n_calls", 0) + 1
    total_badcalls = trace_info.get("n_badcalls", 0)
    total_input_tokens = trace_info.get("input_tokens", 0) + verification_result.get("input_tokens", 0)
    total_output_tokens = trace_info.get("output_tokens", 0) + verification_result.get("output_tokens", 0)

    total_input_tokens += llm_eval.get("judge_input_tokens", 0)
    total_output_tokens += llm_eval.get("judge_output_tokens", 0)

    info_dict = {
        "question_idx": idx,
        "question_text": question_text,
        "initial_answer": initial_answer,
        "answer": final_answer,
        "gt_answer": gt_answer,
        "answer_was_corrected": answer_was_corrected,
        "em": metrics.get("em", 0.0),
        "f1": metrics.get("f1", 0.0),
        "reward": metrics.get("reward", 0.0),
        "n_calls": total_calls,
        "n_badcalls": total_badcalls,
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
        "total_tokens": total_input_tokens + total_output_tokens,
        "verification": {
            "verification_status": verification_result["verification_status"],
            "verification_reasoning": verification_result["verification_reasoning"],
        },
        "traj": trace_info.get("traj", ""),
        "llm_correct": llm_eval["llm_correct"],
        "llm_explanation": llm_eval["llm_explanation"],
        "framework": "self_reflection",
    }

    if to_print:
        print("\n" + "=" * 60)
        print(
            f"[FINAL] Answer: {final_answer} | GT: {gt_answer} | "
            f"EM: {info_dict['em']} | LLM: {info_dict['llm_correct']}"
        )
        print(
            f"[FINAL] Corrected: {answer_was_corrected} | "
            f"Verification: {verification_result['verification_status']}"
        )
        print("=" * 60)

    framework_folder = get_framework_log_folder("self_reflection")
    run_name = get_next_run_number(framework_folder)
    run_folder = os.path.join(framework_folder, run_name)
    os.makedirs(run_folder, exist_ok=True)
    append_to_json(info_dict, os.path.join(run_folder, "results.jsonl"))

    return info_dict["reward"], info_dict


if __name__ == "__main__":
    print("\n[TEST] Running Self-Reflection agent test\n")
    reward, info = run_self_reflection(idx=0, to_print=True, split="test")
    print(f"\n[TEST RESULT] Reward: {reward}, Final Answer: {info['answer']}")
