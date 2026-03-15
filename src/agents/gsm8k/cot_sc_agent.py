"""CoT-SC Agent for GSM8K."""

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from gsm8k_utils import (
    WEBTHINK_PROMPT_TEMPLATE,
    append_to_json,
    extract_trajectories_from_traces,
    get_framework_log_folder,
    get_gsm8k_env,
    get_next_run_number,
    llm_judge_answer,
    run_single_trace,
    synthesize_answer_with_llm,
)


def run_cot_sc(idx, prompt_template=None, to_print=True, num_traces=3, split=None):
    if prompt_template is None:
        prompt_template = WEBTHINK_PROMPT_TEMPLATE

    if to_print:
        print("=" * 60)
        print("[FRAMEWORK] CoT-SC (Multi-Trace + LLM Synthesis)")
        print(f"[CONFIG] Number of traces: {num_traces}")
        print("=" * 60)

    all_traces = []
    question_text = None
    gt_answer = None

    for trace_num in range(num_traces):
        if to_print:
            print(f"\n--- Trace {trace_num + 1}/{num_traces} ---")

        trace_info = run_single_trace(
            idx=idx,
            initial_prompt_template=prompt_template,
            to_print=to_print,
            temperature=0.7,
            split=split,
        )
        all_traces.append(trace_info)

        if trace_num == 0:
            question_text = trace_info.get("question_text")
            gt_answer = trace_info.get("gt_answer")

        if to_print:
            print(f"[TRACE {trace_num + 1}] Answer: {trace_info.get('answer')}")

    trajectories = extract_trajectories_from_traces(all_traces)

    synthesis_tokens = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    if not trajectories:
        synthesized_answer = "null"
    else:
        synthesized_answer, synthesis_tokens = synthesize_answer_with_llm(trajectories, question_text)

    gsm8k_env = get_gsm8k_env(split=split)
    inner_env = gsm8k_env
    while hasattr(inner_env, "env") and not hasattr(inner_env, "get_metrics"):
        inner_env = inner_env.env

    metrics = inner_env.get_metrics({"answer": synthesized_answer}) if hasattr(inner_env, "get_metrics") else {
        "em": 0,
        "f1": 0,
        "reward": 0,
    }

    llm_eval = llm_judge_answer(question_text, synthesized_answer, gt_answer)

    total_calls = sum(t.get("n_calls", 0) for t in all_traces)
    total_badcalls = sum(t.get("n_badcalls", 0) for t in all_traces)
    total_input_tokens = sum(t.get("input_tokens", 0) for t in all_traces)
    total_output_tokens = sum(t.get("output_tokens", 0) for t in all_traces)

    total_input_tokens += synthesis_tokens["input_tokens"] + llm_eval.get("judge_input_tokens", 0)
    total_output_tokens += synthesis_tokens["output_tokens"] + llm_eval.get("judge_output_tokens", 0)

    trace_summaries = []
    for i, trace in enumerate(all_traces):
        trace_summaries.append(
            {
                "trace_num": i + 1,
                "answer": trace.get("answer"),
                "em": trace.get("em", 0.0),
                "n_calls": trace.get("n_calls", 0),
                "input_tokens": trace.get("input_tokens", 0),
                "output_tokens": trace.get("output_tokens", 0),
            }
        )

    info_dict = {
        "question_idx": idx,
        "question_text": question_text,
        "answer": synthesized_answer,
        "gt_answer": gt_answer,
        "em": metrics.get("em", 0.0),
        "f1": metrics.get("f1", 0.0),
        "reward": metrics.get("reward", 0.0),
        "n_calls": total_calls,
        "n_badcalls": total_badcalls,
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
        "total_tokens": total_input_tokens + total_output_tokens,
        "num_traces_run": num_traces,
        "trace_summaries": trace_summaries,
        "full_traces": all_traces,
        "llm_correct": llm_eval["llm_correct"],
        "llm_explanation": llm_eval["llm_explanation"],
        "framework": "cot_sc",
    }

    if to_print:
        print("=" * 60)
        print(
            f"[FINAL] Synthesized Answer: {synthesized_answer} | GT: {gt_answer} | "
            f"EM: {info_dict['em']} | LLM: {info_dict['llm_correct']}"
        )
        print("=" * 60)

    framework_folder = get_framework_log_folder("cot_sc")
    run_name = get_next_run_number(framework_folder)
    run_folder = os.path.join(framework_folder, run_name)
    os.makedirs(run_folder, exist_ok=True)
    append_to_json(info_dict, os.path.join(run_folder, "results.jsonl"))

    return info_dict["reward"], info_dict


if __name__ == "__main__":
    print("\n[TEST] Running CoT-SC agent test\n")
    reward, info = run_cot_sc(idx=0, to_print=True, num_traces=3, split="test")
    print(f"\n[TEST RESULT] Reward: {reward}, Answer: {info['answer']}")
