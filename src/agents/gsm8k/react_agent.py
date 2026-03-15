"""ReAct Agent for GSM8K."""

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from gsm8k_utils import run_single_trace, WEBTHINK_PROMPT_TEMPLATE


def run_react(idx, prompt_template=None, to_print=True, split=None):
    if prompt_template is None:
        prompt_template = WEBTHINK_PROMPT_TEMPLATE

    if to_print:
        print("=" * 60)
        print("[FRAMEWORK] Standard ReAct (Single Trace)")
        print("=" * 60)

    trace_info = run_single_trace(
        idx=idx,
        initial_prompt_template=prompt_template,
        to_print=to_print,
        temperature=0.0,
        split=split,
    )

    reward = trace_info.get("em", 0.0)
    info_dict = {
        "question_idx": trace_info.get("question_idx"),
        "question_text": trace_info.get("question_text"),
        "answer": trace_info.get("answer"),
        "gt_answer": trace_info.get("gt_answer"),
        "em": trace_info.get("em", 0.0),
        "f1": trace_info.get("f1", 0.0),
        "reward": reward,
        "n_calls": trace_info.get("n_calls", 0),
        "n_badcalls": trace_info.get("n_badcalls", 0),
        "input_tokens": trace_info.get("input_tokens", 0),
        "output_tokens": trace_info.get("output_tokens", 0),
        "total_tokens": trace_info.get("total_tokens", 0),
        "traj": trace_info.get("traj", ""),
        "llm_correct": trace_info.get("llm_correct", False),
        "llm_explanation": trace_info.get("llm_explanation", ""),
        "framework": "react",
    }

    if to_print:
        print("=" * 60)
        print(
            f"[FINAL] Answer: {info_dict['answer']} | GT: {info_dict['gt_answer']} | "
            f"EM: {info_dict['em']} | LLM: {info_dict['llm_correct']}"
        )
        print("=" * 60)

    return reward, info_dict


if __name__ == "__main__":
    print("\n[TEST] Running ReAct agent test\n")
    reward, info = run_react(idx=0, to_print=True, split="test")
    print(f"\n[TEST RESULT] Reward: {reward}, Answer: {info['answer']}")
