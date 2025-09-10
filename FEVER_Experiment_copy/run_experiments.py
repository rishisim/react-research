import random
import json
import os
import sys

# Ensure the FEVER_Experiment directory is in the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    import fever_agent as fa
except ImportError as e:
    print(f"Error importing fever_agent: {e}")
    print("Make sure fever_agent.py is in the FEVER_Experiment directory.")
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred during fever_agent import: {e}")
    sys.exit(1)


# --- Main Configuration ---
MAX_FEVER_DEV_EXAMPLES = 7405 # Based on paper_dev.jsonl line count
NUM_TASKS_TODAY = 20 # Keep small for testing, can be increased later
BASELINE_OUTPUT_FILE = 'react_baseline_results.json' # Path relative to FEVER_Experiment
NEW_FRAMEWORK_OUTPUT_FILE = 'react_multi_trace_results.json' # Path relative to FEVER_Experiment
BASELINE_OUTPUT_FILE_PATH = os.path.join(os.path.dirname(__file__), BASELINE_OUTPUT_FILE)
NEW_FRAMEWORK_OUTPUT_FILE_PATH = os.path.join(os.path.dirname(__file__), NEW_FRAMEWORK_OUTPUT_FILE)


print("Setting up dataset indices for FEVER...")

def get_processed_indices(output_file_path):
    processed_indices = set()
    if os.path.exists(output_file_path):
        try:
            with open(output_file_path, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                    if isinstance(data, list):
                        for entry in data:
                            idx = entry.get('question_idx')
                            if idx is not None:
                                processed_indices.add(idx)
                    elif isinstance(data, dict):
                        idx = data.get('question_idx')
                        if idx is not None:
                            processed_indices.add(idx)
                except Exception as e:
                    print(f"Warning: Could not parse {output_file_path} as JSON array: {e}")
        except Exception as e:
            print(f"Warning: Could not read {output_file_path}: {e}")
    return processed_indices

# Collect all processed indices from both output files
processed_indices = get_processed_indices(BASELINE_OUTPUT_FILE_PATH) | get_processed_indices(NEW_FRAMEWORK_OUTPUT_FILE_PATH)

all_indices = list(range(MAX_FEVER_DEV_EXAMPLES))
random.Random(42).shuffle(all_indices)
# Exclude already processed indices
remaining_indices = [idx for idx in all_indices if idx not in processed_indices]
indices_for_today = remaining_indices[:NUM_TASKS_TODAY]
print(f"Prepared to run {len(indices_for_today)} new tasks on FEVER dev set (skipping {len(processed_indices)} already processed).")

# --- Main Execution Loop ---
for i, idx in enumerate(indices_for_today):
    print(f"--- Processing Task {i+1}/{NUM_TASKS_TODAY} (Index: {idx}) ---")

    # 1. Run Baseline ReAct (1 trace)
    print("Running baseline ReAct (num_traces=1)...")
    try:
        _, baseline_info = fa.webthink(idx=idx, initial_prompt_template=fa.WEBTHINK_PROMPT_TEMPLATE, to_print=False, num_traces=1)
        fa.append_to_json(baseline_info, BASELINE_OUTPUT_FILE_PATH)
        print(f"  > Baseline results saved to {BASELINE_OUTPUT_FILE_PATH}")
    except Exception as e:
        print(f"  ERROR during baseline ReAct for index {idx}: {e}")
        error_info = {'question_idx': idx, 'error': str(e), 'details': 'Baseline ReAct failed'}
        fa.append_to_json(error_info, BASELINE_OUTPUT_FILE_PATH)

    # 2. Run New Framework (3 traces)
    print("Running new framework (num_traces=3)...")
    try:
        synthesized_reward, multi_trace_info = fa.webthink(idx=idx, initial_prompt_template=fa.WEBTHINK_PROMPT_TEMPLATE, to_print=False, num_traces=3)
        # Compute metrics for synthesized answer using direct comparison (not get_metrics)
        synthesized_answer = multi_trace_info.get('answer') if isinstance(multi_trace_info, dict) else None
        ground_truth_answer = multi_trace_info.get('gt_answer') if isinstance(multi_trace_info, dict) else None
        if synthesized_answer is not None and ground_truth_answer is not None:
            em = int(synthesized_answer == ground_truth_answer)
            f1 = em
            reward = em
        else:
            em = f1 = reward = 0
        new_framework_result = {
            'question_idx': idx,
            'question_text': multi_trace_info.get('question_text') if isinstance(multi_trace_info, dict) else None,
            'synthesized_answer': synthesized_answer,
            'ground_truth_answer': ground_truth_answer,
            'em': em,
            'reward': reward,
            'f1': f1,
            'traces': multi_trace_info.get('individual_traces') if isinstance(multi_trace_info, dict) else multi_trace_info
        }
        fa.append_to_json(new_framework_result, NEW_FRAMEWORK_OUTPUT_FILE_PATH)
        print(f"  > New framework results saved to {NEW_FRAMEWORK_OUTPUT_FILE_PATH}")
    except Exception as e:
        print(f"  ERROR during new framework for index {idx}: {e}")
        error_info = {'question_idx': idx, 'error': str(e), 'details': 'New framework (3 traces) failed'}
        fa.append_to_json(error_info, NEW_FRAMEWORK_OUTPUT_FILE_PATH)
    print("-" * 20)

print("\nAll FEVER tasks for today completed!")
print(f"Baseline results are in: {BASELINE_OUTPUT_FILE_PATH}")
print(f"New framework results are in: {NEW_FRAMEWORK_OUTPUT_FILE_PATH}")
