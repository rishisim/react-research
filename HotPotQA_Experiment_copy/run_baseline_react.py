import random
import hotpotqa_agent as hqa
import os
import json

NUM_TASKS_TODAY = 20
BASELINE_OUTPUT_FILE = 'HotPotQA_Experiment/react_baseline_results.json'

def get_processed_indices(output_file_path):
    processed_indices = set()
    if os.path.exists(output_file_path):
        try:
            with open(output_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    for entry in data:
                        idx = entry.get('question_idx')
                        if idx is not None:
                            processed_indices.add(idx)
        except Exception as e:
            print(f"Warning: Could not read {output_file_path}: {e}")
    return processed_indices

print("Setting up dataset indices...")
all_indices = list(range(7405))
random.Random(42).shuffle(all_indices)

processed_indices = get_processed_indices(BASELINE_OUTPUT_FILE)
remaining_indices = [idx for idx in all_indices if idx not in processed_indices]
indices_for_today = remaining_indices[:NUM_TASKS_TODAY]
print(f"Prepared to run {len(indices_for_today)} new baseline tasks (skipping {len(processed_indices)} already processed).")

for i, idx in enumerate(indices_for_today):
    print(f"--- Processing Baseline Task {i+1}/{NUM_TASKS_TODAY} (Index: {idx}) ---")
    _, baseline_info = hqa.webthink(idx=idx, initial_prompt_template=hqa.WEBTHINK_PROMPT_TEMPLATE, to_print=False, num_traces=1)
    hqa.append_to_json(baseline_info, BASELINE_OUTPUT_FILE)
    print(f"  > Baseline results saved to {BASELINE_OUTPUT_FILE}")
    print("-" * 20)

print("\nAll baseline tasks for today completed!")
