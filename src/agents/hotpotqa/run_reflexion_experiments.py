import os
import sys
import json

# Ensure the HotPotQA_Experiment directory is in the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    import hotpotqa_agent as hqa
except ImportError as e:
    print(f"Error importing hotpotqa_agent: {e}")
    print("Make sure hotpotqa_agent.py is in the HotPotQA_Experiment directory.")
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred during hotpotqa_agent import: {e}")
    sys.exit(1)

# --- Main Configuration ---
NUM_TASKS_TODAY = 3  # Change as needed
BASELINE_OUTPUT_FILE = 'react_baseline_results.json'
MULTI_TRACE_OUTPUT_FILE = 'react_cot_synth_results.json'
REFLEXION_OUTPUT_FILE = 'react_multi_trace_reflexion_results.json'

# Full paths
# Using absolute paths relative to the project root (assuming script is run from project root or paths are adjusted)
# Adjusting to use absolute paths based on the known structure or relative to the script location if possible, 
# but simply using relative paths from where the script is likely run (root) or navigating up.
# Given the user wants them in `results/reflexion/hotpotqa`, we'll construct paths relative to this script.
# This script is in src/agents/hotpotqa. results/ is in root/results.
# So we need to go up 3 levels: ../../../results/

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REACT_RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results', 'hotpotqa', 'react')
REFLEXION_RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results', 'hotpotqa', 'reflexion')

BASELINE_OUTPUT_FILE_PATH = os.path.join(REACT_RESULTS_DIR, BASELINE_OUTPUT_FILE)
MULTI_TRACE_OUTPUT_FILE_PATH = os.path.join(REACT_RESULTS_DIR, MULTI_TRACE_OUTPUT_FILE)
REFLEXION_OUTPUT_FILE_PATH = os.path.join(REFLEXION_RESULTS_DIR, REFLEXION_OUTPUT_FILE)

print("Setting up reflexion experiments for existing HotPotQA questions...")

def extract_question_indices_from_files():
    """Extract all question indices from baseline and multi-trace results in the order they appear"""
    indices_with_source = []

    print("Extracting question indices from existing result files...")

    # Extract from baseline results
    if os.path.exists(BASELINE_OUTPUT_FILE_PATH):
        try:
            with open(BASELINE_OUTPUT_FILE_PATH, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    for entry in data:
                        idx = entry.get('question_idx')
                        if isinstance(idx, int):
                            indices_with_source.append((idx, 'baseline'))
        except Exception as e:
            print(f"Warning: Could not read {BASELINE_OUTPUT_FILE_PATH}: {e}")

    # Extract from multi-trace results
    if os.path.exists(MULTI_TRACE_OUTPUT_FILE_PATH):
        try:
            with open(MULTI_TRACE_OUTPUT_FILE_PATH, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    existing = {i for i, _ in indices_with_source}
                    for entry in data:
                        idx = entry.get('question_idx')
                        if isinstance(idx, int) and idx not in existing:
                            indices_with_source.append((idx, 'multi_trace'))
        except Exception as e:
            print(f"Warning: Could not read {MULTI_TRACE_OUTPUT_FILE_PATH}: {e}")

    # Extract just the indices in order
    question_indices = [idx for idx, source in indices_with_source]

    print(f"Total unique questions found: {len(question_indices)}")
    return question_indices


def get_processed_reflexion_indices():
    """Get question indices that have already been processed with reflexion"""
    processed_indices = set()

    if os.path.exists(REFLEXION_OUTPUT_FILE_PATH):
        try:
            with open(REFLEXION_OUTPUT_FILE_PATH, 'r', encoding='utf-8') as f:
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
            print(f"Warning: Could not read {REFLEXION_OUTPUT_FILE_PATH}: {e}")

    return processed_indices


def clear_reflexion_results_file():
    try:
        with open(REFLEXION_OUTPUT_FILE_PATH, 'w', encoding='utf-8') as f:
            json.dump([], f, indent=4)
        print(f"Cleared {REFLEXION_OUTPUT_FILE_PATH}")
    except Exception as e:
        print(f"Warning: Could not clear {REFLEXION_OUTPUT_FILE_PATH}: {e}")


def run_reflexion_experiments():
    """Run reflexion experiments on existing questions to catch up results count."""
    # Do not clear by default; allow opt-in via env
    if os.getenv("CLEAR_REFLEXION_RESULTS", "0") == "1":
        clear_reflexion_results_file()

    # Get all question indices from existing results
    all_question_indices = extract_question_indices_from_files()

    if not all_question_indices:
        print("No questions found in existing results. Exiting.")
        return

    # Get already processed reflexion indices (should be empty after clearing, but kept for safety)
    processed_reflexion_indices = get_processed_reflexion_indices()

    remaining_indices = [idx for idx in all_question_indices if idx not in processed_reflexion_indices]
    indices_for_today = remaining_indices[:NUM_TASKS_TODAY]

    print(f"Prepared to run reflexion on {len(indices_for_today)} questions (out of {len(remaining_indices)} remaining)")
    print(f"Processing indices: {indices_for_today}")

    if not indices_for_today:
        print("No new questions to process.")
        return

    for i, idx in enumerate(indices_for_today):
        print(f"--- Processing Reflexion Task {i+1}/{len(indices_for_today)} (Index: {idx}) ---")
        try:
            reward, info = hqa.webthink_reflexion_seq(idx=idx, to_print=False)
            reflexion_result = {
                'framework': info.get('framework', 'multi_trace_reflexion'),
                'question_idx': idx,
                'question_text': info.get('question_text'),
                'answer': info.get('answer'),
                'gt_answer': info.get('gt_answer'),
                'em': info.get('em', 0),
                'f1': info.get('f1', 0),
                'reward': reward if reward is not None else info.get('reward', 0),
                'n_calls': info.get('n_calls', 0),
                'n_badcalls': info.get('n_badcalls', 0),
                'num_traces_run': info.get('num_traces_run', 3),
                'individual_trajectories': info.get('individual_trajectories', []),
                'reflexions': info.get('reflexions', []),
                'finish_action_obs': info.get('finish_action_obs')
            }
            hqa.append_to_json(reflexion_result, REFLEXION_OUTPUT_FILE_PATH)
            print(f"  > Reflexion results saved for question {idx}")
        except Exception as e:
            print(f"  ERROR during reflexion for index {idx}: {e}")
            error_info = {
                'framework': 'multi_trace_reflexion',
                'question_idx': idx,
                'error': str(e),
                'details': 'HotPotQA multi-trace reflexion framework failed'
            }
            hqa.append_to_json(error_info, REFLEXION_OUTPUT_FILE_PATH)

        print("-" * 50)

    print("\nReflexion experiments completed!")
    print(f"Results saved to: {REFLEXION_OUTPUT_FILE_PATH}")
    print(f"Processed {len(indices_for_today)} questions out of {len(remaining_indices)} remaining")
    if len(remaining_indices) > NUM_TASKS_TODAY:
        print(f"Note: {len(remaining_indices) - NUM_TASKS_TODAY} questions still remaining.")
        print(f"Increase NUM_TASKS_TODAY (currently {NUM_TASKS_TODAY}) to process more.")

if __name__ == "__main__":
    print("=== HotPotQA Reflexion Experiment Runner ===")
    print(f"API Limit: {NUM_TASKS_TODAY} questions per run")
    print(f"Baseline file: {BASELINE_OUTPUT_FILE}")
    print(f"Multi-trace file: {MULTI_TRACE_OUTPUT_FILE}")
    print(f"Reflexion output: {REFLEXION_OUTPUT_FILE}")
    print("=" * 50)
    run_reflexion_experiments()
