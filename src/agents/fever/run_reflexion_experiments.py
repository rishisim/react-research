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
NUM_TASKS_TODAY = 3  # Configurable limit for daily API usage - change this as needed
BASELINE_OUTPUT_FILE = 'react_baseline_results.json'
MULTI_TRACE_OUTPUT_FILE = 'react_multi_trace_results.json'
REFLEXION_OUTPUT_FILE = 'react_multi_trace_reflexion_results.json'

# Full paths
# Full paths
# Define paths relative to this script
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REACT_RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results', 'fever', 'react')
REFLEXION_RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results', 'fever', 'reflexion')
os.makedirs(REFLEXION_RESULTS_DIR, exist_ok=True)

BASELINE_OUTPUT_FILE_PATH = os.path.join(REACT_RESULTS_DIR, BASELINE_OUTPUT_FILE)
MULTI_TRACE_OUTPUT_FILE_PATH = os.path.join(REACT_RESULTS_DIR, MULTI_TRACE_OUTPUT_FILE)
REFLEXION_OUTPUT_FILE_PATH = os.path.join(REFLEXION_RESULTS_DIR, REFLEXION_OUTPUT_FILE)

print("Setting up reflexion experiments for existing FEVER questions...")


def extract_question_indices_from_files():
    """Extract all question indices from baseline and multi-trace results in the order they appear"""
    indices_with_source = []
    
    print("Extracting question indices from existing result files...")
    
    # Extract from baseline results
    if os.path.exists(BASELINE_OUTPUT_FILE_PATH):
        try:
            with open(BASELINE_OUTPUT_FILE_PATH, 'r', encoding='utf-8') as f:
                baseline_data = json.load(f)
                if isinstance(baseline_data, list):
                    for entry in baseline_data:
                        idx = entry.get('question_idx')
                        if idx is not None and isinstance(idx, int):
                            indices_with_source.append((idx, 'baseline'))
                            
            print(f"Found {len([x for x in indices_with_source if x[1] == 'baseline'])} questions from baseline results")
        except Exception as e:
            print(f"Warning: Could not read baseline results: {e}")
    
    # Extract from multi-trace results
    if os.path.exists(MULTI_TRACE_OUTPUT_FILE_PATH):
        try:
            with open(MULTI_TRACE_OUTPUT_FILE_PATH, 'r', encoding='utf-8') as f:
                multi_trace_data = json.load(f)
                if isinstance(multi_trace_data, list):
                    for entry in multi_trace_data:
                        idx = entry.get('question_idx')
                        if idx is not None and isinstance(idx, int):
                            # Only add if not already in the list (to maintain order but avoid duplicates)
                            if idx not in [x[0] for x in indices_with_source]:
                                indices_with_source.append((idx, 'multi_trace'))
                                
            print(f"Found {len([x for x in indices_with_source if x[1] == 'multi_trace'])} additional questions from multi-trace results")
        except Exception as e:
            print(f"Warning: Could not read multi-trace results: {e}")
    
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
                    print(f"Warning: Could not parse reflexion results as JSON: {e}")
        except Exception as e:
            print(f"Warning: Could not read reflexion results: {e}")
    
    return processed_indices


def clear_reflexion_results_file():
    try:
        with open(REFLEXION_OUTPUT_FILE_PATH, 'w', encoding='utf-8') as f:
            json.dump([], f, indent=4)
        print(f"Cleared {REFLEXION_OUTPUT_FILE}")
    except Exception as e:
        print(f"Warning: Could not clear reflexion results file: {e}")


def run_reflexion_experiments():
    """Run reflexion experiments on existing questions to catch up results count."""
    # Do not clear by default; allow opt-in via env
    if os.getenv("CLEAR_REFLEXION_RESULTS", "0") == "1":
        clear_reflexion_results_file()
    
    # Get all question indices from existing results
    all_question_indices = extract_question_indices_from_files()
    
    if not all_question_indices:
        print("No questions found in existing result files. Exiting.")
        return
    
    # Get already processed reflexion indices (should be empty after clearing, but kept for safety)
    processed_reflexion_indices = get_processed_reflexion_indices()
    
    # Filter out already processed questions
    remaining_indices = [idx for idx in all_question_indices if idx not in processed_reflexion_indices]
    
    # Limit to NUM_TASKS_TODAY for API limits
    indices_for_today = remaining_indices[:NUM_TASKS_TODAY]
    
    print(f"Prepared to run reflexion on {len(indices_for_today)} questions (out of {len(remaining_indices)} remaining)")
    print(f"Processing indices: {indices_for_today}")
    
    if not indices_for_today:
        print("No new questions to process. All questions may already be completed.")
        return
    
    # --- Main Execution Loop ---
    for i, idx in enumerate(indices_for_today):
        print(f"--- Processing Reflexion Task {i+1}/{len(indices_for_today)} (Index: {idx}) ---")
        
        try:
            # Note: Based on the existing reflexion results, there should be a reflexion function
            # We'll use the same webthink function but need to determine how reflexion is triggered
            # Looking at the results, it seems like reflexion might be handled through a special prompt or parameter
            
            # First, let's try to use webthink with a reflexion-specific approach
            # Since the existing reflexion results show a different structure, we may need to call a specific function
            
            # For now, let's use the standard webthink with multi-trace (3 traces) 
            # and see if we can adapt it for reflexion functionality
            print("Running sequential multi-trace reflexion (1→R1→2→R2→3)...")
            reflexion_reward, reflexion_info = fa.webthink_reflexion_seq(
                idx=idx,
                to_print=False
            )
            
            # Structure the result to match the reflexion format we see in existing results
            reflexion_result = {
                'question_idx': idx,
                'question_text': reflexion_info.get('question_text') if isinstance(reflexion_info, dict) else None,
                'answer': reflexion_info.get('answer') if isinstance(reflexion_info, dict) else None,
                'gt_answer': reflexion_info.get('gt_answer') if isinstance(reflexion_info, dict) else None,
                'em': reflexion_info.get('em') if isinstance(reflexion_info, dict) else 0,
                'f1': reflexion_info.get('f1') if isinstance(reflexion_info, dict) else 0,
                'reward': reflexion_reward if reflexion_reward is not None else 0,
                'n_calls': reflexion_info.get('n_calls') if isinstance(reflexion_info, dict) else 0,
                'n_badcalls': reflexion_info.get('n_badcalls') if isinstance(reflexion_info, dict) else 0,
                'num_traces_run': reflexion_info.get('num_traces_run') if isinstance(reflexion_info, dict) else 3,
                'individual_trajectories': reflexion_info.get('individual_trajectories') if isinstance(reflexion_info, dict) else [],
                'framework': reflexion_info.get('framework', 'multi_trace_reflexion') if isinstance(reflexion_info, dict) else 'multi_trace_reflexion'
            }
            
            if isinstance(reflexion_info, dict) and 'reflexions' in reflexion_info:
                reflexion_result['reflexions'] = reflexion_info['reflexions']
            
            fa.append_to_json(reflexion_result, REFLEXION_OUTPUT_FILE_PATH)
            print(f"  > Reflexion results saved for question {idx}")
            
        except Exception as e:
            print(f"  ERROR during reflexion for index {idx}: {e}")
            error_info = {
                'question_idx': idx, 
                'error': str(e), 
                'details': 'Multi-trace reflexion framework failed',
                'framework': 'multi_trace_reflexion'
            }
            fa.append_to_json(error_info, REFLEXION_OUTPUT_FILE_PATH)
        
        print("-" * 50)
    
    print(f"\nReflexion experiments completed!")
    print(f"Results saved to: {REFLEXION_OUTPUT_FILE_PATH}")
    print(f"Processed {len(indices_for_today)} questions out of {len(remaining_indices)} remaining")
    
    if len(remaining_indices) > NUM_TASKS_TODAY:
        print(f"Note: {len(remaining_indices) - NUM_TASKS_TODAY} questions still remaining for future runs")
        print(f"To process more questions, increase NUM_TASKS_TODAY (currently {NUM_TASKS_TODAY}) and run again")


if __name__ == "__main__":
    print(f"=== FEVER Reflexion Experiment Runner ===")
    print(f"API Limit: {NUM_TASKS_TODAY} questions per run")
    print(f"Baseline file: {BASELINE_OUTPUT_FILE}")
    print(f"Multi-trace file: {MULTI_TRACE_OUTPUT_FILE}")
    print(f"Reflexion output: {REFLEXION_OUTPUT_FILE}")
    print("=" * 50)
    
    run_reflexion_experiments()
