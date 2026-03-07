"""
Experiment runner for Programmatic Combined Action & Context Pruning.

Runs the agent on FEVER dev set with comprehensive logging and result tracking.
Matches the logging and result format of other FEVER experiment runners.
"""

import random
import json
import os
import sys
from datetime import datetime
import traceback

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from prog_ca_pruning_agent import run_prog_ca_pruning_react
except ImportError as e:
    print(f"[ERROR] Failed to import prog_ca_pruning_agent: {e}")
    sys.exit(1)


# Configuration
MAX_FEVER_DEV_EXAMPLES = 7405
OUTPUT_FOLDER = './results/fever/gemini/prog_CA_pruning'
LOG_FILE = os.path.join(OUTPUT_FOLDER, 'prog_CA_pruning_experiment.log')
RESULTS_FILE = os.path.join(OUTPUT_FOLDER, 'prog_CA_pruning_results.json')


def setup_output_folder():
    """Create output folder if it doesn't exist."""
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def log_message(message: str, level: str = "INFO", to_file: bool = True):
    """
    Log a message with timestamp.
    
    Args:
        message: Message to log
        level: Log level (INFO, WARNING, ERROR, etc.)
        to_file: Whether to write to log file
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted = f"[{timestamp}] [{level}] {message}"
    
    print(formatted)
    
    if to_file:
        try:
            with open(LOG_FILE, 'a') as f:
                f.write(formatted + '\n')
        except Exception as e:
            print(f"[WARNING] Could not write to log file: {e}")


def get_processed_indices():
    """Get set of indices already processed."""
    processed = set()
    
    if os.path.exists(RESULTS_FILE):
        try:
            with open(RESULTS_FILE, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    for entry in data:
                        idx = entry.get('question_idx')
                        if idx is not None:
                            processed.add(idx)
        except Exception as e:
            log_message(f"Could not read processed indices: {e}", level="WARNING")
    
    return processed


def save_result(result: dict):
    """
    Append result to results file.
    
    Args:
        result: Result dictionary to save
    """
    # Simplify trajectory for storage (keep first 1000 chars)
    if 'traj' in result and len(result['traj']) > 2000:
        result['traj_summary'] = result['traj'][:1000] + "...[TRUNCATED]"
        del result['traj']
    
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, 'r+') as f:
            try:
                data = json.load(f)
            except:
                data = []
            
            if isinstance(data, list):
                data.append(result)
            else:
                data = [result]
            
            f.seek(0)
            json.dump(data, f, indent=2)
            f.truncate()
    else:
        with open(RESULTS_FILE, 'w') as f:
            json.dump([result], f, indent=2)


def run_experiments(num_examples: int = 10, start_from: int = 0):
    """
    Run experiments on FEVER examples.
    
    Args:
        num_examples: Number of examples to run
        start_from: Which processed set to start from (0 = beginning)
    """
    setup_output_folder()
    
    log_message("="*80, to_file=False)
    log_message("PROG-CA-PRUNING EXPERIMENT STARTED", level="INFO")
    log_message("="*80, to_file=False)
    log_message(f"Max examples: {num_examples}", level="INFO")
    log_message(f"Output folder: {OUTPUT_FOLDER}", level="INFO")
    
    # Get processed indices
    processed_indices = get_processed_indices()
    log_message(f"Already processed: {len(processed_indices)} examples", level="INFO")
    
    # Get remaining indices
    all_indices = list(range(MAX_FEVER_DEV_EXAMPLES))
    random.Random(42).shuffle(all_indices)
    remaining_indices = [idx for idx in all_indices if idx not in processed_indices]
    
    indices_to_run = remaining_indices[:num_examples]
    log_message(f"Will process: {len(indices_to_run)} new examples", level="INFO")
    log_message("-"*80, to_file=False)
    
    # Run experiments
    successful = 0
    failed = 0
    
    for run_num, idx in enumerate(indices_to_run, 1):
        log_message(f"\n[RUN {run_num}/{len(indices_to_run)}] Processing example {idx}", level="INFO")
        
        try:
            # Run agent
            log_message(f"Executing prog_CA_pruning agent...", level="INFO")
            reward, info = run_prog_ca_pruning_react(idx=idx, to_print=False)
            
            # Log result
            log_message(f"[RESULT] Answer: {info.get('answer', '?')} | GT: {info.get('gt_answer', '?')} | EM: {info.get('em', 0)}", level="INFO")
            log_message(f"[EFFICIENCY] Tokens: {info.get('total_tokens', 0)} | Calls: {info.get('n_calls', 0)}", level="INFO")
            
            # Log pruning stats
            if 'pruning' in info:
                pruning = info['pruning']
                action_pruned = pruning.get('action_pruning', {}).get('total_pruned', 0)
                evidence_items = pruning.get('context_pruning', {}).get('evidence_items', 0)
                log_message(f"[PRUNING] Actions pruned: {action_pruned} | Evidence items: {evidence_items}", level="INFO")
            
            # Save result
            save_result(info)
            successful += 1
            
        except Exception as e:
            log_message(f"[ERROR] Failed to process example {idx}: {e}", level="ERROR")
            log_message(f"[DEBUG] {traceback.format_exc()}", level="ERROR")
            
            # Save error result
            error_info = {
                'question_idx': idx,
                'error': str(e),
                'em': 0.0,
                'framework': 'prog_CA_pruning',
            }
            save_result(error_info)
            failed += 1
    
    # Print summary
    log_message("-"*80, to_file=False)
    log_message("EXPERIMENT SUMMARY", level="INFO")
    log_message("-"*80, to_file=False)
    log_message(f"Successful: {successful}", level="INFO")
    log_message(f"Failed: {failed}", level="INFO")
    log_message(f"Total: {successful + failed}", level="INFO")
    
    # Compute aggregate stats
    processed_indices = get_processed_indices()
    
    if processed_indices:
        try:
            with open(RESULTS_FILE, 'r') as f:
                all_results = json.load(f)
            
            em_scores = [r.get('em', 0.0) for r in all_results if 'em' in r and 'error' not in r]
            total_tokens = sum(r.get('total_tokens', 0) for r in all_results if 'total_tokens' in r)
            
            if em_scores:
                avg_em = sum(em_scores) / len(em_scores)
                log_message(f"Average EM: {avg_em:.2%}", level="INFO")
            
            if len(all_results) > 0:
                avg_tokens = total_tokens / len(all_results)
                log_message(f"Average tokens per example: {avg_tokens:.0f}", level="INFO")
        
        except Exception as e:
            log_message(f"Could not compute aggregate stats: {e}", level="WARNING")
    
    log_message("="*80, to_file=False)
    log_message("EXPERIMENT COMPLETED", level="INFO")
    log_message("="*80, to_file=False)
    log_message(f"Results saved to: {RESULTS_FILE}", level="INFO")
    log_message(f"Log saved to: {LOG_FILE}", level="INFO")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run prog_CA_pruning experiments on FEVER')
    parser.add_argument('--num', type=int, default=5, help='Number of examples to process')
    parser.add_argument('--start', type=int, default=0, help='Start from index')
    
    args = parser.parse_args()
    
    run_experiments(num_examples=args.num, start_from=args.start)
