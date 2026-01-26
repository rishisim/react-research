"""
Run Reflexion Pruning Experiment on 10 NEW HotpotQA tasks (from ReAct baseline).

New Tasks: [1888, 3545, 2587, 3726, 3469, 7385, 1873, 672, 6679, 3619]
"""

import sys
import os
import json
import pandas as pd

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '../../../'))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from src.agents.hotpotqa.reflexion_pruning_agent import run_reflexion_pruning

def run_experiment():
    # 10 NEW task IDs from ReAct baseline logs
    new_task_ids = [1888, 3545, 2587, 3726, 3469, 7385, 1873, 672, 6679, 3619]
    
    # Output directory (same as existing experiment)
    output_dir = os.path.join(root_dir, 'results/hotpotqa/reflexion_pruning/experiment_10_tasks')
    os.makedirs(output_dir, exist_ok=True)
    
    json_path = os.path.join(output_dir, 'results.json')
    
    # Load existing results
    existing_results = []
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            existing_results = json.load(f)
    
    # Get already processed indices
    processed_indices = {r['question_idx'] for r in existing_results if 'question_idx' in r}
    
    print(f"Starting HotpotQA Reflexion Pruning Experiment on {len(new_task_ids)} NEW tasks...")
    print(f"Already processed: {processed_indices}")
    
    results = existing_results.copy()
    
    for i, idx in enumerate(new_task_ids):
        if idx in processed_indices:
            print(f"[{i+1}/{len(new_task_ids)}] Task {idx} already processed, skipping...")
            continue
            
        print(f"\n[{i+1}/{len(new_task_ids)}] Running task {idx}...")
        try:
            # Run the agent with max_trials=4 for speed
            em_score, info = run_reflexion_pruning(idx=idx, to_print=True, max_trials=4)
            
            # Add to results
            results.append(info)
            
            # Incremental save to JSON
            with open(json_path, 'w') as f:
                json.dump(results, f, indent=4)
                
        except Exception as e:
            print(f"[ERROR] Failed on task {idx}: {e}")
            import traceback
            traceback.print_exc()
            
            # Log error
            error_info = {
                'question_idx': idx,
                'error': str(e),
                'status': 'failed'
            }
            results.append(error_info)
            with open(json_path, 'w') as f:
                json.dump(results, f, indent=4)
    
    print(f"\nExperiment completed.")
    print(f"Results saved to {output_dir}")
    
    # Final stats
    valid_results = [r for r in results if 'em' in r]
    success_count = sum(1 for r in valid_results if r.get('reward') == 1.0)
    print(f"Total valid results: {len(valid_results)}")
    print(f"Accuracy (Reward): {success_count}/{len(valid_results)} = {success_count/len(valid_results):.2f}")

if __name__ == '__main__':
    run_experiment()
