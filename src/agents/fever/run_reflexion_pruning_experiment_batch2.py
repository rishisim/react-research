"""
Run Reflexion Pruning Experiment on Batch 2 (10 additional tasks).

Tasks: 4795, 9500, 344, 2194, 3125, 7350, 4187, 1564, 8501, 7826
"""

import sys
import os
import json
import time

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '../../../'))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from src.agents.fever.reflexion_pruning_agent import run_reflexion_pruning

def run_experiment_batch2():
    new_task_ids = [4795, 9500, 344, 2194, 3125, 7350, 4187, 1564, 8501, 7826]
    
    # Define output directory
    output_dir = os.path.join(root_dir, 'results/fever/gemini/reflexion_pruning/experiment_10_tasks')
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, 'results.json')
    
    results = []
    # Load existing results if available
    if os.path.exists(json_path):
        print(f"Loading existing results from {json_path}")
        with open(json_path, 'r') as f:
            results = json.load(f)
            
    existing_ids = {r.get('question_idx') for r in results}
    print(f"Loaded {len(results)} existing records.")
    
    print(f"Starting Reflexion Pruning Experiment Batch 2 on {len(new_task_ids)} tasks...")
    
    for i, idx in enumerate(new_task_ids):
        if idx in existing_ids:
            print(f"Skipping task {idx} (already exists)")
            continue
            
        print(f"\n[{i+1}/{len(new_task_ids)}] Running task {idx}...")
        try:
            # Run the agent
            em_score, info = run_reflexion_pruning(idx=idx, to_print=True, max_trials=7)
            
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
    
    print(f"\nBatch 2 completed.")
    print(f"Total results: {len(results)}")
    
    # helper stats
    success_count = sum(1 for r in results if r.get('em') == 1.0)
    print(f"Total Accuracy: {success_count}/{len(results)} = {success_count/len(results):.2f}")

if __name__ == '__main__':
    run_experiment_batch2()
