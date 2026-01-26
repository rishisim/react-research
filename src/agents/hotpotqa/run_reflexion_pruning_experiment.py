"""
Run Reflexion Pruning Experiment on 20 HotpotQA tasks.

Tasks: [1399, 4569, 3699, 5490, 6883, 1472, 6905, 1169, 3600, 1102, 2708, 1144, 6606, 6253, 3625, 4387, 2732, 2853, 3156, 3977]
"""

import sys
import os
import json
import pandas as pd
import time

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '../../../'))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from src.agents.hotpotqa.reflexion_pruning_agent import run_reflexion_pruning

def run_experiment():
    # Reduced to 10 tasks for speed
    task_ids = [1399, 4569, 3699, 5490, 6883, 1472, 6905, 1169, 3600, 1102]
    
    results = []
    
    print(f"Starting HotpotQA Reflexion Pruning Experiment on {len(task_ids)} tasks (optimization: max_trials=4)...")
    
    # Define output directory
    output_dir = os.path.join(root_dir, 'results/hotpotqa/reflexion_pruning/experiment_10_tasks')
    os.makedirs(output_dir, exist_ok=True)
    
    json_path = os.path.join(output_dir, 'results.json')
    csv_path = os.path.join(output_dir, 'results.csv')
    
    for i, idx in enumerate(task_ids):
        print(f"\n[{i+1}/{len(task_ids)}] Running task {idx}...")
        try:
            # Run the agent with REDUCED MAX_TRIALS for speed
            em_score, info = run_reflexion_pruning(idx=idx, to_print=True, max_trials=4)
            
            # Add to results
            results.append(info)
            
            # Incremental save to JSON
            with open(json_path, 'w') as f:
                json.dump(results, f, indent=4)
                
            # Incremental save to CSV (simplified)
            df = pd.DataFrame(results)
            # Flatten/filter columns similar to what we did for FEVER
            cols = [
                'question_idx', 'question_text', 'answer', 'gt_answer', 'em', 'f1', 'reward',
                'n_calls', 'n_badcalls', 'input_tokens', 'output_tokens', 'total_tokens',
                'num_trials', 'max_trials', 'framework'
            ]
            existing_cols = [c for c in cols if c in df.columns]
            df = df[existing_cols]
            # Use pipe delimiter
            df.to_csv(csv_path, index=False, sep='|')
            
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
    
    # helper stats
    success_count = sum(1 for r in results if r.get('reward') == 1.0)
    print(f"Accuracy (Reward): {success_count}/{len(task_ids)} = {success_count/len(task_ids):.2f}")

if __name__ == '__main__':
    run_experiment()
