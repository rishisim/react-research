"""
Run Reflexion Pruning Experiment on 10 specific tasks.

Tasks: 4373, 4411, 7752, 2078, 7777, 7398, 2021, 8755, 4242, 5880
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

from src.agents.fever.reflexion_pruning_agent import run_reflexion_pruning

def run_experiment():
    task_ids = [4373, 4411, 7752, 2078, 7777, 7398, 2021, 8755, 4242, 5880]
    
    results = []
    
    print(f"Starting Reflexion Pruning Experiment on {len(task_ids)} tasks...")
    
    # Define output directory
    output_dir = os.path.join(root_dir, 'results/fever/reflexion_pruning/experiment_10_tasks')
    os.makedirs(output_dir, exist_ok=True)
    
    json_path = os.path.join(output_dir, 'results.json')
    csv_path = os.path.join(output_dir, 'results.csv')
    
    for i, idx in enumerate(task_ids):
        print(f"\n[{i+1}/{len(task_ids)}] Running task {idx}...")
        try:
            # Run the agent
            em_score, info = run_reflexion_pruning(idx=idx, to_print=True, max_trials=7)
            
            # Add to results
            results.append(info)
            
            # Incremental save to JSON
            with open(json_path, 'w') as f:
                json.dump(results, f, indent=4)
                
            # Incremental save to CSV (simplified)
            df = pd.DataFrame(results)
            # Flatten some fields if needed
            df.to_csv(csv_path, index=False)
            
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
    
    print(f"\nExperiment failed? No. completed.")
    print(f"Results saved to {output_dir}")
    
    # helper stats
    success_count = sum(1 for r in results if r.get('em') == 1.0)
    print(f"Accuracy: {success_count}/{len(task_ids)} = {success_count/len(task_ids):.2f}")

if __name__ == '__main__':
    run_experiment()
