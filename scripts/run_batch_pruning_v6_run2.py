
import sys
import os
import json
import csv

# Add paths
sys.path.append(os.path.join(os.getcwd(), 'src/agents/fever'))
sys.path.append(os.path.join(os.getcwd(), 'src/agents/fever/prog_CA_pruning'))

from prog_ca_pruning_agent import run_prog_ca_pruning_react
from fever_utils import append_to_json

def run_batch():
    # Task IDs provided by user
    task_ids = [4373, 4411, 7752, 2078, 7777, 7398, 2021, 8755, 4242, 5880]
    
    # Setup output directory
    output_dir = 'results/fever/prog_CA_pruning/react_sample_10_tuned_v6_run2'
    os.makedirs(output_dir, exist_ok=True)
    
    json_path = os.path.join(output_dir, 'prog_ca_pruning_results.json')
    csv_path = os.path.join(output_dir, 'prog_ca_pruning_results.csv')
    
    # Clear existing file if any (fresh run)
    if os.path.exists(json_path):
        os.remove(json_path)
    
    print(f"Running {len(task_ids)} tasks with Base Prompt + Hard Pruning...")
    print(f"Output: {json_path}")
    
    results = []
    
    for i, idx in enumerate(task_ids):
        print(f"Processing {i+1}/{len(task_ids)}...")
        try:
            print(f"\nRunning Task ID: {idx}")
            reward, info = run_prog_ca_pruning_react(idx=idx, to_print=True)
            
            # Ensure framework name is set
            info['framework'] = 'prog_CA_pruning'
            
            # Save specific result fields
            
            # Extract metrics
            context_pruned = info.get('pruning', {}).get('context_pruning', {}).get('pruned_sentences_count', 0)
            action_pruned = info.get('pruning', {}).get('action_pruning', {}).get('total_pruned', 0)
            
            result_entry = {
                'question_idx': info.get('question_idx'),
                'question_text': info.get('question_text'),
                'gt_answer': info.get('gt_answer'),
                'answer': info.get('answer'),
                'em': info.get('em'),
                'reward': info.get('reward'),
                'n_calls': info.get('n_calls'),
                'n_badcalls': info.get('n_badcalls'),
                'total_tokens': info.get('total_tokens'),
                'input_tokens': info.get('input_tokens'),
                'output_tokens': info.get('output_tokens'),
                'traj': info.get('traj'),
                'action_pruned_happen': action_pruned,
                'context_pruned_happen': context_pruned,
                'total_pruning': action_pruned + context_pruned
            }
            
            results.append(result_entry)
            append_to_json(result_entry, json_path)
            
        except Exception as e:
            print(f"Error running task {idx}: {e}")
            import traceback
            traceback.print_exc()
    
    # Convert to CSV
    print(f"\nConverting to CSV: {csv_path}")
    if results:
        fieldnames = [
        'question_idx', 'question_text', 'answer', 'gt_answer', 'em', 'f1', 
        'n_calls', 'n_badcalls', 'input_tokens', 'output_tokens', 'total_tokens', 
        'action_pruned_happen', 'context_pruned_happen', 'total_pruning', 'visited_pages'
        ]
        
        with open(csv_path, 'w', newline='') as output_file:
            dict_writer = csv.DictWriter(output_file, fieldnames=fieldnames)
            dict_writer.writeheader()
            
            for entry in results:
                # Re-map sparse data to full schema
                row = {
                    'question_idx': entry.get('question_idx'),
                    'question_text': entry.get('question_text'),
                    'answer': entry.get('answer'),
                    'gt_answer': entry.get('gt_answer'),
                    'em': entry.get('em'),
                    'f1': entry.get('em'),
                    'n_calls': entry.get('n_calls'),
                    'n_badcalls': entry.get('n_badcalls'),
                    'input_tokens': entry.get('input_tokens'),
                    'output_tokens': entry.get('output_tokens'),
                    'total_tokens': entry.get('total_tokens'),
                    'action_pruned_happen': entry.get('action_pruned_happen', 0),
                    'context_pruned_happen': entry.get('context_pruned_happen', 0),
                    'total_pruning': entry.get('total_pruning', 0),
                    'visited_pages': 0 # Basic fill
                }
                dict_writer.writerow(row)
                
    print("Done!")

if __name__ == "__main__":
    run_batch()
