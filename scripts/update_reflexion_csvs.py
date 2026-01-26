
import json
import pandas as pd
import os

def update_csvs():
    base_dir = "/Users/rishisim/Documents/research/react-research/results/fever/reflexion_pruning/experiment_10_tasks"
    json_path = os.path.join(base_dir, "results.json")
    results_csv_path = os.path.join(base_dir, "results.csv")
    baseline_csv_path = os.path.join(base_dir, "reflexion_baseline.csv")

    # 1. Update results.csv from results.json
    print(f"Loading {json_path}...")
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Create a mapping of question_idx -> pruning stats
    pruning_stats = {}
    for item in data:
        q_idx = item.get('question_idx')
        
        total_action_pruned = 0
        total_context_pruned = 0
        
        all_traces = item.get('all_traces', {})
        for trace_key, trace_data in all_traces.items():
            pruning = trace_data.get('pruning', {})
            
            # Action pruning
            action_pruning = pruning.get('action_pruning', {})
            total_action_pruned += action_pruning.get('total_pruned', 0)
            
            # Context pruning
            context_pruning = pruning.get('context_pruning', {})
            total_context_pruned += context_pruning.get('pruned_sentences_count', 0)
            
        pruning_stats[q_idx] = {
            'action_pruned_happen': total_action_pruned,
            'context_pruned_happen': total_context_pruned,
            'total_pruning': total_action_pruned + total_context_pruned
        }

    print(f"Updating {results_csv_path}...")
    df_results = pd.read_csv(results_csv_path, sep='|')
    
    # Add columns if they don't exist
    for col in ['action_pruned_happen', 'context_pruned_happen', 'total_pruning']:
        if col not in df_results.columns:
            df_results[col] = 0

    # Fill values
    for idx, row in df_results.iterrows():
        q_idx = row['question_idx']
        if q_idx in pruning_stats:
            stats = pruning_stats[q_idx]
            df_results.at[idx, 'action_pruned_happen'] = stats['action_pruned_happen']
            df_results.at[idx, 'context_pruned_happen'] = stats['context_pruned_happen']
            df_results.at[idx, 'total_pruning'] = stats['total_pruning']

    df_results.to_csv(results_csv_path, sep='|', index=False)
    print(f"Updated {results_csv_path}")

    # 2. Update reflexion_baseline.csv with 0s
    print(f"Updating {baseline_csv_path}...")
    if os.path.exists(baseline_csv_path):
        df_baseline = pd.read_csv(baseline_csv_path, sep='|')
        
        for col in ['action_pruned_happen', 'context_pruned_happen', 'total_pruning']:
            df_baseline[col] = 0
            
        df_baseline.to_csv(baseline_csv_path, sep='|', index=False)
        print(f"Updated {baseline_csv_path}")
    else:
        print(f"Warning: {baseline_csv_path} not found.")

if __name__ == "__main__":
    update_csvs()
