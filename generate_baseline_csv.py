import json
import pandas as pd
import os

# Define paths
base_dir = '/Users/rishisim/Documents/research/react-research'

def generate_baseline_csv(dataset_type='fever'):
    if dataset_type == 'fever':
         baseline_json = os.path.join(base_dir, 'results/fever/reflexion/seed42_gemini-2.5-flash/reflexion.json')
         results_json = os.path.join(base_dir, 'results/fever/reflexion_pruning/experiment_10_tasks/results.json')
         output_csv = os.path.join(base_dir, 'results/fever/reflexion_pruning/experiment_10_tasks/reflexion_baseline.csv')
    elif dataset_type == 'hotpotqa':
         baseline_json = os.path.join(base_dir, 'results/hotpotqa/reflexion/seed42_gemini-2.5-flash/reflexion.json')
         results_json = os.path.join(base_dir, 'results/hotpotqa/reflexion_pruning/experiment_10_tasks/results.json')
         output_csv = os.path.join(base_dir, 'results/hotpotqa/reflexion_pruning/experiment_10_tasks/reflexion_baseline.csv')
    else:
        print(f"Unknown dataset type: {dataset_type}")
        return

    print(f"[{dataset_type.upper()}] Reading Target IDs from {results_json}...")
    try:
        with open(results_json, 'r') as f:
            results_data = json.load(f)
        target_ids = [r['question_idx'] for r in results_data]
        print(f"Found {len(target_ids)} target IDs.")
    except Exception as e:
        print(f"Error reading results JSON: {e}")
        return

    print(f"Reading Baseline JSON from {baseline_json}...")
    try:
        with open(baseline_json, 'r') as f:
            data = json.load(f)
        print(f"Loaded {len(data)} records.")
        
        # Filter for target IDs
        filtered_data = [d for d in data if d.get('question_idx') in target_ids]
        print(f"Found {len(filtered_data)} matching records.")
        
        # Create DataFrame
        df = pd.DataFrame(filtered_data)
        
        # Only keep essential columns - exclude large trace/reflection data
        cols = [
            'question_idx', 'question_text', 'answer', 'gt_answer', 'em', 'f1', 'reward',
            'n_calls', 'n_badcalls', 'input_tokens', 'output_tokens', 'total_tokens',
            'num_trials', 'max_trials', 'framework'
        ]
        
        # Add missing columns if any
        for col in cols:
            if col not in df.columns:
                df[col] = None
                
        # Filter/Order columns
        df = df[cols]
        
        print(f"Writing baseline CSV to {output_csv}...")
        df.to_csv(output_csv, index=False, sep='|')
        print("Done.")
        
    except Exception as e:
        print(f"Error generating baseline CSV: {e}")

if __name__ == "__main__":
    # You can pass argument or uncomment/comment
    # generate_baseline_csv('fever')
    generate_baseline_csv('hotpotqa')
