import json
import pandas as pd
import os

# Define paths
base_dir = '/Users/rishisim/Documents/research/react-research'
input_json = os.path.join(base_dir, 'results/fever/reflexion_pruning/experiment_10_tasks/results.json')
output_csv = os.path.join(base_dir, 'results/fever/reflexion_pruning/experiment_10_tasks/results.csv')

def fix_csv():
    print(f"Reading JSON from {input_json}...")
    with open(input_json, 'r') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} records.")
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Only keep essential columns - exclude large trace/reflection data
    cols = [
        'question_idx', 'question_text', 'answer', 'gt_answer', 'em', 'f1', 'reward',
        'n_calls', 'n_badcalls', 'input_tokens', 'output_tokens', 'total_tokens',
        'num_trials', 'max_trials', 'framework'
    ]
    
    # Filter columns
    existing_cols = [c for c in cols if c in df.columns]
    df = df[existing_cols]
    
    print(f"Writing fixed CSV to {output_csv}...")
    df.to_csv(output_csv, index=False, sep='|')
    print("Done.")

if __name__ == "__main__":
    fix_csv()

