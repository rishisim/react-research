import json
import pandas as pd
import os

def update_baseline_csv():
    """
    Update reflexion_baseline.csv by extracting data from the existing
    reflexion.json (which has 500 entries) for the same task IDs as results.csv.
    """
    base_dir = "results/hotpotqa/reflexion_pruning/experiment_10_tasks"
    reflexion_json_path = "results/hotpotqa/reflexion/seed42_gemini-2.5-flash/reflexion.json"
    
    results_csv_path = os.path.join(base_dir, "results.csv")
    baseline_csv_path = os.path.join(base_dir, "reflexion_baseline.csv")
    
    # Canonical columns
    canonical_cols = [
        'question_idx', 'question_text', 'answer', 'gt_answer', 'em', 'f1', 
        'n_calls', 'n_badcalls', 'input_tokens', 'output_tokens', 'total_tokens',
        'action_pruned_happen', 'context_pruned_happen', 'total_pruning'
    ]
    
    # Get target task IDs from results.csv
    df_results = pd.read_csv(results_csv_path)
    target_ids = set(df_results['question_idx'].tolist())
    print(f"Target IDs from results.csv: {sorted(target_ids)}")
    
    # Load reflexion.json
    with open(reflexion_json_path, 'r') as f:
        reflexion_data = json.load(f)
    
    # Filter for target IDs
    matching_entries = [e for e in reflexion_data if e.get('question_idx') in target_ids]
    print(f"Found {len(matching_entries)} matching entries in reflexion.json")
    
    # Convert to DataFrame
    rows = []
    for entry in matching_entries:
        row = {
            'question_idx': entry.get('question_idx'),
            'question_text': entry.get('question_text'),
            'answer': entry.get('answer'),
            'gt_answer': entry.get('gt_answer'),
            'em': int(entry.get('em', False)),
            'f1': entry.get('f1', 0.0),
            'n_calls': entry.get('n_calls', 0),
            'n_badcalls': entry.get('n_badcalls', 0),
            'input_tokens': entry.get('input_tokens', 0),
            'output_tokens': entry.get('output_tokens', 0),
            'total_tokens': entry.get('total_tokens', 0),
            # Baseline has no pruning
            'action_pruned_happen': 0,
            'context_pruned_happen': 0,
            'total_pruning': 0
        }
        rows.append(row)
    
    df_baseline = pd.DataFrame(rows)
    
    # Sort by question_idx to match results.csv order
    df_baseline = df_baseline.sort_values('question_idx')
    
    # Select canonical cols
    df_baseline = df_baseline[canonical_cols]
    
    # Save
    df_baseline.to_csv(baseline_csv_path, index=False)
    print(f"Updated {baseline_csv_path} with {len(df_baseline)} rows")
    
    # Verify both have same IDs
    results_ids = set(df_results['question_idx'].tolist())
    baseline_ids = set(df_baseline['question_idx'].tolist())
    
    if results_ids == baseline_ids:
        print("Both CSVs now have matching task IDs!")
    else:
        print(f"Mismatch! Results: {len(results_ids)}, Baseline: {len(baseline_ids)}")
        print(f"Missing in baseline: {results_ids - baseline_ids}")

if __name__ == "__main__":
    update_baseline_csv()
