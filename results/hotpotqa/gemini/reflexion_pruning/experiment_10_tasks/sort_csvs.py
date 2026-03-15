import pandas as pd
import os

def sort_and_standardize_csvs():
    base_dir = "results/hotpotqa/reflexion_pruning/experiment_10_tasks"
    
    results_path = os.path.join(base_dir, "results.csv")
    baseline_path = os.path.join(base_dir, "reflexion_baseline.csv")
    
    # Canonical column order
    canonical_cols = [
        'question_idx', 'question_text', 'answer', 'gt_answer', 'em', 'f1', 
        'n_calls', 'n_badcalls', 'input_tokens', 'output_tokens', 'total_tokens',
        'action_pruned_happen', 'context_pruned_happen', 'total_pruning'
    ]
    
    # Load results.csv
    df_results = pd.read_csv(results_path)
    
    # Sort by question_idx
    df_results = df_results.sort_values('question_idx')
    
    # Get the ordered indices from results (after sorting)
    ordered_idx = df_results['question_idx'].tolist()
    print(f"Results has {len(ordered_idx)} tasks: {ordered_idx}")
    
    # Add missing cols and ensure only canonical
    for col in canonical_cols:
        if col not in df_results.columns:
            df_results[col] = 0
    if 'em' in df_results.columns:
        df_results['em'] = df_results['em'].astype(int)
    
    df_results = df_results[canonical_cols]
    df_results.to_csv(results_path, index=False)
    print(f"Updated and sorted {results_path}")
    
    # Load baseline and sort to match
    if os.path.exists(baseline_path):
        df_baseline = pd.read_csv(baseline_path)
        
        # Sort both to same order
        df_baseline = df_baseline.sort_values('question_idx')
        
        # Add missing cols
        for col in canonical_cols:
            if col not in df_baseline.columns:
                df_baseline[col] = 0
        if 'em' in df_baseline.columns:
            df_baseline['em'] = df_baseline['em'].astype(int)
            
        df_baseline = df_baseline[canonical_cols]
        df_baseline.to_csv(baseline_path, index=False)
        print(f"Updated and sorted {baseline_path}")
        
        # Verify order matches
        baseline_idx = df_baseline['question_idx'].tolist()
        results_idx = df_results['question_idx'].tolist()
        
        if set(baseline_idx) == set(results_idx):
            print("Both CSVs have matching task IDs")
        else:
            missing_in_baseline = set(results_idx) - set(baseline_idx)
            missing_in_results = set(baseline_idx) - set(results_idx)
            if missing_in_baseline:
                print(f"Missing in baseline: {missing_in_baseline}")
            if missing_in_results:
                print(f"Missing in results: {missing_in_results}")

if __name__ == "__main__":
    sort_and_standardize_csvs()
