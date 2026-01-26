import pandas as pd
import os

def standardize_reflexion_csvs():
    base_dir = "results/hotpotqa/reflexion_pruning/experiment_10_tasks"
    
    # Canonical column order (matching FEVER/Prog format)
    canonical_cols = [
        'question_idx', 'question_text', 'answer', 'gt_answer', 'em', 'f1', 
        'n_calls', 'n_badcalls', 'input_tokens', 'output_tokens', 'total_tokens',
        'action_pruned_happen', 'context_pruned_happen', 'total_pruning'
    ]
    
    # Process reflexion_baseline.csv (pipe-delimited)
    baseline_path = os.path.join(base_dir, "reflexion_baseline.csv")
    if os.path.exists(baseline_path):
        print(f"Processing {baseline_path}")
        df = pd.read_csv(baseline_path, sep='|')
        
        # Add missing columns with 0
        for col in canonical_cols:
            if col not in df.columns:
                df[col] = 0
                
        # Cast int types
        if 'em' in df.columns:
            df['em'] = df['em'].astype(int)
        
        # Select canonical columns only
        df_final = df[canonical_cols].copy()
        
        # Save as comma-separated
        df_final.to_csv(baseline_path, index=False)
        print(f"Updated {baseline_path} with {len(df_final)} rows, comma-separated.")
    
    # Process results.csv (comma-delimited already, but ensure column order)
    results_path = os.path.join(base_dir, "results.csv")
    if os.path.exists(results_path):
        print(f"Processing {results_path}")
        df = pd.read_csv(results_path)
        
        # Add missing columns with 0
        for col in canonical_cols:
            if col not in df.columns:
                df[col] = 0
        
        # Cast int types
        if 'em' in df.columns:
            df['em'] = df['em'].astype(int)
            
        # Select canonical columns only
        df_final = df[canonical_cols].copy()
        
        # Save
        df_final.to_csv(results_path, index=False)
        print(f"Updated {results_path} with {len(df_final)} rows.")

if __name__ == "__main__":
    standardize_reflexion_csvs()
