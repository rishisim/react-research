import json
import pandas as pd
import os
import sys

def update_reflexion_csvs():
    base_dir = "results/hotpotqa/reflexion_pruning/experiment_10_tasks"
    
    # Files to process
    files_map = {
        "results.json": "results.csv",
        "reflexion_baseline.json": "reflexion_baseline.csv"
    }
    
    for json_file, csv_file in files_map.items():
        json_path = os.path.join(base_dir, json_file)
        csv_path = os.path.join(base_dir, csv_file)
        
        if not os.path.exists(json_path):
            print(f"Warning: {json_path} not found.")
            # Fallback: Read CSV directly if it exists, to just add the columns
            if os.path.exists(csv_path):
                 print(f"Fallback: Reading {csv_path} directly to add columns.")
                 # Load CSV
                 try:
                    # Detect separator based on filename
                    if 'reflexion_baseline.csv' in csv_file:
                        df = pd.read_csv(csv_path, sep='|')
                        sep = '|'
                    else:
                        df = pd.read_csv(csv_path)
                        sep = ','
                    
                    # Target columns
                    target_cols = [
                        'question_idx', 'question_text', 'answer', 'gt_answer', 'em', 'f1', 
                        'n_calls', 'n_badcalls', 'input_tokens', 'output_tokens', 'total_tokens',
                        'action_pruned_happen', 'context_pruned_happen', 'total_pruning'
                    ]
                    
                    # Add missing with 0
                    for col in target_cols:
                        if col not in df.columns:
                            df[col] = 0
                    
                    # Ensure int casting
                    if 'em' in df.columns:
                         df['em'] = df['em'].astype(int)
                    if 'llm_correct' in df.columns:
                         df['llm_correct'] = df['llm_correct'].fillna(0).astype(int)
                         
                    # Save back
                    df.to_csv(csv_path, index=False, sep=sep)
                    print(f"Updated {csv_path} (fallback) with {len(df)} rows.")
                 except Exception as e:
                     print(f"Error reading CSV fallback: {e}")
            continue
            
        print(f"Processing {json_path} -> {csv_file}")
        
        with open(json_path, 'r') as f:
            data = json.load(f)
            
        processed_rows = []
        for entry in data:
            row = entry.copy()
            
            # Pruning stats
            # Check for 'pruning' key (like in prog)
            pruning = entry.get('pruning', {})
            act_p = pruning.get('action_pruning', {})
            ctx_p = pruning.get('context_pruning', {})
            
            # Extract or default to 0
            row['action_pruned_happen'] = act_p.get('total_pruned', 0)
            row['context_pruned_happen'] = ctx_p.get('pruned_sentences_count', 0)
            
            # Logic: total = action + context
            # Sometimes 'total_pruned' might be aggregated differently, but this is consistent with Prog
            row['total_pruning'] = row['action_pruned_happen'] + row['context_pruned_happen']
            
            # Keep other key columns if they exist, or clean up structure
            # Remove complex nested objects from CSV row to avoid format issues
            if 'all_traces' in row: del row['all_traces']
            if 'pruning' in row: del row['pruning']
            if 'llm_calls' in row: del row['llm_calls']
            
            processed_rows.append(row)
            
        # Convert to DF
        df = pd.DataFrame(processed_rows)
        
        # Desired columns (add any missing ones as 0/NaN)
        target_cols = [
            'question_idx', 'question_text', 'answer', 'gt_answer', 'em', 'f1', 
            'n_calls', 'n_badcalls', 'input_tokens', 'output_tokens', 'total_tokens',
            'action_pruned_happen', 'context_pruned_happen', 'total_pruning'
        ]
        
        # Add missing cols with 0
        for col in target_cols:
            if col not in df.columns:
                df[col] = 0
                
        # Cast em/llm_correct to int if present
        if 'em' in df.columns:
            df['em'] = df['em'].astype(int)
            
        # Select and save
        available_cols = [c for c in target_cols if c in df.columns]
        df_final = df[available_cols]
        
        df_final.to_csv(csv_path, index=False)
        print(f"Updated {csv_path} with {len(df_final)} rows.")

if __name__ == "__main__":
    update_reflexion_csvs()
