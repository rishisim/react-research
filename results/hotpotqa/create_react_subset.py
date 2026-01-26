import pandas as pd
import os

def create_react_subset():
    base_dir = "results/hotpotqa"
    # The reference set of 10 tasks comes from our prog_ca_pruning run
    prog_path = os.path.join(base_dir, "prog_ca_pruning/seed42_gemini-2.5-flash/prog_ca_pruning.csv")
    react_path = os.path.join(base_dir, "react/seed42_gemini-2.5-flash/react.csv")
    output_path = os.path.join(base_dir, "react/seed42_gemini-2.5-flash/react_10_tasks.csv")
    
    if not os.path.exists(prog_path):
        print("Prog csv not found to determine subset indices.")
        return
        
    if not os.path.exists(react_path):
        print("ReAct csv not found.")
        return
        
    # Get target IDs
    df_prog = pd.read_csv(prog_path)
    # Ensure question_idx is int or consistent type
    target_ids = df_prog['question_idx'].unique()
    print(f"Found {len(target_ids)} target task IDs: {target_ids}")
    
    # Load ReAct
    df_react = pd.read_csv(react_path)
    
    # Filter
    df_subset = df_react[df_react['question_idx'].isin(target_ids)].copy()
    
    # Sort to strictly match Prog order
    # Create a categorical type based on the order of target_ids
    df_subset['question_idx'] = pd.Categorical(df_subset['question_idx'], categories=target_ids, ordered=True)
    df_subset = df_subset.sort_values('question_idx')
    
    # Select specific columns
    desired_cols = [
        'question_idx', 'question_text', 'answer', 'gt_answer', 
        'em', 'f1', 'llm_correct', 
        'n_calls', 'n_badcalls', 
        'input_tokens', 'output_tokens', 'total_tokens'
    ]
    
    # Verify columns exist
    final_cols = [c for c in desired_cols if c in df_subset.columns]
    
    df_final = df_subset[final_cols].copy()
    
    # Ensure em is int (0/1) instead of boolean
    if 'em' in df_final.columns:
        df_final['em'] = df_final['em'].astype(int)

    # Ensure llm_correct is int (0/1) instead of boolean
    if 'llm_correct' in df_final.columns:
        df_final['llm_correct'] = df_final['llm_correct'].astype(int)

    
    df_final.to_csv(output_path, index=False)
    print(f"Created {output_path}")
    print(f"Rows: {len(df_final)}")
    print("Columns:", list(df_final.columns))

if __name__ == "__main__":
    create_react_subset()
