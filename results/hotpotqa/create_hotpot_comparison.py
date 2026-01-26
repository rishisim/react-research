import pandas as pd
import sys
import os

def create_comparison_csv(prog_path, react_path, output_path):
    print(f"Loading Prog results from: {prog_path}")
    if not os.path.exists(prog_path):
        print(f"Error: Prog file {prog_path} not found.")
        return
        
    print(f"Loading ReAct results from: {react_path}")
    if not os.path.exists(react_path):
        print(f"Error: ReAct file {react_path} not found.")
        return

    # Load dataframes
    df_prog = pd.read_csv(prog_path)
    df_react = pd.read_csv(react_path)
    
    # Ensure they are sorted by question key or index if needed, but we will merge
    # Assuming 'question_idx' is the key
    
    # Select specific columns to keep
    # The user wants standard headers:
    # question_idx, question_text, answer, gt_answer, em, f1, n_calls, n_badcalls, input_tokens, output_tokens, total_tokens
    
    target_cols = [
        'question_idx', 'question_text', 'answer', 'gt_answer', 
        'em', 'f1', 'n_calls', 'n_badcalls', 
        'input_tokens', 'output_tokens', 'total_tokens'
    ]
    
    # Ensure frameworks have these columns
    # (The converters should have produced them)
    
    # Add framework identifier
    df_prog['framework'] = 'prog_ca_pruning'
    df_react['framework'] = 'react'
    
    # Filter columns (plus framework)
    # We'll include 'framework' to distinguish rows, even if not explicitly asked, 
    # otherwise identical QIDs are confusing. User said "i just simply want [headers]"
    # but likely implies standard structure. We will put framework at the end or beginning.
    # Actually, if the user was strict about headers, maybe they don't want 'framework'?
    # But then how to tell? comparison file implies comparison.
    # I'll add 'framework' as the last column.
    
    available_cols_prog = [c for c in target_cols if c in df_prog.columns]
    df_prog_sub = df_prog[available_cols_prog + ['framework']].copy()
    
    available_cols_react = [c for c in target_cols if c in df_react.columns]
    df_react_sub = df_react[available_cols_react + ['framework']].copy()
    
    # Concatenate
    df_final = pd.concat([df_prog_sub, df_react_sub], ignore_index=True)
    
    # Sort by question_idx then framework
    df_final = df_final.sort_values(by=['question_idx', 'framework'])
    
    # Reorder columns to match user list exactly, then framework
    final_cols_ordered = [c for c in target_cols if c in df_final.columns] + ['framework']
    df_final = df_final[final_cols_ordered]
    
    # Save
    df_final.to_csv(output_path, index=False)
    print(f"Comparison CSV created successfully: {output_path}")
    print(f"Total rows: {len(df_final)}")

if __name__ == "__main__":
    # Hardcoded paths based on current context
    base_dir = "results/hotpotqa"
    prog_csv = os.path.join(base_dir, "prog_ca_pruning/seed42_gemini-2.5-flash/prog_ca_pruning.csv")
    react_csv = os.path.join(base_dir, "react/seed42_gemini-2.5-flash/react.csv")
    output_csv = os.path.join(base_dir, "prog_ca_pruning/seed42_gemini-2.5-flash/comparison_prog_react.csv")
    
    create_comparison_csv(prog_csv, react_csv, output_csv)
