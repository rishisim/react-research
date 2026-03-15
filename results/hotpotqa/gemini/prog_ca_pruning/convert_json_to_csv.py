import json
import pandas as pd
import sys
import os

def convert_json_to_csv(json_path):
    if not os.path.exists(json_path):
        print(f"Error: File {json_path} not found.")
        return

    with open(json_path, 'r') as f:
        data = json.load(f)

    if not data:
        print("Error: JSON file is empty.")
        return

    # Process data to match FEVER format specific fields
    processed_data = []
    for entry in data:
        row = entry.copy()
        
        # Flatten pruning stats if present
        pruning = entry.get('pruning', {})
        action_p = pruning.get('action_pruning', {})
        context_p = pruning.get('context_pruning', {})
        
        # FEVER format columns mapping
        row['action_pruned_happen'] = action_p.get('total_pruned', 0)
        # For context pruning, we can use 'pruned_sentences_count' or just a binary flag if that was the intent.
        # Looking at FEVER data "context_pruned_happen" has values like 2, 3, 6, so it's likely a count.
        # In context_pruner.py we track 'pruned_sentences_count'.
        row['context_pruned_happen'] = context_p.get('pruned_sentences_count', 0)
        
        row['total_pruning'] = row['action_pruned_happen'] + row['context_pruned_happen']
        row['visited_pages'] = context_p.get('visited_pages', 0)
        
        processed_data.append(row)

    # Convert to DataFrame
    df = pd.DataFrame(processed_data)

    # FEVER Column order
    fever_cols = [
        'question_idx',
        'question_text', 
        'answer',
        'gt_answer',
        'em',
        'f1',
        'llm_correct',
        'n_calls',
        'n_badcalls',
        'input_tokens',
        'output_tokens',
        'total_tokens',
        'action_pruned_happen',
        'context_pruned_happen',
        'total_pruning',
        'visited_pages'
    ]
    
    # Filter/Order columns. Add any missing as NaN/0
    existing_cols = [c for c in fever_cols if c in df.columns]
    
    # Select only the FEVER columns to exact match
    df_final = df[existing_cols].copy()
    
    # Ensure em is int (0/1) instead of boolean
    if 'em' in df_final.columns:
        df_final['em'] = df_final['em'].astype(int)

    # Ensure llm_correct is int (0/1) instead of boolean
    if 'llm_correct' in df_final.columns:
        df_final['llm_correct'] = df_final['llm_correct'].fillna(0).astype(int)
    

    # Save to CSV
    csv_path = json_path.replace('.json', '.csv')
    df_final.to_csv(csv_path, index=False)
    print(f"Successfully converted {json_path} to {csv_path} (FEVER format)")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python convert_json_to_csv.py <path_to_json>")
    else:
        convert_json_to_csv(sys.argv[1])
