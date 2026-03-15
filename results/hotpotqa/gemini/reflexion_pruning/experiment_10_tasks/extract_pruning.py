import json
import pandas as pd
import os

def extract_pruning_from_reflexion():
    base_dir = "results/hotpotqa/reflexion_pruning/experiment_10_tasks"
    json_path = os.path.join(base_dir, "results.json")
    csv_path = os.path.join(base_dir, "results.csv")
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    rows = []
    for entry in data:
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
        }
        
        # Aggregate pruning from all traces
        total_action_pruned = 0
        total_context_pruned = 0
        
        all_traces = entry.get('all_traces', {})
        for trace_key, trace_data in all_traces.items():
            if 'pruning' in trace_data:
                pruning = trace_data['pruning']
                action_p = pruning.get('action_pruning', {})
                context_p = pruning.get('context_pruning', {})
                
                total_action_pruned += action_p.get('total_pruned', 0)
                total_context_pruned += context_p.get('pruned_sentences_count', 0)
        
        row['action_pruned_happen'] = total_action_pruned
        row['context_pruned_happen'] = total_context_pruned
        row['total_pruning'] = total_action_pruned + total_context_pruned
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    print(f"Updated {csv_path} with {len(df)} rows.")
    print(f"Sample pruning values: {df[['question_idx', 'action_pruned_happen', 'context_pruned_happen', 'total_pruning']].head()}")

if __name__ == "__main__":
    extract_pruning_from_reflexion()
