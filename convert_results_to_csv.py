
import json
import csv
import os

def convert_to_csv():
    json_path = 'results/fever/prog_CA_pruning/react_sample_10_tuned_v3/prog_ca_pruning_results.json'
    csv_path = 'results/fever/prog_CA_pruning/react_sample_10_tuned_v3/prog_ca_pruning_results.csv'
    
    # Columns from v2 format
    fieldnames = [
        'question_idx', 'question_text', 'answer', 'gt_answer', 'em', 'f1', 
        'n_calls', 'n_badcalls', 'input_tokens', 'output_tokens', 'total_tokens', 
        'pruned_actions', 'evidence_items', 'visited_pages'
    ]
    
    print(f"Reading {json_path}...")
    
    # Read the entire file as a JSON array
    data = []
    with open(json_path, 'r') as f:
        try:
            content = f.read()
            # Handle case where file might be a sequence of objects not a list (hybrid)
            # But based on the error logs showing `[` and commas, it looks like a proper JSON array or pretty-printed list.
            # Let's try loading it as a whole.
            data = json.loads(content)
            if not isinstance(data, list):
                # Fallback if it was actually just one object
                data = [data]
        except json.JSONDecodeError:
            print("Failed to parse as full JSON array. Trying line-by-line fallback...")
            # Fallback to line-by-line if it was actually NDJSON but with some formatting
            f.seek(0)
            for line in f:
                if line.strip():
                    try:
                        data.append(json.loads(line))
                    except:
                        pass

    print(f"Found {len(data)} records.")

    # Write to CSV
    with open(csv_path, 'w', newline='') as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        
        for entry in data:
            # Extract basic fields
            row = {
                'question_idx': entry.get('question_idx'),
                'question_text': entry.get('question_text'),
                'answer': entry.get('answer'),
                'gt_answer': entry.get('gt_answer'),
                'em': entry.get('em'),
                'f1': entry.get('em'), # In FEVER, F1 is same as EM usually
                'n_calls': entry.get('n_calls'),
                'n_badcalls': entry.get('n_badcalls'),
                'input_tokens': entry.get('input_tokens'),
                'output_tokens': entry.get('output_tokens'),
                'total_tokens': entry.get('total_tokens'),
                
                # Extract pruning stats
                'pruned_actions': entry.get('pruning', {}).get('action_pruning', {}).get('total_pruned', 0),
                'evidence_items': entry.get('pruning', {}).get('context_pruning', {}).get('evidence_items', 0),
                
                # Extract visited pages count (was missing in previous script)
                'visited_pages': len(entry.get('pruning', {}).get('context_pruning', {}).get('visited_pages', []))
            }
            
            # Map top-level keys if they were flat in the JSON
            if 'pruned_actions' in entry and row['pruned_actions'] == 0:
                 row['pruned_actions'] = entry.get('pruned_actions')
            if 'evidence_items' in entry and row['evidence_items'] == 0:
                 row['evidence_items'] = entry.get('evidence_items')
            
            writer.writerow(row)
            
    print(f"Converted to {csv_path}")

if __name__ == "__main__":
    convert_to_csv()
