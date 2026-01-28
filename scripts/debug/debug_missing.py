
import json
import sys
from pathlib import Path

def find_missing_task():
    # Paths
    base_dir = Path("/Users/rishisim/Documents/research/react-research")
    task_ids_file = base_dir / "results/fever/dataset_sample_tasks/first_500_task_ids.json"
    processed_indices_file = base_dir / "results/fever/self_reflection/seed42_gemini-2.5-flash/processed_indices.json"
    fever_data_path = base_dir / "data/fever/paper_dev.jsonl"
    results_file = base_dir / "results/fever/self_reflection/seed42_gemini-2.5-flash/self_reflection.json"

    # Load task IDs (claim IDs)
    with open(task_ids_file, 'r') as f:
        data = json.load(f)
        if isinstance(data, list):
            target_claim_ids = set(data)
        elif isinstance(data, dict):
            target_claim_ids = set(data['task_ids'])
    
    print(f"Target claim IDs: {len(target_claim_ids)}")

    # Load processed indices (line indices)
    with open(processed_indices_file, 'r') as f:
        processed_indices = set(json.load(f))
    
    print(f"Processed line indices: {len(processed_indices)}")
    
    # Load results to check count
    if results_file.exists():
        with open(results_file, 'r') as f:
            results = json.load(f)
        print(f"Results in JSON file: {len(results)}")
    
    # Build mapping from line index to claim ID for the fever dev set
    # We only need to check the ones relevant to our target claim IDs to save time, 
    # but we need to map the PROCESSED indices back to claim IDs or vice versa.
    # More efficiently: map target claim IDs to line indices.
    
    claim_id_to_line_idx = {}
    line_idx_to_claim_id = {}
    
    print("Building mapping from FEVER data...")
    with open(fever_data_path, 'r') as f:
        for line_idx, line in enumerate(f):
            record = json.loads(line.strip())
            claim_id = record.get('id')
            if claim_id in target_claim_ids:
                claim_id_to_line_idx[claim_id] = line_idx
                line_idx_to_claim_id[line_idx] = claim_id

    # Convert target claim IDs to line indices
    target_line_indices = set(claim_id_to_line_idx.values())
    print(f"Found {len(target_line_indices)} matching line indices for the 500 claim IDs.")

    # Find missing
    missing_indices = target_line_indices - processed_indices
    
    if missing_indices:
        print(f"\nMISSING INDICES ({len(missing_indices)}):")
        for idx in missing_indices:
            print(f"Line Index: {idx}, Claim ID: {line_idx_to_claim_id.get(idx)}")
    else:
        print("\nNo missing indices found based on processed_indices.json.")

    # Check against results file as well
    if results_file.exists():
        result_indices = set()
        for r in results:
             # Assuming 'question_idx' or similar is the line index
             result_indices.add(r.get('question_idx'))
        
        missing_in_results = target_line_indices - result_indices
        if missing_in_results:
             print(f"\nMISSING IN RESULTS JSON ({len(missing_in_results)}):")
             for idx in missing_in_results:
                 print(f"Line Index: {idx}, Claim ID: {line_idx_to_claim_id.get(idx)}")
        else:
             print("\nNo missing indices in results JSON.")

if __name__ == "__main__":
    find_missing_task()
