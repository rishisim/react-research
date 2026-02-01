import json
import pandas as pd
import os
import glob

# Configuration
BASE_DIR = '/Users/rishisim/Documents/research/react-research/results/hotpotqa'
OUTPUT_DIR = os.path.join(BASE_DIR, 'pareto_aggregation')
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'pareto_summary.csv')

# Define frameworks and their specific result file names
# All are expected to be in results/hotpotqa/{framework}/seed42_gemini-2.5-flash/
FRAMEWORKS = {
    'reflexion': 'reflexion.json',
    'react': 'react.json',
    'cot_sc': 'cot_sc.json',
    'action_prune': 'action_prune.json',
    'majority_voting': 'majority_voting.json',
    'prog_ca_pruning': 'prog_ca_pruning.json',
    'self_reflection': 'self_reflection.json'
}

def load_data():
    all_records = []

    for fw_name, json_filename in FRAMEWORKS.items():
        # Construct path
        # Pattern: results/hotpotqa/{fw_name}/seed42_gemini-2.5-flash/{json_filename}
        file_path = os.path.join(BASE_DIR, fw_name, 'seed42_gemini-2.5-flash', json_filename)
        
        if not os.path.exists(file_path):
            print(f"Warning: File not found for {fw_name}: {file_path}")
            continue

        print(f"Processing {fw_name} from {file_path}...")
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                
            # Data is expected to be a list of dicts
            if isinstance(data, list):
                for item in data:
                    # Extract relevant fields
                    # We need: question_idx, em (bool), total_tokens (int)
                    # Some files might have missing fields, safe get
                    
                    # Handle renaming for self_reflection
                    display_name = fw_name
                    if fw_name == 'self_reflection':
                        display_name = 'Trajectory-Conditioned Answer Revision (TCAR)'

                    record = {
                        'framework': display_name,
                        'question_idx': item.get('question_idx'),
                        'em': 1 if item.get('em') else 0, # Convert bool to int for easy summing
                        'total_tokens': item.get('total_tokens', 0)
                    }
                    all_records.append(record)
            else:
                 print(f"Error: Unexpected data format in {file_path}. Expected list.")

        except Exception as e:
            print(f"Error processing {fw_name}: {e}")

    return all_records

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    records = load_data()
    
    if not records:
        print("No records found!")
        return

    df = pd.DataFrame(records)
    
    print(f"Total records loaded: {len(df)}")
    print(df.groupby('framework').size())

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Summary saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
