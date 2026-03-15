import pandas as pd
import json
import os

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    qwen_results_dir = os.path.join(os.path.dirname(base_dir), "seed42_mixed")
    output_file = os.path.join(base_dir, "pareto_summary.csv")
    
    # JSON files to process and their framework names
    json_files = {
        'react.json': 'react',
        'cot_sc.json': 'cot_sc',
        'majority_voting.json': 'majority_voting',
        'reflexion.json': None,  # Use framework from JSON itself
        'self_reflection.json': 'self_reflection',
    }
    
    all_data = []
    
    for json_file, fw_override in json_files.items():
        filepath = os.path.join(qwen_results_dir, json_file)
        if not os.path.exists(filepath):
            print(f"Warning: {filepath} not found, skipping.")
            continue
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            if not isinstance(data, list):
                print(f"Warning: {json_file} is not a list, skipping.")
                continue
            
            for item in data:
                framework = fw_override if fw_override else item.get('framework', 'unknown')
                
                # Rename self_reflection -> TCAR, reflexion -> reflexion_react for consistency
                if framework == 'self_reflection':
                    framework = 'Trajectory-Conditioned Answer Revision (TCAR)'
                elif framework == 'reflexion':
                    framework = 'reflexion_react'
                
                row = {
                    'question_idx': item.get('question_idx', ''),
                    'framework': framework,
                    'em': item.get('em', 0),
                    'f1': item.get('f1', 0),
                    'total_tokens': item.get('total_tokens', 0),
                }
                all_data.append(row)
            
            print(f"Processed {json_file}: {len(data)} entries")
            
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
    
    if all_data:
        df = pd.DataFrame(all_data)
        df.to_csv(output_file, index=False)
        print(f"\nSuccessfully created {output_file}")
        print(f"Total rows: {len(df)}")
        print(f"\nFramework counts:")
        print(df['framework'].value_counts().to_string())
    else:
        print("No data found to process.")

if __name__ == "__main__":
    main()
