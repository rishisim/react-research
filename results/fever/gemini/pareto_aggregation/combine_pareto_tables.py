import pandas as pd
import json
import os

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    gemini_dir = os.path.dirname(base_dir)  # results/fever/gemini
    output_file = os.path.join(base_dir, "pareto_summary.csv")
    
    # Find all seed*_mixed directories
    seed_dirs = sorted([
        d for d in os.listdir(gemini_dir)
        if os.path.isdir(os.path.join(gemini_dir, d)) and d.startswith("seed") and "_mixed" in d
    ])
    
    if not seed_dirs:
        print("No seed*_mixed directories found.")
        return
    
    print(f"Found seed directories: {seed_dirs}")
    
    # JSON result files to look for
    json_files = [
        'react.json', 'cot_sc.json', 'majority_voting.json',
        'reflexion.json', 'self_reflection.json', 'action_prune.json'
    ]
    
    all_data = []
    
    for seed_dir_name in seed_dirs:
        seed_path = os.path.join(gemini_dir, seed_dir_name)
        
        for json_file in json_files:
            filepath = os.path.join(seed_path, json_file)
            if not os.path.exists(filepath):
                continue
            
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                
                if not isinstance(data, list):
                    continue
                
                for item in data:
                    framework = item.get('framework', json_file.replace('.json', ''))
                    
                    # Rename self_reflection -> TCAR
                    if framework == 'self_reflection':
                        framework = 'Trajectory-Conditioned Answer Revision (TCAR)'
                    
                    row = {
                        'question_idx': item.get('question_idx', ''),
                        'framework': framework,
                        'em': item.get('em', 0),
                        'f1': item.get('f1', 0),
                        'total_tokens': item.get('total_tokens', 0),
                    }
                    all_data.append(row)
                
                print(f"  {seed_dir_name}/{json_file}: {len(data)} entries")
                
            except Exception as e:
                print(f"  Error processing {seed_dir_name}/{json_file}: {e}")
    
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
