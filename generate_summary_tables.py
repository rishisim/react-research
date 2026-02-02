import os
import json
import csv
import pandas as pd

def load_data(dataset, framework, base_path):
    """
    Loads data for a given dataset and framework.
    Returns a DataFrame with columns: ['em', 'total_tokens'].
    """
    
    # Construct the directory path
    # Standard pattern: results/<dataset>/<framework>/seed42_gemini-2.5-flash/
    # But files might be named differently.
    
    dir_path = os.path.join(base_path, dataset, framework, "seed42_gemini-2.5-flash")
    
    data = []

    if framework == 'react':
        # React uses CSV files
        if dataset == 'hotpotqa':
            file_path = os.path.join(dir_path, 'react.csv')
        else: # fever
            file_path = os.path.join(dir_path, 'pareto_table.csv')
            
        if not os.path.exists(file_path):
            print(f"Warning: File not found: {file_path}")
            return None

        try:
            df = pd.read_csv(file_path)
            # Ensure 'em' and 'total_tokens' exist
            if 'em' not in df.columns or 'total_tokens' not in df.columns:
                 print(f"Warning: Missing columns in {file_path}. Columns found: {df.columns}")
                 return None
            return df[['em', 'total_tokens']]
        except Exception as e:
            print(f"Error reading CSV {file_path}: {e}")
            return None

    else:
        # Others use JSON files
        # framework names map to json filenames usually, but lets match exact findings
        # self_reflection -> self_reflection.json
        # reflexion -> reflexion.json
        # majority_voting -> majority_voting.json
        # cot_sc -> cot_sc.json
        
        filename = f"{framework}.json"
        
        file_path = os.path.join(dir_path, filename)
        
        if not os.path.exists(file_path):
             print(f"Warning: File not found: {file_path}")
             return None
        
        try:
            with open(file_path, 'r') as f:
                json_data = json.load(f)
            
            # JSON structure is usually a list of dicts
            if isinstance(json_data, list):
                for item in json_data:
                    # 'em' might be boolean or int. 'total_tokens' is int.
                    em = 1 if item.get('em') else 0
                    total_tokens = item.get('total_tokens', 0)
                    data.append({'em': em, 'total_tokens': total_tokens})
            else:
                 print(f"Warning: Unexpected JSON structure in {file_path}")
                 return None

            return pd.DataFrame(data)

        except Exception as e:
            print(f"Error reading JSON {file_path}: {e}")
            return None

def calculate_metrics(df, dataset, framework, baseline_metrics=None):
    """
    Calculates metrics for a dataframe.
    """
    if df is None or df.empty:
        return None

    num_tasks = len(df)
    sum_em = df['em'].sum()
    accuracy = sum_em / num_tasks if num_tasks > 0 else 0
    
    sum_tokens = df['total_tokens'].sum()
    tokens_per_task = sum_tokens / num_tasks if num_tasks > 0 else 0
    
    # Tokens per success task
    success_df = df[df['em'] == 1]
    sum_tokens_success = success_df['total_tokens'].sum()
    num_success = len(success_df)
    tokens_per_success_task = sum_tokens_success / num_success if num_success > 0 else 0
    
    metrics = {
        'Dataset': dataset,
        'Framework': framework,
        'Accuracy': accuracy,
        'Num': num_tasks,
        'SUM of em': sum_em,
        'COUNT of em': num_tasks, # Same as Num
        'SUM of total_tokens': sum_tokens,
        'Tokens/Task': tokens_per_task,
        'Tokens/Success Task': tokens_per_success_task
    }
    
    # Calculate Gains if baseline is provided
    if baseline_metrics:
        gain = accuracy - baseline_metrics['Accuracy']
        metrics['Gain'] = gain
        
        add_tokens = tokens_per_task - baseline_metrics['Tokens/Task']
        metrics['Add. Tokens'] = add_tokens
        
        # Avoid division by zero
        if gain > 0:
             metrics['Add. Token Per Percent Gain'] = add_tokens / (gain * 100)
        else:
             metrics['Add. Token Per Percent Gain'] = 0 # Or N/A, but keeping numeric for clean table
    else:
        # Baseline itself (React)
        metrics['Gain'] = 0.0
        metrics['Add. Tokens'] = 0.0
        metrics['Add. Token Per Percent Gain'] = 0.0

    return metrics

def main():
    base_path = "/Users/rishisim/Documents/research/react-research/results"
    datasets = ["hotpotqa", "fever"]
    # Order: React (Baseline) first, then others
    frameworks = ["react", "reflexion", "self_reflection", "majority_voting", "cot_sc"]
    
    # Readable names map
    framework_names = {
        "react": "ReAct",
        "reflexion": "Reflexion",
        "self_reflection": "Trajectory-Conditioned Answer Revision (TCAR)",
        "majority_voting": "Majority Voting",
        "cot_sc": "CoT-SC"
    }

    all_metrics = []

    for dataset in datasets:
        print(f"Processing dataset: {dataset}")
        baseline_metrics = None
        
        for fw in frameworks:
            print(f"  Processing framework: {fw}")
            df = load_data(dataset, fw, base_path)
            
            if df is not None:
                # Use React as baseline
                is_baseline = (fw == "react")
                
                current_metrics = calculate_metrics(df, dataset, framework_names.get(fw, fw), baseline_metrics)
                
                if is_baseline:
                    baseline_metrics = current_metrics
                
                all_metrics.append(current_metrics)
            else:
                print(f"  Skipping {fw} due to missing data.")

    # Create Summary Tables
    results_df = pd.DataFrame(all_metrics)
    
    # Formatting
    # Reorder columns
    cols = ['Dataset', 'Framework', 'Accuracy', 'Num', 'SUM of em', 'COUNT of em', 
            'SUM of total_tokens', 'Tokens/Task', 'Tokens/Success Task', 
            'Gain', 'Add. Tokens', 'Add. Token Per Percent Gain']
            
    results_df = results_df[cols]

    # Display tables for each dataset
    for dataset in datasets:
        print(f"\n### {dataset.upper()} Summary Table")
        dataset_df = results_df[results_df['Dataset'] == dataset].copy()
        
        if dataset_df.empty:
            print("No data found.")
            continue
            
        # Remove 'Dataset' column as it's redundant in the split view
        display_df = dataset_df.drop(columns=['Dataset'])
        
        # Manually format as markdown to avoid tabulate dependency
        headers = display_df.columns.tolist()
        print("| " + " | ".join(headers) + " |")
        print("| " + " | ".join(["---"] * len(headers)) + " |")
        
        for _, row in display_df.iterrows():
            row_str = []
            for col in headers:
                val = row[col]
                if isinstance(val, float):
                    row_str.append(f"{val:.4f}")
                else:
                    row_str.append(str(val))
            print("| " + " | ".join(row_str) + " |")

    # --- New Combined Accuracy Table ---
    print("\n### Combined Accuracy Summary Table")
    
    # Pivot or merge to get side-by-side
    hotpot_df = results_df[results_df['Dataset'] == 'hotpotqa'][['Framework', 'Accuracy', 'SUM of em', 'Num']].copy()
    fever_df = results_df[results_df['Dataset'] == 'fever'][['Framework', 'Accuracy', 'SUM of em', 'Num']].copy()
    
    # Rename columns
    hotpot_df = hotpot_df.rename(columns={'Accuracy': 'HotPotQA Acc', 'SUM of em': 'HP_Correct', 'Num': 'HP_Total'})
    fever_df = fever_df.rename(columns={'Accuracy': 'FEVER Acc', 'SUM of em': 'FV_Correct', 'Num': 'FV_Total'})
    
    # Merge on Framework
    merged_df = pd.merge(hotpot_df, fever_df, on='Framework', how='outer')
    
    # Format columns
    # Accuracy: * 100
    # Fraction: Correct / Total
    
    display_rows = []
    
    for _, row in merged_df.iterrows():
        display_row = {'Framework': row['Framework']}
        
        # HotPotQA
        if pd.notna(row['HotPotQA Acc']):
            display_row['HotPotQA Accuracy'] = f"{row['HotPotQA Acc'] * 100:.2f}"
            display_row['HotPotQA Num'] = f"{int(row['HP_Correct'])}/{int(row['HP_Total'])}"
        else:
            display_row['HotPotQA Accuracy'] = "-"
            display_row['HotPotQA Num'] = "-"
            
        # FEVER
        if pd.notna(row['FEVER Acc']):
            display_row['FEVER Accuracy'] = f"{row['FEVER Acc'] * 100:.2f}"
            display_row['FEVER Num'] = f"{int(row['FV_Correct'])}/{int(row['FV_Total'])}"
        else:
            display_row['FEVER Accuracy'] = "-"
            display_row['FEVER Num'] = "-"
            
        display_rows.append(display_row)
        
    final_df = pd.DataFrame(display_rows)
    
    # Print Markdown Table
    headers = ['Framework', 'HotPotQA Accuracy', 'HotPotQA Num', 'FEVER Accuracy', 'FEVER Num']
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---"] * len(headers)) + " |")
    
    for _, row in final_df.iterrows():
        print(f"| {row['Framework']} | {row['HotPotQA Accuracy']} | {row['HotPotQA Num']} | {row['FEVER Accuracy']} | {row['FEVER Num']} |")

if __name__ == "__main__":
    main()
