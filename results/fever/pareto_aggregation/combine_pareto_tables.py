import pandas as pd
import os

def main():
    base_dir = "results/fever"
    output_dir = os.path.join(base_dir, "pareto_aggregation")
    output_file = os.path.join(output_dir, "pareto_summary.csv")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    all_data = []
    
    # List all subdirectories in results/fever
    if not os.path.exists(base_dir):
        print(f"Directory {base_dir} does not exist.")
        return

    # Helper function to find pareto_table.csv recursively
    def find_pareto_files(directory):
        pareto_files = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file == "pareto_table.csv":
                    pareto_files.append(os.path.join(root, file))
        return pareto_files

    frameworks = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    
    for framework in frameworks:
        # Skip the aggregation folder itself if it exists
        if framework == "pareto_aggregation":
            continue
            
        framework_path = os.path.join(base_dir, framework)
        pareto_files = find_pareto_files(framework_path)
        
        for file_path in pareto_files:
            try:
                df = pd.read_csv(file_path)
                
                # Check if required columns exist before proceeding
                required_cols = ['question_idx', 'em', 'f1', 'total_tokens']
                if not all(col in df.columns for col in required_cols):
                    print(f"Skipping {file_path}: Missing columns. Found: {df.columns.tolist()}")
                    continue

                # Add framework column if not present (it usually is, but good to ensure/overwrite for consistency)
                # If 'framework' is already there, we might want to trust it or overwrite it with the directory name.
                # The user requirement implies we might need to rely on the 'framework' column in the csv,
                # OR set it based on the folder. The prompt asks to combine pareto CSVs.
                # Let's check if 'framework' exists, if so use it, if not use directory name.
                if 'framework' not in df.columns:
                     df['framework'] = framework
                
                # Select specific columns
                cols_to_keep = ['question_idx', 'framework', 'em', 'f1', 'total_tokens']
                df_subset = df[cols_to_keep]
                
                all_data.append(df_subset)
                print(f"Processed {file_path} (Rows: {len(df_subset)})")
                
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_df.to_csv(output_file, index=False)
        print(f"\nSuccessfully created {output_file}")
        print(f"Total rows: {len(combined_df)}")
        try:
            print(combined_df.head(2).to_markdown(index=False, numalign="left", stralign="left"))
        except ImportError:
            print(combined_df.head(2))
    else:
        print("No pareto_table.csv files found or processed.")

if __name__ == "__main__":
    main()
