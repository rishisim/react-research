import json
import csv
import argparse
import os

def create_pareto_table(json_path, output_path, framework_name):
    """
    Reads a JSON results file and creates a pareto_table.csv.
    """
    if not os.path.exists(json_path):
        print(f"Error: JSON file not found at {json_path}")
        return

    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON at {json_path}: {e}")
        return

    if not isinstance(data, list):
        print(f"Error: Expected a list of records in JSON, got {type(data)}")
        return
    
    # Define columns based on the user's existing pareto_table.csv format
    columns = [
        "question_idx",
        "question_text",
        "answer",
        "gt_answer",
        "em",
        "f1",
        "reward",
        "n_calls",
        "n_badcalls",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "framework",
        "status"
    ]

    try:
        with open(output_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(columns)

            for item in data:
                # Handle potential missing keys gracefully
                row = [
                    item.get("question_idx", ""),
                    item.get("question_text", ""),
                    item.get("answer", ""),
                    item.get("gt_answer", ""),
                    item.get("em", 0),
                    item.get("f1", 0),
                    item.get("reward", 0),
                    item.get("n_calls", 0),
                    item.get("n_badcalls", 0),
                    item.get("input_tokens", 0),
                    item.get("output_tokens", 0),
                    item.get("total_tokens", 0),
                    framework_name,  # Use the passed framework name
                    item.get("status", "success") # Default to success if not present
                ]
                writer.writerow(row)
        
        print(f"Successfully created pareto table at: {output_path}")

    except Exception as e:
        print(f"Error writing CSV to {output_path}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate pareto_table.csv from experiment JSON results.")
    parser.add_argument("--json_path", required=True, help="Path to the input JSON results file.")
    parser.add_argument("--output_path", help="Path to save the output CSV. Defaults to 'pareto_table.csv' in the JSON directory.")
    parser.add_argument("--framework_name", required=True, help="Name of the framework to log in the 'framework' column.")

    args = parser.parse_args()

    json_path = args.json_path
    if args.output_path:
        output_path = args.output_path
    else:
        # Default to same directory as JSON
        output_path = os.path.join(os.path.dirname(json_path), "pareto_table.csv")
    
    create_pareto_table(json_path, output_path, args.framework_name)
