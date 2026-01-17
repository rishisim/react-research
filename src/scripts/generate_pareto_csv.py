import json
import csv
import argparse
import os

def generate_csv(input_file, output_file):
    """
    Reads a JSON file containing a list of objects and writes specific fields to a CSV file.
    """
    try:
        with open(input_file, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in '{input_file}'.")
        return

    if not isinstance(data, list):
        print("Error: Input JSON must contain a list of objects.")
        return

    fields = [
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
        # ensure output directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fields)
            writer.writeheader()
            
            for item in data:
                row = {field: item.get(field, "") for field in fields}
                writer.writerow(row)
                
        print(f"Successfully wrote {len(data)} records to '{output_file}'.")
        
    except Exception as e:
        print(f"An error occurred while writing to CSV: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract Pareto table data to CSV.")
    parser.add_argument("input_file", help="Path to the input JSON file.")
    parser.add_argument("output_file", help="Path to the output CSV file.")
    args = parser.parse_args()

    generate_csv(args.input_file, args.output_file)
