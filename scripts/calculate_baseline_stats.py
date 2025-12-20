
import json
import argparse
import sys
import numpy as np

def calculate_stats(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
        return

    # Filter distinct by question_idx (take latest run for each)
    latest_results = {}
    for entry in data:
        idx = entry.get('question_idx')
        latest_results[idx] = entry

    data = list(latest_results.values())
    total = len(data)
    if total == 0:
        print("No results found.")
        return

    # Calculate metrics
    em_scores = [float(entry.get('em', 0)) for entry in data]
    f1_scores = [float(entry.get('f1', 0)) for entry in data]
    answer_presence = [1 if entry.get('answer', 'UNKNOWN') != 'UNKNOWN' else 0 for entry in data]
    
    # Identify failures
    failures = [entry for entry in data if entry.get('em', 0) == 0]

    print(f"--- Statistics for {file_path} ---")
    print(f"Total Unique Examples: {total}")
    print(f"Exact Match (EM): {np.mean(em_scores):.2%}")
    print(f"F1 Score: {np.mean(f1_scores):.2%}")
    print(f"Answer Rate: {np.mean(answer_presence):.2%}")
    print("-" * 30)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file_path", help="Path to results.json file")
    args = parser.parse_args()
    calculate_stats(args.file_path)
