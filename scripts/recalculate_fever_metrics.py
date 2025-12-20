
import json
import argparse
import sys
import os
import re
import string

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def recalculate_metrics(results_path, dataset_path):
    # Load Results
    try:
        with open(results_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
    except FileNotFoundError:
        print(f"Error: Results file not found: {results_path}")
        return

    # Load Dataset
    dataset = []
    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                dataset.append(json.loads(line))
    except FileNotFoundError:
        print(f"Error: Dataset file not found: {dataset_path}")
        return
        
    # Map index in dataset to Label
    # The 'index' in dataset might not match the list index if lines are skipped, 
    # but run logic usually treats line number (0-indexed) as index.
    # Let's verify if dataset has 'id' or if we rely on line number.
    # ReAct agent uses `self.data[self.data_idx]`. Wrappers use line index.
    
    # Calculate
    updated_results = []
    correct_count = 0
    total_count = 0
    
    for entry in results:
        idx = entry.get('question_idx')
        pred = entry.get('answer', 'UNKNOWN')
        
        if idx is not None and idx < len(dataset):
            gt_label = dataset[idx]['label']
            
            norm_pred = normalize_answer(pred)
            norm_gt = normalize_answer(gt_label)
            
            is_correct = (norm_pred == norm_gt)
            entry['gt_answer'] = gt_label
            entry['em'] = 1.0 if is_correct else 0.0
            entry['f1'] = 1.0 if is_correct else 0.0
            
            if is_correct:
                correct_count += 1
            total_count += 1
            updated_results.append(entry)
        else:
            print(f"Warning: Index {idx} out of bounds or missing.")
            updated_results.append(entry)

    # Save back if needed, or just print stats
    # Overwriting is risky if logic is wrong, so let's just output stats first.
    
    print(f"--- Recalculated Statistics for {results_path} ---")
    print(f"Total Examples: {total_count}")
    print(f"Accuracy (EM): {correct_count/total_count:.2%}" if total_count else "0.00%")
    print("-" * 30)

    # Backup and Save
    backup_path = results_path + ".bak"
    with open(backup_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2) # Save original state just in case
        
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(updated_results, f, indent=2)
    print(f"Updated results saved (Backup at {backup_path})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results_path", help="Path to nexus.json")
    parser.add_argument("dataset_path", help="Path to paper_dev.jsonl")
    args = parser.parse_args()
    recalculate_metrics(args.results_path, args.dataset_path)
