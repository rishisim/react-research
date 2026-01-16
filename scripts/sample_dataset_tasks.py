"""
Script to randomly sample 3000 tasks from FEVER and HotPotQA dev datasets.
Saves the sampled task IDs along with the random seed used.
"""

import json
import random
from pathlib import Path

# Configuration
SAMPLE_SIZE = 3000
RANDOM_SEED = 20260115  # Date-based seed: 2026-01-15

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

def sample_fever_tasks():
    """Sample 3000 task IDs from FEVER paper_dev.jsonl"""
    fever_dev_path = DATA_DIR / "fever" / "paper_dev.jsonl"
    output_dir = RESULTS_DIR / "fever" / "dataset_sample_tasks"
    
    # Load all task IDs
    task_ids = []
    with open(fever_dev_path, 'r') as f:
        for line in f:
            item = json.loads(line.strip())
            task_ids.append(item['id'])
    
    print(f"FEVER: Loaded {len(task_ids)} tasks from dev set")
    
    # Set seed and sample
    random.seed(RANDOM_SEED)
    sampled_ids = random.sample(task_ids, SAMPLE_SIZE)
    
    # Save to JSON
    output_data = {
        "dataset": "FEVER",
        "source_file": "paper_dev.jsonl",
        "total_available_tasks": len(task_ids),
        "sample_size": SAMPLE_SIZE,
        "random_seed": RANDOM_SEED,
        "sampled_task_ids": sampled_ids
    }
    
    output_path = output_dir / "sampled_3000_task_ids.json"
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"FEVER: Saved {len(sampled_ids)} sampled task IDs to {output_path}")
    return sampled_ids

def sample_hotpotqa_tasks():
    """Sample 3000 task IDs from HotPotQA distractor dev set"""
    hotpotqa_dev_path = DATA_DIR / "hotpotqa" / "hotpot_dev_distractor_v1.json"
    output_dir = RESULTS_DIR / "hotpotqa" / "dataset_sample_tasks"
    
    # Load all task IDs
    with open(hotpotqa_dev_path, 'r') as f:
        data = json.load(f)
    
    task_ids = [item['_id'] for item in data]
    print(f"HotPotQA: Loaded {len(task_ids)} tasks from distractor dev set")
    
    # Set seed and sample
    random.seed(RANDOM_SEED)
    sampled_ids = random.sample(task_ids, SAMPLE_SIZE)
    
    # Save to JSON
    output_data = {
        "dataset": "HotPotQA",
        "source_file": "hotpot_dev_distractor_v1.json",
        "total_available_tasks": len(task_ids),
        "sample_size": SAMPLE_SIZE,
        "random_seed": RANDOM_SEED,
        "sampled_task_ids": sampled_ids
    }
    
    output_path = output_dir / "sampled_3000_task_ids.json"
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"HotPotQA: Saved {len(sampled_ids)} sampled task IDs to {output_path}")
    return sampled_ids

if __name__ == "__main__":
    print(f"Using random seed: {RANDOM_SEED}")
    print("-" * 50)
    
    fever_ids = sample_fever_tasks()
    print("-" * 50)
    
    hotpotqa_ids = sample_hotpotqa_tasks()
    print("-" * 50)
    
    print("Sampling complete!")
