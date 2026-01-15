#!/usr/bin/env python3
"""
Organize existing results into framework-specific folders with numbered runs.

This script:
1. Finds all existing result folders (timestamp-based and seed-based)
2. Organizes them into framework folders (reflexion, majority_voting, cot_sc, self_reflection)
3. Renames them to run_001, run_002, etc. based on chronological order
"""

import os
import json
import shutil
from pathlib import Path
from datetime import datetime

# Base results directory
RESULTS_DIR = Path(__file__).parent.parent / "results"

# Framework mappings
FRAMEWORKS = {
    'reflexion': 'reflexion.json',
    'majority_voting': 'majority_voting.json',
    'cot_sc': 'cot_sc.json',
    'self_reflection': 'self_reflection.json'
}

def get_folder_timestamp(folder_path):
    """Extract timestamp from folder name or config file."""
    folder_name = folder_path.name
    
    # Try to parse timestamp from folder name (YYYYMMDD_HHMMSS format)
    if '_' in folder_name and folder_name.split('_')[0].isdigit():
        try:
            date_part = folder_name.split('_')[0]
            time_part = folder_name.split('_')[1].split('_')[0]  # Handle additional underscores
            return datetime.strptime(f"{date_part}_{time_part}", "%Y%m%d_%H%M%S")
        except:
            pass
    
    # Try to get from config.json if it exists
    config_file = folder_path / "config.json"
    if config_file.exists():
        try:
            with open(config_file) as f:
                config = json.load(f)
                if 'timestamp' in config:
                    return datetime.fromisoformat(config['timestamp'])
        except:
            pass
    
    # Fallback to folder modification time
    return datetime.fromtimestamp(folder_path.stat().st_mtime)

def organize_dataset_results(dataset_path):
    """Organize results for a single dataset (fever or hotpotqa)."""
    dataset_name = dataset_path.name
    print(f"\n{'='*60}")
    print(f"Organizing {dataset_name.upper()} results")
    print(f"{'='*60}")
    
    # Find all result folders (exclude framework folders and action_prune)
    result_folders = []
    for item in dataset_path.iterdir():
        if item.is_dir() and item.name not in ['reflexion', 'majority_voting', 'cot_sc', 'self_reflection', 'action_prune']:
            result_folders.append(item)
    
    if not result_folders:
        print(f"No result folders found in {dataset_name}")
        return
    
    print(f"Found {len(result_folders)} result folders to organize")
    
    # Group folders by framework
    framework_folders = {fw: [] for fw in FRAMEWORKS.keys()}
    
    for folder in result_folders:
        # Check which framework files exist in this folder
        for framework, json_file in FRAMEWORKS.items():
            if (folder / json_file).exists():
                timestamp = get_folder_timestamp(folder)
                framework_folders[framework].append((folder, timestamp))
                print(f"  {folder.name} -> {framework} (timestamp: {timestamp})")
    
    # Move and rename folders for each framework
    for framework, folders in framework_folders.items():
        if not folders:
            continue
        
        print(f"\n{framework.upper()}:")
        
        # Sort by timestamp
        folders.sort(key=lambda x: x[1])
        
        # Create framework directory
        framework_dir = dataset_path / framework
        framework_dir.mkdir(exist_ok=True)
        
        # Move and rename each folder
        for idx, (folder, timestamp) in enumerate(folders, start=1):
            run_name = f"run_{idx:03d}"
            dest_path = framework_dir / run_name
            
            # If destination exists, skip or handle
            if dest_path.exists():
                print(f"  SKIP: {folder.name} -> {run_name} (already exists)")
                continue
            
            print(f"  MOVE: {folder.name} -> {framework}/{run_name}")
            shutil.move(str(folder), str(dest_path))
    
    # Also handle any stray files
    stray_files = [f for f in dataset_path.iterdir() if f.is_file()]
    if stray_files:
        print(f"\nStray files found:")
        for f in stray_files:
            print(f"  {f.name}")

def main():
    print("Results Organization Script")
    print("="*60)
    
    # Process FEVER results
    fever_path = RESULTS_DIR / "fever"
    if fever_path.exists():
        organize_dataset_results(fever_path)
    
    # Process HotPotQA results
    hotpotqa_path = RESULTS_DIR / "hotpotqa"
    if hotpotqa_path.exists():
        organize_dataset_results(hotpotqa_path)
    
    print("\n" + "="*60)
    print("Organization complete!")
    print("="*60)
    
    # Show final structure
    print("\nFinal structure:")
    for dataset in ['fever', 'hotpotqa']:
        dataset_path = RESULTS_DIR / dataset
        if dataset_path.exists():
            print(f"\n{dataset}/")
            for framework in FRAMEWORKS.keys():
                framework_path = dataset_path / framework
                if framework_path.exists():
                    run_folders = sorted([d.name for d in framework_path.iterdir() if d.is_dir() and d.name.startswith('run_')])
                    if run_folders:
                        print(f"  {framework}/")
                        for run in run_folders:
                            print(f"    {run}/")

if __name__ == "__main__":
    main()
