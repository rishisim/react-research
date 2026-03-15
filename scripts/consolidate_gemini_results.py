"""
Consolidate scattered Gemini FEVER results into a single directory per seed/run.

Gemini results are currently scattered across per-framework directories:
  gemini/react/seed42_gemini-2.5-flash/react.json
  gemini/cot_sc/seed42_gemini-2.5-flash/cot_sc.json
  ...

This script consolidates them into:
  gemini/seed42_mixed/react.json
  gemini/seed42_mixed/cot_sc.json
  ...
"""

import json
import os
import shutil
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "results" / "fever" / "gemini"

# Frameworks to consolidate
FRAMEWORKS = ['react', 'cot_sc', 'majority_voting', 'reflexion', 'self_reflection', 'action_prune']

# Map framework dir name -> json filename (they match, but let's be explicit)
FRAMEWORK_JSON_MAP = {
    'react': 'react.json',
    'cot_sc': 'cot_sc.json',
    'majority_voting': 'majority_voting.json',
    'reflexion': 'reflexion.json',
    'self_reflection': 'self_reflection.json',
    'action_prune': 'action_prune.json',
}


def consolidate_seed(seed: int, model_suffix: str = "gemini-2.5-flash"):
    """Consolidate all framework results for a given seed into one directory."""
    
    seed_dir_name = f"seed{seed}_{model_suffix}"
    output_dir = RESULTS_DIR / f"seed{seed}_mixed"
    
    if output_dir.exists():
        print(f"Output directory already exists: {output_dir}")
        print("Skipping to avoid overwriting. Delete it first if you want to re-consolidate.")
        return
    
    output_dir.mkdir(parents=True)
    print(f"Created: {output_dir}")
    
    all_processed_indices = set()
    config = None
    
    for fw_name, json_filename in FRAMEWORK_JSON_MAP.items():
        source_dir = RESULTS_DIR / fw_name / seed_dir_name
        source_json = source_dir / json_filename
        
        if not source_json.exists():
            # Try the framework name as the json filename
            source_json = source_dir / f"{fw_name}.json"
        
        if not source_json.exists():
            print(f"  WARNING: {source_json} not found, skipping {fw_name}")
            continue
        
        # Copy the JSON file
        dest_json = output_dir / json_filename
        shutil.copy2(source_json, dest_json)
        
        # Count entries
        with open(dest_json) as f:
            data = json.load(f)
        print(f"  Copied {fw_name}: {len(data)} entries")
        
        # Collect processed indices
        processed_path = source_dir / "processed_indices.json"
        if processed_path.exists():
            with open(processed_path) as f:
                indices = json.load(f)
            all_processed_indices.update(indices)
        
        # Grab config from first available
        if config is None:
            config_path = source_dir / "config.json"
            if config_path.exists():
                with open(config_path) as f:
                    config = json.load(f)
    
    # Write merged processed_indices
    with open(output_dir / "processed_indices.json", 'w') as f:
        json.dump(sorted(list(all_processed_indices)), f, indent=2)
    print(f"  Merged processed_indices: {len(all_processed_indices)} tasks")
    
    # Write config
    if config:
        with open(output_dir / "config.json", 'w') as f:
            json.dump(config, f, indent=2)
        print(f"  Config saved")
    
    print(f"\nDone! Consolidated to: {output_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Consolidate Gemini FEVER results")
    parser.add_argument("--seed", type=int, default=42, help="Seed to consolidate")
    parser.add_argument("--model-suffix", type=str, default="gemini-2.5-flash")
    args = parser.parse_args()
    
    consolidate_seed(args.seed, args.model_suffix)
