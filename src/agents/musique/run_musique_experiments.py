"""
Musique Experiment Runner
"""

import sys
import os
import argparse
import time
import json
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from react_agent import run_react
from nexus_wrapper import run_nexus
from musique_utils import analyze_decomposition_performance

FRAMEWORK_MAP = {
    'react': run_react,
    'nexus': run_nexus
}

def run_experiments(num_examples, frameworks, start_idx=0, indices=None):
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../results/musique")
    os.makedirs(results_dir, exist_ok=True)
    
    summary_stats = {}
    
    # Determine indices to run
    if indices:
        target_indices = indices
    else:
        target_indices = list(range(start_idx, start_idx + num_examples))
    
    print(f"Target Indices: {target_indices}")
    
    for fram in frameworks:
        runner = FRAMEWORK_MAP[fram]
        framework_results = []
        
        print(f"\nRunning {fram} on {len(target_indices)} examples...")
        
        correct = 0
        
        for i in target_indices:
            try:
                reward, info = runner(i, to_print=True)
                framework_results.append(info)
                if reward == 1.0:
                    correct += 1
            except Exception as e:
                print(f"[ERROR] Failed on index {i}: {e}")
                import traceback
                traceback.print_exc()
        
        # Save results
        if indices:
            suffix = "targeted"
        else:
            suffix = f"{start_idx}_{start_idx+num_examples}"
            
        filename = f"{results_dir}/{fram}_{suffix}.json"
        
        # If targeted, maybe append timestamp or unique name to avoid overwrite? 
        # For now, just overwrite 'targeted.json' or user can manage files.
        if indices:
             # simple filename for now
             pass
             
        with open(filename, 'w') as f:
            json.dump(framework_results, f, indent=4)
            
        print(f"Saved results to {filename}")
        
        # Analysis
        Accuracy = correct / len(target_indices) if target_indices else 0
        print(f"Overall Accuracy: {Accuracy:.2f}")
        
        decomp_summary = analyze_decomposition_performance(framework_results)
        print(decomp_summary)
        
        summary_stats[fram] = {
            'accuracy': Accuracy,
            'breakdown': decomp_summary
        }

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num', type=int, default=5)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--frameworks', nargs='+', default=['react', 'nexus'])
    parser.add_argument('--indices', nargs='+', type=int, help='Specific indices to run')
    args = parser.parse_args()
    
    run_experiments(args.num, args.frameworks, args.start, args.indices)
