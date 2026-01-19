"""
Combined Experiment Runner for HotPotQA

Executes the optimized combined agent (Majority Voting + CoT-SC) and logs
results to their respective standard directories as if they were independent runs.
"""

import os
import sys
import json
import random
import argparse
from pathlib import Path
from typing import List, Dict, Any, Set

from combined_cot_majority_agent import run_combined_agent
from hotpotqa_utils import WEBTHINK_PROMPT_TEMPLATE

class CombinedExperimentRunner:
    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        num_examples: int = 5,
        results_base_dir: str = "../../../results/fever",
        seed: int = 42,
        task_ids_file: str = None
    ):
        self.model = model
        self.num_examples = num_examples
        self.seed = seed
        self.max_hotpotqa_dev_examples = 7405 # Approx
        self.task_ids_file = task_ids_file
        
        script_dir = Path(__file__).parent
        project_root = script_dir.parent.parent.parent
        
        # Setup separate result directories
        base_mv_path = project_root / "results/hotpotqa/majority_voting"
        base_cot_path = project_root / "results/hotpotqa/cot_sc"
        
        run_name = f"seed{seed}_{model.replace('/', '-')}"
        
        self.mv_results_dir = (base_mv_path / run_name).resolve()
        self.cot_results_dir = (base_cot_path / run_name).resolve()
        
        self.mv_results_dir.mkdir(parents=True, exist_ok=True)
        self.cot_results_dir.mkdir(parents=True, exist_ok=True)
        
        print("="*70)
        print(f"[EXPERIMENT] COMBINED HotPotQA Execution (MV + CoT-SC)")
        print("="*70)
        print(f"MV Results: {self.mv_results_dir}")
        print(f"CoT Results: {self.cot_results_dir}")
        print(f"Seed: {seed}")
        print(f"Model: {model}")
        
        self.processed_indices_path = self.mv_results_dir / "processed_indices.json"
        
        self.mv_results = []
        self.cot_results = []
        
        self._load_existing_results()
        
        # IDs setup
        self.predefined_task_ids = None
        self.idx_to_task_id = {}
        if task_ids_file:
            self.predefined_task_ids, self.idx_to_task_id = self._load_task_ids_from_file()

    def _load_existing_results(self):
        # Load MV
        mv_path = self.mv_results_dir / "majority_voting.json"
        if mv_path.exists():
            with open(mv_path, 'r', encoding='utf-8') as f:
                self.mv_results = json.load(f)
        
        # Load CoT
        cot_path = self.cot_results_dir / "cot_sc.json"
        if cot_path.exists():
            with open(cot_path, 'r', encoding='utf-8') as f:
                self.cot_results = json.load(f)

    def load_processed_indices(self) -> Set[int]:
        if not self.processed_indices_path.exists():
            return set()
        with open(self.processed_indices_path, 'r', encoding='utf-8') as f:
            return set(json.load(f))

    def save_processed_index(self, idx: int):
        processed = self.load_processed_indices()
        processed.add(idx)
        with open(self.processed_indices_path, 'w', encoding='utf-8') as f:
            json.dump(sorted(list(processed)), f, indent=2)
            
        # Mirror to CoT
        cot_processed_path = self.cot_results_dir / "processed_indices.json"
        with open(cot_processed_path, 'w', encoding='utf-8') as f:
             json.dump(sorted(list(processed)), f, indent=2)

    def _load_task_ids_from_file(self):
        if not self.task_ids_file:
            print("[INFO] No task IDs file provided")
            return None, {}
        
        try:
            with open(self.task_ids_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            task_ids = data if isinstance(data, list) else data.get('task_ids', [])
            
            # For HotPotQA, we need to map task IDs (which are usually strings like '5a8b57f25542995d1e6f1371')
            # to dataset indices (0, 1, 2...) because WikiEnv uses indices.
            
            script_dir = Path(__file__).parent
            project_root = script_dir.parent.parent.parent
            hotpot_data_path = project_root / "data" / "hotpotqa" / "hotpot_dev_distractor_v1.json"
            
            print(f"[LOADING] Loading HotPotQA dataset from {hotpot_data_path}...")
            with open(hotpot_data_path, 'r', encoding='utf-8') as f:
                full_data = json.load(f)
                
            task_id_to_idx = {item['_id']: i for i, item in enumerate(full_data)}
            idx_to_task_id = {i: item['_id'] for i, item in enumerate(full_data)}
            
            valid_indices = []
            for tid in task_ids:
                if tid in task_id_to_idx:
                    valid_indices.append(task_id_to_idx[tid])
                else:
                    print(f"[WARNING] Task ID {tid} not found in dataset")
            
            print(f"[TASK_IDS] Mapped {len(valid_indices)} task IDs to indices")
            return valid_indices, idx_to_task_id
            
        except Exception as e:
            print(f"[ERROR] Failed to load task IDs: {e}")
            raise

    def select_indices(self) -> List[int]:
        if self.predefined_task_ids:
            all_indices = self.predefined_task_ids
        else:
            # Fallback to random sampling if no file provided
            # Logic: We might not want to load the whole 7405 items just to get indices if we assume they exist
            # But safer to just assume a range for random sampling
            all_indices = list(range(self.max_hotpotqa_dev_examples))
            random.Random(self.seed).shuffle(all_indices)
            
        processed = self.load_processed_indices()
        unprocessed = [idx for idx in all_indices if idx not in processed]
        return unprocessed[:self.num_examples]

    def run_all(self):
        indices = self.select_indices()
        if not indices:
            print("[COMPLETE] No new indices to process")
            return

        print(f"\n[START] Processing {len(indices)} examples\n")
        
        successful_count = 0
        
        for i, idx in enumerate(indices, 1):
            task_id = self.idx_to_task_id.get(idx, "Unknown")
            print(f"\n{'-'*70}")
            print(f"[EXAMPLE {i}/{len(indices)}] Index: {idx} (Task ID: {task_id})")
            print(f"{'-'*70}")
            
            try:
                # RUN COMBINED AGENT
                combined_results = run_combined_agent(
                    idx=idx, 
                    prompt_template=WEBTHINK_PROMPT_TEMPLATE, 
                    to_print=True
                )
                
                # Extract results
                mv_res = combined_results['majority_voting']
                cot_res = combined_results['cot_sc']
                
                mv_res['status'] = 'success'
                cot_res['status'] = 'success'
                
                # Add HotPotQA specific ID
                mv_res['hotpot_id'] = task_id
                cot_res['hotpot_id'] = task_id
                
                # Add to lists
                self.mv_results.append(mv_res)
                self.cot_results.append(cot_res)
                
                # Save to files
                self._save_results()
                
                # Mark done
                self.save_processed_index(idx)
                successful_count += 1
                
            except Exception as e:
                print(f"[ERROR] Failed on index {idx}: {e}")
                import traceback
                traceback.print_exc()
                continue
                
        self.generate_summaries()
        print(f"\n[COMPLETE] Finished {successful_count}/{len(indices)} tasks")

    def _save_results(self):
        # Save MV
        with open(self.mv_results_dir / "majority_voting.json", 'w', encoding='utf-8') as f:
            json.dump(self.mv_results, f, indent=2, ensure_ascii=False)
            
        # Save CoT
        with open(self.cot_results_dir / "cot_sc.json", 'w', encoding='utf-8') as f:
            json.dump(self.cot_results, f, indent=2, ensure_ascii=False)

    def generate_summaries(self):
        # Generate and save summary.json for MV
        self._generate_single_summary(self.mv_results, self.mv_results_dir, "majority_voting")
        
        # Generate and save summary.json for CoT
        self._generate_single_summary(self.cot_results, self.cot_results_dir, "cot_sc")

    def _generate_single_summary(self, results, directory, framework_name):
        valid = [r for r in results if r.get('status') == 'success']
        if not valid:
            return
            
        total = len(results)
        valid_count = len(valid)
        avg_em = sum(r.get('em', 0) for r in valid) / valid_count
        avg_f1 = sum(r.get('f1', 0) for r in valid) / valid_count
        
        summary = {
            framework_name: {
                'total_examples': total,
                'valid_examples': valid_count,
                'accuracy_em': round(avg_em, 4),
                'accuracy_f1': round(avg_f1, 4),
                'total_llm_calls': sum(r.get('n_calls', 0) for r in valid),
                'total_tokens': sum(r.get('total_tokens', 0) for r in valid)
            }
        }
        
        with open(directory / "summary.json", 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description="Run Combined HotPotQA Experiments")
    parser.add_argument('--num-examples', type=int, default=5)
    parser.add_argument('--task-ids-file', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    runner = CombinedExperimentRunner(
        num_examples=args.num_examples,
        task_ids_file=args.task_ids_file,
        seed=args.seed
    )
    
    runner.run_all()

if __name__ == '__main__':
    main()
