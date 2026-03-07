"""
Combined Experiment Runner for FEVER

Executes the optimized combined agent (Majority Voting + CoT-SC) and logs
results to their respective standard directories as if they were independent runs.
"""

import os
import sys
import json
import random
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Set

from combined_cot_majority_agent import run_combined_agent
from fever_utils import WEBTHINK_PROMPT_TEMPLATE

class CombinedExperimentRunner:
    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        num_examples: int = 5,
        results_base_dir: str = "../../../results/fever",
        seed: int = 42,
        retry_failed: bool = False,
        task_ids_file: str = None
    ):
        self.model = model
        self.num_examples = num_examples
        self.seed = seed
        self.retry_failed = retry_failed
        self.max_fever_dev_examples = 7405
        self.task_ids_file = task_ids_file
        
        script_dir = Path(__file__).parent
        project_root = script_dir.parent.parent.parent
        
        # Setup separate result directories
        # Route Qwen vs Gemini to appropriate model directories
        if "qwen" in model.lower():
            base_mv_path = project_root / "results/fever/qwen"
            base_cot_path = project_root / "results/fever/qwen"
        else:
            base_mv_path = project_root / "results/fever/gemini/majority_voting"
            base_cot_path = project_root / "results/fever/gemini/cot_sc"
        
        run_name = f"seed{seed}_{model.replace('/', '-')}"
        
        self.mv_results_dir = (base_mv_path / run_name).resolve()
        self.cot_results_dir = (base_cot_path / run_name).resolve()
        
        self.mv_results_dir.mkdir(parents=True, exist_ok=True)
        self.cot_results_dir.mkdir(parents=True, exist_ok=True)
        
        print("="*70)
        print(f"[EXPERIMENT] COMBINED FEVER Execution (MV + CoT-SC)")
        print("="*70)
        print(f"MV Results: {self.mv_results_dir}")
        print(f"CoT Results: {self.cot_results_dir}")
        print(f"Seed: {seed}")
        print(f"Model: {model}")
        
        # We share tracking files (processed/failed) in the MV directory as the 'primary'
        # But we should probably mirror them or just count on the framework logic.
        # Let's keep separate tracking to be safe, but simpler: 
        # We will check processed indices from ONE of them (MV) to avoid double work.
        
        self.processed_indices_path = self.mv_results_dir / "processed_indices.json"
        self.failed_indices_path = self.mv_results_dir / "failed_indices.json"
        
        self.mv_results = []
        self.cot_results = []
        
        self._load_existing_results()
        
        # IDs setup
        self.predefined_task_ids = None
        self.line_idx_to_claim_id = {}
        if task_ids_file:
            self.predefined_task_ids, _, self.line_idx_to_claim_id = self._load_task_ids_from_file()

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
            
        # Mirror to CoT for completeness, though we drive off MV
        cot_processed_path = self.cot_results_dir / "processed_indices.json"
        with open(cot_processed_path, 'w', encoding='utf-8') as f:
             json.dump(sorted(list(processed)), f, indent=2)

    def _load_task_ids_from_file(self):
        # (Reusing logic from run_fever_experiments.py - simplified copy)
        # Note: In a real refactor, this would be a shared utility.
        if not self.task_ids_file:
            print("[INFO] No task IDs file provided")
            return None, None, {}
        
        try:
            with open(self.task_ids_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            task_ids = data if isinstance(data, list) else data.get('task_ids', [])
            
            # Map to line indices (requires loading full dataset - expensive but necessary)
            script_dir = Path(__file__).parent
            project_root = script_dir.parent.parent.parent
            fever_data_path = project_root / "data" / "fever" / "paper_dev.jsonl"
            
            claim_id_to_line_idx = {}
            with open(fever_data_path, 'r', encoding='utf-8') as f:
                for line_idx, line in enumerate(f):
                    record = json.loads(line.strip())
                    claim_id = record.get('id')
                    if claim_id is not None:
                        claim_id_to_line_idx[claim_id] = line_idx
            
            line_indices = []
            original_claim_ids = []
            line_idx_to_claim_id = {}
            
            for claim_id in task_ids:
                if claim_id in claim_id_to_line_idx:
                    line_idx = claim_id_to_line_idx[claim_id]
                    line_indices.append(line_idx)
                    original_claim_ids.append(claim_id)
                    line_idx_to_claim_id[line_idx] = claim_id
            
            print(f"[TASK_IDS] Loaded and mapped {len(line_indices)} tasks")
            return line_indices, original_claim_ids, line_idx_to_claim_id
            
        except Exception as e:
            print(f"[ERROR] Failed to load task IDs: {e}")
            raise

    def select_indices(self) -> List[int]:
        if self.predefined_task_ids:
            all_indices = self.predefined_task_ids
        else:
            all_indices = list(range(self.max_fever_dev_examples))
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
            claim_id = self.line_idx_to_claim_id.get(idx, "Unknown")
            print(f"\n{'-'*70}")
            print(f"[EXAMPLE {i}/{len(indices)}] Line Index: {idx} (Claim ID: {claim_id})")
            print(f"{'-'*70}")
            
            try:
                # RUN COMBINED AGENT
                # This returns {'majority_voting': ..., 'cot_sc': ...}
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
                # We do not save failed indices logic here for simplicity, 
                # but could add it if needed.
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
        
        summary = {
            framework_name: {
                'total_examples': total,
                'valid_examples': valid_count,
                'accuracy_em': round(avg_em, 4),
                'total_llm_calls': sum(r.get('n_calls', 0) for r in valid),
                'total_tokens': sum(r.get('total_tokens', 0) for r in valid)
            }
        }
        
        with open(directory / "summary.json", 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description="Run Combined FEVER Experiments")
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
