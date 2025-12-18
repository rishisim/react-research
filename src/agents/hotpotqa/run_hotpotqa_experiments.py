"""
HotPotQA Experiment Runner with Continuation System

Features:
- Seed-based directory naming for easy result accumulation
- Question continuation system (resume from previous runs)
- Flexible framework selection (react, cot_sc, majority_voting, reflexion, self_reflection)
- Error handling with retry capability
- LLM-as-judge evaluation alongside EM/F1
"""

import os
import sys
import json
import random
import argparse
from datetime import datetime
from typing import List, Dict, Any, Set
from pathlib import Path

from hotpotqa_utils import WEBTHINK_PROMPT_TEMPLATE
from react_agent import run_react
from reflexion_react_agent import run_reflexion_react
from majority_voting_agent import run_majority_voting
from cot_sc_agent import run_cot_sc
from self_reflection_agent import run_self_reflection


class HotPotQAExperimentRunner:
    
    # Map framework names to their execution functions
    FRAMEWORK_MAP = {
        'react': run_react,
        'reflexion': run_reflexion_react,
        'majority_voting': run_majority_voting,
        'cot_sc': run_cot_sc,
        'self_reflection': run_self_reflection
    }
    
    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        num_examples: int = 5,
        frameworks: List[str] = None,
        results_base_dir: str = "../../../results/hotpotqa",
        seed: int = 42,
        retry_failed: bool = False
    ):
        """
        Initialize experiment runner.
        
        Args:
            model: Gemini model to use
            num_examples: Number of HotPotQA examples to run
            frameworks: List of frameworks ['react', 'reflexion', 'majority_voting', 'cot_sc', 'self_reflection']
            results_base_dir: Base directory for results
            seed: Random seed for reproducibility
            retry_failed: Whether to retry previously failed questions
        """
        self.model = model
        self.num_examples = num_examples
        self.frameworks = frameworks or ['react']
        self.seed = seed
        self.retry_failed = retry_failed
        self.max_hotpotqa_dev_examples = 7405
        
        # Create seed-based run directory (accumulates across runs)
        run_name = f"seed{seed}_{model.replace('/', '-')}"
        
        script_dir = Path(__file__).parent
        self.results_dir = (script_dir / results_base_dir / run_name).resolve()
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        print("="*70, flush=True)
        print(f"[EXPERIMENT] HotPotQA Agent Evaluation", flush=True)
        print("="*70, flush=True)
        print(f"Results directory: {self.results_dir}", flush=True)
        print(f"Seed: {seed}", flush=True)
        print(f"Model: {model}", flush=True)
        print(f"Frameworks: {', '.join(self.frameworks)}", flush=True)
        print(f"Examples to run: {num_examples}", flush=True)
        print(f"Retry failed: {retry_failed}", flush=True)
        print("="*70, flush=True)
        
        # File paths
        self.config_path = self.results_dir / "config.json"
        self.processed_indices_path = self.results_dir / "processed_indices.json"
        self.failed_indices_path = self.results_dir / "failed_indices.json"
        self.run_history_path = self.results_dir / "run_history.json"
        
        # Initialize or load config
        self.config = self._load_or_create_config()
        
        # Initialize result storage
        self.results = {fw: [] for fw in self.frameworks}
        self._load_existing_results()
    
    def _load_or_create_config(self) -> Dict:
        """Load existing config or create new one."""
        if self.config_path.exists():
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            print(f"[CONFIG] Loaded existing config from previous runs")
            return config
        else:
            config = {
                "seed": self.seed,
                "model": self.model,
                "max_hotpotqa_dev_examples": self.max_hotpotqa_dev_examples,
                "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            print(f"[CONFIG] Created new config")
            return config
    
    def _load_existing_results(self):
        """Load existing results for each framework."""
        for framework in self.frameworks:
            result_path = self.results_dir / f"{framework}.json"
            if result_path.exists():
                with open(result_path, 'r', encoding='utf-8') as f:
                    self.results[framework] = json.load(f)
                print(f"[LOAD] Found {len(self.results[framework])} existing results for {framework}")
    
    def load_processed_indices(self) -> Set[int]:
        """Load indices that have been successfully processed."""
        if not self.processed_indices_path.exists():
            return set()
        
        with open(self.processed_indices_path, 'r', encoding='utf-8') as f:
            indices = json.load(f)
        return set(indices)
    
    def load_failed_indices(self) -> Set[int]:
        """Load indices that have failed in previous runs."""
        if not self.failed_indices_path.exists():
            return set()
        
        with open(self.failed_indices_path, 'r', encoding='utf-8') as f:
            indices = json.load(f)
        return set(indices)
    
    def save_processed_index(self, idx: int):
        """Mark an index as successfully processed."""
        processed = self.load_processed_indices()
        processed.add(idx)
        
        with open(self.processed_indices_path, 'w', encoding='utf-8') as f:
            json.dump(sorted(list(processed)), f, indent=2)
    
    def save_failed_index(self, idx: int):
        """Mark an index as failed."""
        failed = self.load_failed_indices()
        failed.add(idx)
        
        with open(self.failed_indices_path, 'w', encoding='utf-8') as f:
            json.dump(sorted(list(failed)), f, indent=2)
    
    def select_indices(self) -> List[int]:
        """Select indices to process, skipping already processed ones."""
        all_indices = list(range(self.max_hotpotqa_dev_examples))
        random.Random(self.seed).shuffle(all_indices)
        
        processed = self.load_processed_indices()
        failed = self.load_failed_indices()
        
        # Determine which indices to skip
        if self.retry_failed:
            skip_indices = processed  # Only skip successful ones
        else:
            skip_indices = processed | failed  # Skip both successful and failed
        
        # Find unprocessed indices
        unprocessed = [idx for idx in all_indices if idx not in skip_indices]
        
        selected = unprocessed[:self.num_examples]
        
        print(f"\n[INDICES] Total available: {self.max_hotpotqa_dev_examples}")
        print(f"[INDICES] Already processed: {len(processed)}")
        print(f"[INDICES] Previously failed: {len(failed)}")
        print(f"[INDICES] Selected for this run: {len(selected)}")
        
        return selected
    
    def run_framework(self, framework: str, idx: int) -> Dict[str, Any]:
        """
        Run a specific framework on a question index.
        
        Args:
            framework: Framework name
            idx: Question index
            
        Returns:
            Result dictionary (may contain 'error' key if failed)
        """
        print(f"  Running {framework}...", flush=True)
        
        try:
            agent_func = self.FRAMEWORK_MAP[framework]
            _, result = agent_func(idx=idx, prompt_template=WEBTHINK_PROMPT_TEMPLATE, to_print=False)
            
            # Add status flag
            result['status'] = 'success'
            
            answer = result.get('answer', 'UNKNOWN')
            gt = result.get('gt_answer', 'UNKNOWN')
            em = result.get('em', 0.0)
            llm_correct = result.get('llm_correct', False)
            
            print(f"  > {framework}: Answer={answer[:50]}... | GT={gt[:50]}... | EM={em} | LLM={llm_correct}", flush=True)
            
            return result
            
        except Exception as e:
            print(f"  > {framework}: ERROR - {str(e)}")
            return {
                'question_idx': idx,
                'error': str(e),
                'status': 'failed',
                'framework': framework
            }
    
    def run_all(self):
        """Run all configured experiments."""
        indices = self.select_indices()
        
        if not indices:
            print("\n[COMPLETE] No new indices to process!")
            return
        
        # Record run start
        run_start_time = datetime.now()
        
        print(f"\n{'='*70}", flush=True)
        print(f"[START] Processing {len(indices)} examples", flush=True)
        print(f"{'='*70}\n", flush=True)
        
        successful_count = 0
        failed_count = 0
        
        for i, idx in enumerate(indices, 1):
            print(f"\n{'-'*70}", flush=True)
            print(f"[EXAMPLE {i}/{len(indices)}] Index: {idx}", flush=True)
            print(f"{'-'*70}", flush=True)
            
            framework_results = {}
            example_success = True
            
            # Run each framework for this example
            for framework in self.frameworks:
                result = self.run_framework(framework, idx)
                framework_results[framework] = result
                
                # Check if this framework succeeded
                if result.get('status') == 'failed':
                    example_success = False
            
            # Save results for each framework
            for framework, result in framework_results.items():
                self.results[framework].append(result)
                self._save_framework_results(framework)
            
            # Mark index as processed or failed
            if example_success:
                self.save_processed_index(idx)
                successful_count += 1
                print(f"  [STATUS] Successfully processed")
            else:
                self.save_failed_index(idx)
                failed_count += 1
                print(f"  [STATUS] Failed (saved for potential retry)")
            
            # Update config
            self._save_config()
        
        # Record run completion
        run_end_time = datetime.now()
        self._save_run_history(run_start_time, run_end_time, len(indices), successful_count, failed_count)
        
        # Generate summary
        self.generate_summary()
        
        print(f"\n{'='*70}")
        print(f"[COMPLETE] Experiment finished")
        print(f"  Successful: {successful_count}/{len(indices)}")
        print(f"  Failed: {failed_count}/{len(indices)}")
        print(f"  Results saved to: {self.results_dir}")
        print(f"{'='*70}\n")
    
    def _save_framework_results(self, framework: str):
        """Save results for a specific framework."""
        result_path = self.results_dir / f"{framework}.json"
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(self.results[framework], f, indent=2, ensure_ascii=False)
    
    def _save_config(self):
        """Save configuration."""
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)
    
    def _save_run_history(self, start_time, end_time, attempted, successful, failed):
        """Save run history entry."""
        history = []
        if self.run_history_path.exists():
            with open(self.run_history_path, 'r', encoding='utf-8') as f:
                history = json.load(f)
        
        run_entry = {
            "run_id": len(history) + 1,
            "start_time": start_time.strftime("%Y-%m-%d %H:%M:%S"),
            "end_time": end_time.strftime("%Y-%m-%d %H:%M:%S"),
            "duration_minutes": (end_time - start_time).total_seconds() / 60,
            "num_attempted": attempted,
            "num_successful": successful,
            "num_failed": failed,
            "frameworks": self.frameworks
        }
        
        history.append(run_entry)
        
        with open(self.run_history_path, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
    
    def generate_summary(self):
        """Generate aggregate summary statistics."""
        summary = {}
        
        for framework, results in self.results.items():
            if not results:
                continue
            
            # Filter valid results (no errors)
            valid_results = [r for r in results if r.get('status') == 'success']
            total = len(results)
            valid = len(valid_results)
            
            if valid > 0:
                avg_em = sum(r.get('em', 0) for r in valid_results) / valid
                avg_f1 = sum(r.get('f1', 0) for r in valid_results) / valid
                avg_llm_correct = sum(1 for r in valid_results if r.get('llm_correct', False)) / valid
                total_calls = sum(r.get('n_calls', 0) for r in valid_results)
                total_badcalls = sum(r.get('n_badcalls', 0) for r in valid_results)
                em_success_count = sum(1 for r in valid_results if r.get('em', 0) == 1.0)
                llm_success_count = sum(1 for r in valid_results if r.get('llm_correct', False))
                
                summary[framework] = {
                    'total_examples': total,
                    'valid_examples': valid,
                    'error_count': total - valid,
                    'accuracy_em': round(avg_em, 4),
                    'accuracy_f1': round(avg_f1, 4),
                    'accuracy_llm_judge': round(avg_llm_correct, 4),
                    'em_success_count': em_success_count,
                    'llm_success_count': llm_success_count,
                    'total_llm_calls': total_calls,
                    'total_bad_calls': total_badcalls,
                    'avg_calls_per_example': round(total_calls / valid, 2) if valid > 0 else 0
                }
        
        # Save summary
        summary_path = self.results_dir / "summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        # Print summary
        print(f"\n{'='*70}")
        print(f"[SUMMARY] Experiment Statistics")
        print(f"{'='*70}")
        for framework, stats in summary.items():
            print(f"\n{framework.upper().replace('_', ' ')}:")
            print(f"  Valid Examples: {stats['valid_examples']}/{stats['total_examples']}")
            print(f"  Accuracy (EM): {stats['accuracy_em']:.2%}")
            print(f"  Accuracy (F1): {stats['accuracy_f1']:.2%}")
            print(f"  Accuracy (LLM-Judge): {stats['accuracy_llm_judge']:.2%}")
            print(f"  EM Success Count: {stats['em_success_count']}")
            print(f"  LLM Success Count: {stats['llm_success_count']}")
            print(f"  Total LLM Calls: {stats['total_llm_calls']}")
            print(f"  Avg Calls/Example: {stats['avg_calls_per_example']}")


def main():
    """Main entry point for running experiments."""
    parser = argparse.ArgumentParser(description="Run HotPotQA experiments with continuation support")
    parser.add_argument('--model', type=str, default='gemini-2.5-flash',
                       help='Gemini model to use')
    parser.add_argument('--num-examples', type=int, default=5,
                       help='Number of examples to run')
    parser.add_argument('--frameworks', type=str, nargs='+',
                       default=['react'],
                       choices=['react', 'reflexion', 'majority_voting', 'cot_sc', 'self_reflection'],
                       help='Frameworks to run')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--retry-failed', action='store_true',
                       help='Retry previously failed questions')
    
    args = parser.parse_args()
    
    runner = HotPotQAExperimentRunner(
        model=args.model,
        num_examples=args.num_examples,
        frameworks=args.frameworks,
        seed=args.seed,
        retry_failed=args.retry_failed
    )
    
    runner.run_all()


if __name__ == '__main__':
    main()
