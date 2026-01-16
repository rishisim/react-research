"""
FEVER Experiment Runner with Organized Results System

This script runs FEVER experiments with baseline ReAct and Multi-Trace ReAct,
saving results in a structured, timestamped format.
"""

import os
import sys
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import fever_agent as fa


class ExperimentRunner:
    """Manages FEVER experiment execution and result storage."""
    
    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        num_examples: int = 5,
        frameworks: Optional[List[str]] = None,
        results_base_dir: str = "../../../results/fever",
        seed: int = 42
    ):
        """
        Initialize experiment runner.
        
        Args:
            model: Gemini model to use
            num_examples: Number of FEVER examples to run
            frameworks: List of frameworks to run ['baseline', 'multi_trace', 'reflexion']
            results_base_dir: Base directory for results (relative to this file)
            seed: Random seed for reproducibility
        """
        self.model = model
        self.num_examples = num_examples
        self.frameworks = frameworks or ['baseline', 'multi_trace']
        self.seed = seed
        
        # Create timestamped run directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{timestamp}_n{num_examples}_{model.replace('/', '-')}"
        
        script_dir = Path(__file__).parent
        self.results_dir = (script_dir / results_base_dir / run_name).resolve()
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"📁 Results will be saved to: {self.results_dir}")
        
        # Initialize result storage
        self.config = {
            "timestamp": timestamp,
            "model": model,
            "num_examples": num_examples,
            "frameworks": self.frameworks,
            "seed": seed,
            "max_fever_dev_examples": 7405,
            "indices": [],
            "rate_limit_sleep": 4.1
        }
        
        self.results = {
            'baseline': [],
            'multi_trace': [],
            'reflexion': []
        }
        
    def select_indices(self) -> List[int]:
        """Select random indices for experiments."""
        all_indices = list(range(self.config["max_fever_dev_examples"]))
        random.Random(self.seed).shuffle(all_indices)
        indices = all_indices[:self.num_examples]
        self.config["indices"] = indices
        return indices
    
    def run_baseline(self, idx: int) -> Dict[str, Any]:
        """Run baseline ReAct (single trace)."""
        print(f"  • Running baseline ReAct (num_traces=1)...")
        try:
            _, baseline_info = fa.webthink(
                idx=idx,
                initial_prompt_template=fa.WEBTHINK_PROMPT_TEMPLATE,
                to_print=False,
                num_traces=1
            )
            
            # Clean up result for storage
            result = {
                'question_idx': baseline_info.get('question_idx'),
                'question_text': baseline_info.get('question_text'),
                'answer': baseline_info.get('answer'),
                'gt_answer': baseline_info.get('gt_answer'),
                'em': baseline_info.get('em', 0),
                'f1': baseline_info.get('f1', 0),
                'reward': baseline_info.get('reward', 0),
                'n_calls': baseline_info.get('n_calls', 0),
                'n_badcalls': baseline_info.get('n_badcalls', 0),
                'traj': baseline_info.get('traj', '')
            }
            print(f"    ✓ Answer: {result['answer']} | GT: {result['gt_answer']} | EM: {result['em']}")
            return result
            
        except Exception as e:
            print(f"    ✗ ERROR: {e}")
            return {
                'question_idx': idx,
                'error': str(e),
                'framework': 'baseline'
            }
    
    def run_multi_trace(self, idx: int) -> Dict[str, Any]:
        """Run Multi-Trace ReAct (3 traces with synthesis)."""
        print(f"  • Running Multi-Trace ReAct (num_traces=3)...")
        try:
            _, multi_trace_info = fa.webthink(
                idx=idx,
                initial_prompt_template=fa.WEBTHINK_PROMPT_TEMPLATE,
                to_print=False,
                num_traces=3
            )
            
            # Extract individual trace info
            traces = multi_trace_info.get('individual_traces', [])
            trace_summaries = []
            for t in traces:
                trace_summaries.append({
                    'answer': t.get('answer'),
                    'em': t.get('em', 0),
                    'n_calls': t.get('n_calls', 0)
                })
            
            result = {
                'question_idx': multi_trace_info.get('question_idx'),
                'question_text': multi_trace_info.get('question_text'),
                'synthesized_answer': multi_trace_info.get('answer'),
                'gt_answer': multi_trace_info.get('gt_answer'),
                'em': multi_trace_info.get('em', 0),
                'f1': multi_trace_info.get('f1', 0),
                'reward': multi_trace_info.get('reward', 0),
                'n_calls': multi_trace_info.get('n_calls', 0),
                'n_badcalls': multi_trace_info.get('n_badcalls', 0),
                'num_traces_run': multi_trace_info.get('num_traces_run', 3),
                'individual_trace_summaries': trace_summaries,
                'full_traces': traces  # Store complete trace info
            }
            print(f"    ✓ Synthesized: {result['synthesized_answer']} | GT: {result['gt_answer']} | EM: {result['em']}")
            return result
            
        except Exception as e:
            print(f"    ✗ ERROR: {e}")
            return {
                'question_idx': idx,
                'error': str(e),
                'framework': 'multi_trace'
            }
    
    def run_reflexion(self, idx: int) -> Dict[str, Any]:
        """Run Multi-Trace ReAct with Reflexion."""
        print(f"  • Running Multi-Trace ReAct + Reflexion...")
        try:
            _, reflexion_info = fa.webthink_multi_trace_reflexion(
                idx=idx,
                initial_prompt_template=fa.WEBTHINK_PROMPT_TEMPLATE,
                to_print=False
            )
            
            result = {
                'question_idx': reflexion_info.get('question_idx'),
                'question_text': reflexion_info.get('question_text'),
                'answer': reflexion_info.get('answer'),
                'gt_answer': reflexion_info.get('gt_answer'),
                'em': reflexion_info.get('em', 0),
                'f1': reflexion_info.get('f1', 0),
                'reward': reflexion_info.get('reward', 0),
                'n_calls': reflexion_info.get('n_calls', 0),
                'n_badcalls': reflexion_info.get('n_badcalls', 0),
                'num_traces_run': reflexion_info.get('num_traces_run', 3),
                'reflexions': reflexion_info.get('reflexions', []),
                'framework': reflexion_info.get('framework', 'multi_trace_reflexion')
            }
            print(f"    ✓ Answer: {result['answer']} | GT: {result['gt_answer']} | EM: {result['em']}")
            return result
            
        except Exception as e:
            print(f"    ✗ ERROR: {e}")
            return {
                'question_idx': idx,
                'error': str(e),
                'framework': 'reflexion'
            }
    
    def run_all(self):
        """Run all configured experiments."""
        indices = self.select_indices()
        
        print(f"\n{'='*60}")
        print(f"🚀 Starting FEVER Experiments")
        print(f"{'='*60}")
        print(f"Model: {self.model}")
        print(f"Examples: {self.num_examples}")
        print(f"Frameworks: {', '.join(self.frameworks)}")
        print(f"Indices: {indices}")
        print(f"{'='*60}\n")
        
        for i, idx in enumerate(indices, 1):
            print(f"\n{'─'*60}")
            print(f"📝 Example {i}/{self.num_examples} (Index: {idx})")
            print(f"{'─'*60}")
            
            if 'baseline' in self.frameworks:
                result = self.run_baseline(idx)
                self.results['baseline'].append(result)
                self.save_progress()
            
            if 'multi_trace' in self.frameworks:
                result = self.run_multi_trace(idx)
                self.results['multi_trace'].append(result)
                self.save_progress()
            
            if 'reflexion' in self.frameworks:
                result = self.run_reflexion(idx)
                self.results['reflexion'].append(result)
                self.save_progress()
        
        # Generate final summary
        self.generate_summary()
        
        print(f"\n{'='*60}")
        print(f"✅ All experiments completed!")
        print(f"📊 Results saved to: {self.results_dir}")
        print(f"{'='*60}\n")
    
    def save_progress(self):
        """Save current progress (config + results)."""
        # Save config
        config_path = self.results_dir / "config.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)
        
        # Save individual framework results
        for framework, results in self.results.items():
            if results:  # Only save if we have results
                result_path = self.results_dir / f"{framework}.json"
                with open(result_path, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
    
    def generate_summary(self):
        """Generate aggregate summary statistics."""
        summary = {}
        
        for framework, results in self.results.items():
            if not results:
                continue
            
            # Filter out error entries
            valid_results = [r for r in results if 'error' not in r]
            total = len(results)
            valid = len(valid_results)
            
            if valid > 0:
                avg_em = sum(r.get('em', 0) for r in valid_results) / valid
                avg_f1 = sum(r.get('f1', 0) for r in valid_results) / valid
                total_calls = sum(r.get('n_calls', 0) for r in valid_results)
                total_badcalls = sum(r.get('n_badcalls', 0) for r in valid_results)
                success_count = sum(1 for r in valid_results if r.get('em', 0) == 1.0)
                
                summary[framework] = {
                    'total_examples': total,
                    'valid_examples': valid,
                    'error_count': total - valid,
                    'accuracy_em': round(avg_em, 4),
                    'accuracy_f1': round(avg_f1, 4),
                    'success_count': success_count,
                    'total_llm_calls': total_calls,
                    'total_bad_calls': total_badcalls,
                    'avg_calls_per_example': round(total_calls / valid, 2) if valid > 0 else 0
                }
        
        # Save summary
        summary_path = self.results_dir / "summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        # Print summary to console
        print(f"\n{'='*60}")
        print(f"📊 EXPERIMENT SUMMARY")
        print(f"{'='*60}")
        for framework, stats in summary.items():
            print(f"\n{framework.upper().replace('_', ' ')}:")
            print(f"  ✓ Valid Examples: {stats['valid_examples']}/{stats['total_examples']}")
            print(f"  📈 Accuracy (EM): {stats['accuracy_em']:.2%}")
            print(f"  🎯 Success Count: {stats['success_count']}")
            print(f"  🔢 Total LLM Calls: {stats['total_llm_calls']}")
            print(f"  📊 Avg Calls/Example: {stats['avg_calls_per_example']}")


def main():
    """Main entry point for running experiments."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run FEVER experiments with organized results")
    parser.add_argument('--model', type=str, default='gemini-2.5-flash',
                       help='Gemini model to use')
    parser.add_argument('--num-examples', type=int, default=5,
                       help='Number of examples to run')
    parser.add_argument('--frameworks', type=str, nargs='+',
                       default=['baseline', 'multi_trace'],
                       choices=['baseline', 'multi_trace', 'reflexion'],
                       help='Frameworks to run')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    runner = ExperimentRunner(
        model=args.model,
        num_examples=args.num_examples,
        frameworks=args.frameworks,
        seed=args.seed
    )
    
    runner.run_all()


if __name__ == '__main__':
    main()
