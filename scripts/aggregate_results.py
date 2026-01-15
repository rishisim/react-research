import os
import json

def report_individual_results(root_dir):
    print(f"| Dataset | Run ID | Variant | Score | Percentage |")
    print(f"|---|---|---|---|---|")

    for dataset in ['fever', 'hotpotqa']:
        dataset_path = os.path.join(root_dir, dataset)
        if not os.path.exists(dataset_path):
            continue

        # Sort runs for consistent output
        run_dirs = sorted(os.listdir(dataset_path))

        for run_dir in run_dirs:
            run_path = os.path.join(dataset_path, run_dir)
            if not os.path.isdir(run_path):
                continue

            summary_path = os.path.join(run_path, 'summary.json')
            if not os.path.exists(summary_path):
                continue

            try:
                with open(summary_path, 'r') as f:
                    data = json.load(f)

                for variant, metrics in data.items():
                    # Normalize variant names
                    variant_name = variant
                    if variant == 'baseline':
                        variant_name = 'react'
                    
                    # Extract metrics
                    success = metrics.get('success_count')
                    if success is None:
                        success = metrics.get('em_success_count', 0)
                    
                    total = metrics.get('valid_examples')
                    if total is None:
                        total = metrics.get('total_examples', 0)

                    if total > 0:
                        percentage = (success / total) * 100
                        print(f"| {dataset.upper()} | {run_dir} | {variant_name} | {success}/{total} | {percentage:.1f}% |")

            except Exception as e:
                # print(f"Error reading {summary_path}: {e}")
                pass

if __name__ == "__main__":
    results_dir = "/Users/rishisim/Documents/research/react-research/results"
    report_individual_results(results_dir)
