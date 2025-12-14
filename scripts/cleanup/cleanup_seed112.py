"""
Cleanup script to remove failed network error entries from seed112 results
"""
import json
from pathlib import Path

results_dir = Path("results/fever/seed112_gemini-2.5-flash")

# Indices that completed successfully (before network error)
successful_indices = {1821, 4009}

# Indices that had network errors or partial failures
failed_indices = {6855, 5643, 2327, 5719, 1491, 319, 7288}

# All indices to remove from results (failed + partial)
indices_to_remove = failed_indices

# Frameworks to clean
frameworks = ['react', 'reflexion', 'majority_voting', 'cot_sc']

for framework in frameworks:
    result_file = results_dir / f"{framework}.json"
    if result_file.exists():
        with open(result_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        # Keep only successful entries
        cleaned_results = [r for r in results if r.get('question_idx') not in indices_to_remove]
        
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(cleaned_results, f, indent=2, ensure_ascii=False)
        
        print(f"Cleaned {framework}.json: {len(results)} -> {len(cleaned_results)} entries")

# Update processed indices
processed_file = results_dir / "processed_indices.json"
with open(processed_file, 'w', encoding='utf-8') as f:
    json.dump(sorted(list(successful_indices)), f, indent=2)
print(f"Updated processed_indices.json: {successful_indices}")

# Clear failed indices
failed_file = results_dir / "failed_indices.json"
with open(failed_file, 'w', encoding='utf-8') as f:
    json.dump([], f, indent=2)
print(f"Cleared failed_indices.json")

print("\nCleanup complete! Ready to resume with --num-examples 10")
