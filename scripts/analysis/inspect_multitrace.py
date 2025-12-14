import json
import os

# Get latest results directory
results_dir = 'results/fever'
# Filter for directories starting with 20251125_21
dirs = [d for d in os.listdir(results_dir) if d.startswith('20251125_21')]
if not dirs:
    print("No results found.")
    exit()
    
latest = sorted(dirs, reverse=True)[0]
path = os.path.join(results_dir, latest, 'multi_trace.json')

print(f"Analyzing: {path}")
if not os.path.exists(path):
    print("multi_trace.json not found yet.")
    exit()

data = json.load(open(path, encoding='utf-8'))

print(f"Total Examples: {len(data)}")
for i, d in enumerate(data):
    print(f"\nExample {i+1}:")
    print(f"  Question: {d.get('question_text', 'N/A')[:50]}...")
    print(f"  Synthesized Answer: {d.get('synthesized_answer')}")
    print(f"  Ground Truth: {d.get('gt_answer')}")
    print(f"  Match (EM): {d.get('em')}")
