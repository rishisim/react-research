"""Show full Nexus traces for targeted questions."""
import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

# Load from main results file which has full traces
with open('results/feverous/seed42_gemini-2.5-flash/nexus.json', encoding='utf-8') as f:
    nexus_data = json.load(f)

TARGET_INDICES = [1914, 224, 901, 6948, 3902, 7684, 6981, 7298, 412, 7622, 3364, 4919, 5273]

for idx in TARGET_INDICES:
    nd = next((r for r in nexus_data if r['question_idx'] == idx), None)
    if nd:
        print("="*80)
        print(f"QUESTION {idx}")
        print("="*80)
        print(f"GT: {nd['gt_answer']} | Nexus: {nd['answer']} | EM: {nd['em']}")
        print("-"*80)
        print("FULL TRACE:")
        print("-"*80)
        print(nd.get('traj', 'No trace available'))
        print("\n\n")
