import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

with open('results/feverous/seed42_gemini-2.5-flash/nexus.json', encoding='utf-8') as f:
    nexus_data = json.load(f)

# Find question 4299
d = next((r for r in nexus_data if r['question_idx'] == 4299), None)
if d:
    print("="*70)
    print(f"NEXUS FULL TRACE: Question 4299")
    print("="*70)
    print(f"Ground Truth: {d['gt_answer']}")
    print(f"Nexus Answer: {d['answer']}")
    print(f"EM Score: {d['em']}")
    print(f"\n{'='*70}")
    print("FULL TRAJECTORY:")
    print("="*70)
    print(d.get('traj', 'No trajectory found'))
