import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

# Load results
with open('results/feverous/seed42_gemini-2.5-flash/nexus.json', encoding='utf-8') as f:
    nexus_data = json.load(f)

with open('results/feverous/seed42_gemini-2.5-flash/react.json', encoding='utf-8') as f:
    react_data = json.load(f)

# find cases where Nexus said NEI but GT was different
nei_wrong = [d for d in nexus_data if d.get('answer') == 'NOT ENOUGH INFO' and d.get('gt_answer') != 'NOT ENOUGH INFO']

print(f"Nexus 'NOT ENOUGH INFO' when GT was different: {len(nei_wrong)}/50")
print()

# Compare to ReAct on same questions
for d in nei_wrong[:5]:
    idx = d['question_idx']
    react_result = next((r for r in react_data if r['question_idx'] == idx), None)
    
    print(f"=== Question {idx} ===")
    print(f"GT Answer: {d['gt_answer']}")
    print(f"Nexus: NOT ENOUGH INFO (WRONG)")
    if react_result:
        print(f"ReAct: {react_result['answer']} ({'CORRECT' if react_result['em'] == 1 else 'WRONG'})")
    
    # Extract the Adjudicator reasoning from Nexus
    traj = d.get('traj', '')
    if '[Phase 3: Adjudicator]' in traj:
        adj_part = traj.split('[Phase 3: Adjudicator]')[1][:800]
        print(f"\nNexus Adjudicator Reasoning (excerpt):")
        print(adj_part)
    print("\n" + "-"*60 + "\n")
