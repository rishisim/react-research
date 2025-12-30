import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

with open('results/feverous/seed42_gemini-2.5-flash/nexus.json', encoding='utf-8') as f:
    nexus_data = json.load(f)

with open('results/feverous/seed42_gemini-2.5-flash/react.json', encoding='utf-8') as f:
    react_data = json.load(f)

# Find cases with evidence but NEI
examples = []
for d in nexus_data:
    if d['answer'] == 'NOT ENOUGH INFO' and d['gt_answer'] != 'NOT ENOUGH INFO':
        traj = d.get('traj', '')
        if '[Phase 3: Adjudicator]' in traj:
            adj_part = traj.split('[Phase 3: Adjudicator]')[1]
            if any(p in adj_part.lower() for p in ['supported', 'confirms', 'states', 'mentions', 'provides']):
                examples.append(d)

# Show examples 3 and 4 (skip 1 and 2 already shown)
for i, d in enumerate(examples[2:4], 3):
    idx = d['question_idx']
    rd = next((r for r in react_data if r['question_idx'] == idx), None)
    
    print("="*70)
    print(f"EXAMPLE {i}: Question {idx}")
    print("="*70)
    print(f"CLAIM: {d.get('question_text', 'N/A')}")
    print(f"\nGround Truth: {d['gt_answer']}")
    print(f"Nexus Answer: {d['answer']} (WRONG)")
    if rd:
        print(f"ReAct Answer: {rd['answer']} ({'CORRECT' if rd['em'] == 1 else 'WRONG'})")
    
    # Nexus Adjudicator
    traj = d.get('traj', '')
    if '[Phase 3: Adjudicator]' in traj:
        adj = traj.split('[Phase 3: Adjudicator]')[1]
        print(f"\n--- NEXUS ADJUDICATOR ---")
        print(adj[:1200])
    
    # ReAct trace
    if rd:
        rtraj = rd.get('traj', '')
        if 'Claim:' in rtraj:
            parts = rtraj.split('Claim:')
            relevant = 'Claim:' + parts[-1] if len(parts) > 2 else 'Claim:' + parts[1]
            print(f"\n--- REACT TRAJECTORY ---")
            print(relevant[:1500])
    
    print("\n")
