import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

with open('results/feverous/seed42_gemini-2.5-flash/nexus.json', encoding='utf-8') as f:
    nexus_data = json.load(f)

# Find cases with evidence but NEI
examples = []
for d in nexus_data:
    if d['answer'] == 'NOT ENOUGH INFO' and d['gt_answer'] != 'NOT ENOUGH INFO':
        traj = d.get('traj', '')
        if '[Phase 3: Adjudicator]' in traj:
            adj_part = traj.split('[Phase 3: Adjudicator]')[1]
            # Check for evidence language
            if any(p in adj_part.lower() for p in ['supported', 'confirms', 'states', 'mentions', 'provides']):
                examples.append(d)

print(f"Found {len(examples)} examples with evidence but NEI answer\n")

# Show 2 detailed examples
for i, d in enumerate(examples[:2], 1):
    print("="*70)
    print(f"EXAMPLE {i}: Question {d['question_idx']}")
    print("="*70)
    print(f"CLAIM: {d.get('question_text', 'N/A')[:200]}...")
    print(f"\nGround Truth: {d['gt_answer']}")
    print(f"Nexus Answer: {d['answer']} (WRONG)")
    
    traj = d.get('traj', '')
    if '[Phase 3: Adjudicator]' in traj:
        adj = traj.split('[Phase 3: Adjudicator]')[1]
        print(f"\n--- ADJUDICATOR REASONING ---")
        print(adj[:1500])
    print("\n")
