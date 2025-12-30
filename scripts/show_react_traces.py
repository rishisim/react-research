import json

# Use seed42 which has both frameworks on same questions
with open('results/hotpotqa/seed42_gemini-2.5-flash/react.json', 'r', encoding='utf-8') as f:
    react_data = json.load(f)
    
with open('results/hotpotqa/seed42_gemini-2.5-flash/nexus.json', 'r', encoding='utf-8') as f:
    nexus_data = json.load(f)

# Get question indices from both
react_idxs = {r['question_idx'] for r in react_data}
nexus_idxs = {r['question_idx'] for r in nexus_data}

# Find intersection
common = react_idxs & nexus_idxs
print(f"ReAct has {len(react_idxs)} questions")
print(f"Nexus has {len(nexus_idxs)} questions")
print(f"Common questions: {len(common)}")

# Find questions where Nexus wrong, ReAct correct (limit to 3 examples)
count = 0
for r in react_data:
    idx = r['question_idx']
    if idx in nexus_idxs:
        nexus_r = next((n for n in nexus_data if n['question_idx'] == idx), None)
        if nexus_r and not nexus_r['em'] and r['em']:  # Nexus wrong, ReAct correct
            count += 1
            if count > 3:
                break
            print("=" * 80)
            print(f"EXAMPLE {count}: Q{idx}")
            print(f"Question: {r['question_text']}")
            print(f"GT: {r['gt_answer']}")
            print(f"ReAct Answer: {r['answer']} ✓")
            print(f"Nexus Answer: {nexus_r['answer'][:80]}...")
            print()
            print("--- ReAct Trace ---")
            print(r['traj'][:2000])
            print()
            print("--- Nexus Trace ---")
            print(nexus_r['traj'][:1500])
            print("\n\n")
