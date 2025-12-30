import json

# Load the react and nexus results
with open('results/feverous/seed42_gemini-2.5-flash/react.json', 'r', encoding='utf-8') as f:
    react_results = json.load(f)

with open('results/feverous/seed42_gemini-2.5-flash/nexus.json', 'r', encoding='utf-8') as f:
    nexus_results = json.load(f)

# Target indices
TARGET_INDICES = [1914, 224, 901, 6948, 3902, 7684, 6981, 7298, 412, 7622, 3364, 4919, 5273]

# Create lookup dictionaries
react_lookup = {r['question_idx']: r for r in react_results}
nexus_lookup = {r['question_idx']: r for r in nexus_results}

# Build combined results
combined = []
for idx in TARGET_INDICES:
    react = react_lookup.get(idx, {})
    nexus = nexus_lookup.get(idx, {})
    combined.append({
        'question_idx': idx,
        'claim': react.get('question_text', nexus.get('question_text', '')),
        'gt_answer': react.get('gt_answer', nexus.get('gt_answer', '')),
        'react_answer': react.get('answer', ''),
        'react_em': react.get('em', 0),
        'react_calls': react.get('n_calls', 0),
        'react_trace': react.get('traj', ''),
        'nexus_answer': nexus.get('answer', ''),
        'nexus_em': nexus.get('em', 0),
        'nexus_calls': nexus.get('n_calls', 0),
        'nexus_trace': nexus.get('traj', '')
    })

# Save to new file
with open('results/feverous/targeted_13_v3.json', 'w', encoding='utf-8') as f:
    json.dump(combined, f, indent=2, ensure_ascii=False)

print(f'Saved {len(combined)} results with traces and call counts')
for r in combined:
    print(f"  {r['question_idx']}: ReAct={r['react_calls']} calls, Nexus={r['nexus_calls']} calls")
