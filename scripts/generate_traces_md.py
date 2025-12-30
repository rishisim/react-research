"""Generate full traces as a markdown file."""
import json
import sys
import os

# Load from main results file which has full traces
with open('results/feverous/seed42_gemini-2.5-flash/nexus.json', encoding='utf-8') as f:
    nexus_data = json.load(f)

TARGET_INDICES = [1914, 224, 901, 6948, 3902, 7684, 6981, 7298, 412, 7622, 3364, 4919, 5273]

output = []
output.append("# Nexus Full Traces (13 Targeted Questions)\n\n")

for idx in TARGET_INDICES:
    nd = next((r for r in nexus_data if r['question_idx'] == idx), None)
    if nd:
        status = "✓" if nd['em'] == 1 else "✗"
        output.append(f"## Question {idx} {status}\n\n")
        output.append(f"**GT:** {nd['gt_answer']} | **Nexus:** {nd['answer']} | **EM:** {nd['em']}\n\n")
        output.append("```\n")
        output.append(nd.get('traj', 'No trace available'))
        output.append("\n```\n\n---\n\n")

# Write to file
with open('results/feverous/targeted_nexus_traces.md', 'w', encoding='utf-8') as f:
    f.write(''.join(output))

print("Traces saved to: results/feverous/targeted_nexus_traces.md")
