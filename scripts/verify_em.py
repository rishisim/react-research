import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

# Load results
with open('results/feverous/seed42_gemini-2.5-flash/nexus.json', encoding='utf-8') as f:
    nexus_data = json.load(f)

with open('results/feverous/seed42_gemini-2.5-flash/react.json', encoding='utf-8') as f:
    react_data = json.load(f)

# Target indices
TARGET_INDICES = [1914, 224, 901, 6948, 3902, 7684, 6981, 7298, 412, 7622, 3364, 4919, 5273]

print("Raw EM values from result files:")
print("-" * 60)

react_total = 0
nexus_total = 0

for idx in TARGET_INDICES:
    nd = next((r for r in nexus_data if r['question_idx'] == idx), None)
    rd = next((r for r in react_data if r['question_idx'] == idx), None)
    
    if nd and rd:
        r_em = rd.get('em', 0)
        n_em = nd.get('em', 0)
        react_total += r_em
        nexus_total += n_em
        print(f"Index {idx}: GT={nd['gt_answer'][:15]:<15} | ReAct EM={r_em} | Nexus EM={n_em}")
    else:
        print(f"Index {idx}: NOT FOUND")

print("-" * 60)
print(f"ReAct Total: {react_total}/13")
print(f"Nexus Total: {nexus_total}/13")
