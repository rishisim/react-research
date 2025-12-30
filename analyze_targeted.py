import json
import sys
import re

sys.stdout.reconfigure(encoding='utf-8')

# Load results
with open('results/feverous/seed42_gemini-2.5-flash/nexus.json', encoding='utf-8') as f:
    nexus_data = json.load(f)

with open('results/feverous/seed42_gemini-2.5-flash/react.json', encoding='utf-8') as f:
    react_data = json.load(f)

# Target indices
TARGET_INDICES = [1914, 224, 901, 6948, 3902, 7684, 6981, 7298, 412, 7622, 3364, 4919, 5273]

print("| Question | GT | ReAct | Nexus | Reason for Nexus Failure |")
print("|----------|-----|-------|-------|--------------------------|")

for idx in TARGET_INDICES:
    nd = next((r for r in nexus_data if r['question_idx'] == idx), None)
    rd = next((r for r in react_data if r['question_idx'] == idx), None)
    
    if not nd or not rd:
        continue
    
    gt = nd['gt_answer']
    react_status = "✓" if rd['em'] == 1 else "✗"
    nexus_status = "✓" if nd['em'] == 1 else "✗"
    
    # Extract reason for Nexus failure
    reason = "N/A (Correct)"
    if nd['em'] == 0:
        traj = nd.get('traj', '')
        nexus_answer = nd['answer']
        
        # Analyze the trace
        if '[Phase 3: Adjudicator]' in traj:
            adj = traj.split('[Phase 3: Adjudicator]')[1][:500]
            
            # Categorize the failure
            if 'NOT ENOUGH INFO' in nexus_answer.upper():
                if 'no information' in adj.lower() or 'no evidence' in adj.lower():
                    reason = "NEI: Adjudicator claimed no info"
                elif 'insufficient' in adj.lower():
                    reason = "NEI: Adjudicator said insufficient"
                else:
                    reason = "NEI: Over-conservative verdict"
            elif nexus_answer.upper() != gt.upper():
                reason = f"Wrong verdict: {nexus_answer} (GT: {gt})"
        else:
            reason = "Trace analysis unavailable"
    
    print(f"| {idx} | {gt} | {react_status} | {nexus_status} | {reason} |")
