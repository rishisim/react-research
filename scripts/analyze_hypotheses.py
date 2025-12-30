import json
import sys
import re

sys.stdout.reconfigure(encoding='utf-8')

# Load results
with open('results/feverous/seed42_gemini-2.5-flash/nexus.json', encoding='utf-8') as f:
    nexus_data = json.load(f)

with open('results/feverous/seed42_gemini-2.5-flash/react.json', encoding='utf-8') as f:
    react_data = json.load(f)

print("="*70)
print("ANALYSIS: Why does Nexus over-predict NOT ENOUGH INFO?")
print("="*70)

# Q1: How often does Nexus say NEI vs other answers?
nexus_nei = sum(1 for d in nexus_data if d.get('answer') == 'NOT ENOUGH INFO')
nexus_supports = sum(1 for d in nexus_data if d.get('answer') == 'SUPPORTS')
nexus_refutes = sum(1 for d in nexus_data if d.get('answer') == 'REFUTES')

react_nei = sum(1 for d in react_data if d.get('answer') == 'NOT ENOUGH INFO')
react_supports = sum(1 for d in react_data if d.get('answer') == 'SUPPORTS')
react_refutes = sum(1 for d in react_data if d.get('answer') == 'REFUTES')

print("\n1. ANSWER DISTRIBUTION:")
print(f"   Nexus: SUPPORTS={nexus_supports}, REFUTES={nexus_refutes}, NEI={nexus_nei}")
print(f"   ReAct: SUPPORTS={react_supports}, REFUTES={react_refutes}, NEI={react_nei}")

# Q2: Ground truth distribution
gt_nei = sum(1 for d in nexus_data if d.get('gt_answer') == 'NOT ENOUGH INFO')
gt_supports = sum(1 for d in nexus_data if d.get('gt_answer') == 'SUPPORTS')
gt_refutes = sum(1 for d in nexus_data if d.get('gt_answer') == 'REFUTES')

print(f"\n   Ground Truth: SUPPORTS={gt_supports}, REFUTES={gt_refutes}, NEI={gt_nei}")

# Q3: Direct comparison - same evidence, different outcomes?
print("\n" + "="*70)
print("2. SAME QUESTION COMPARISON (Both frameworks get same evidence)")
print("="*70)

# Find cases where Nexus=NEI but ReAct=correct
interesting_cases = []
for nd in nexus_data:
    idx = nd['question_idx']
    rd = next((r for r in react_data if r['question_idx'] == idx), None)
    if rd and nd['answer'] == 'NOT ENOUGH INFO' and rd['em'] == 1:
        interesting_cases.append((nd, rd))

print(f"\nCases where Nexus=NEI (wrong), ReAct=correct: {len(interesting_cases)}")

# Analyze a specific case in detail
if interesting_cases:
    nd, rd = interesting_cases[0]
    print(f"\n--- Detailed Case: Question {nd['question_idx']} ---")
    print(f"GT: {nd['gt_answer']}")
    print(f"Nexus: {nd['answer']} | ReAct: {rd['answer']}")
    
    # Count evidence pieces for Nexus
    traj = nd.get('traj', '')
    scout_searches = traj.count('Scout Search[')
    bridge_attempts = traj.count('Bridge Attempt')
    print(f"\nNexus Evidence Gathering: {scout_searches} scout searches, {bridge_attempts} bridge attempts")
    
    # Count steps for ReAct
    react_traj = rd.get('traj', '')
    react_actions = len(re.findall(r'Action \d+:', react_traj))
    print(f"ReAct Steps: {react_actions} actions")

# Q4: Analyzing when both have same evidence
print("\n" + "="*70)
print("3. HYPOTHESIS TEST: Is it strictness or evidence?")
print("="*70)

# Categorize Nexus failures
nei_no_evidence = 0  # Nexus failed to get useful evidence
nei_has_evidence_strict = 0  # Nexus got evidence but was too strict

for nd in nexus_data:
    if nd['answer'] == 'NOT ENOUGH INFO' and nd['gt_answer'] != 'NOT ENOUGH INFO':
        traj = nd.get('traj', '')
        # Check if Adjudicator mentions finding evidence
        if '[Phase 3: Adjudicator]' in traj:
            adj_part = traj.split('[Phase 3: Adjudicator]')[1]
            # Look for patterns indicating evidence was found
            if any(phrase in adj_part.lower() for phrase in ['supported', 'confirms', 'states that', 'mentions', 'provides information']):
                nei_has_evidence_strict += 1
            else:
                nei_no_evidence += 1

print(f"\nNexus NEI failures breakdown:")
print(f"  - Had relevant evidence but too strict: {nei_has_evidence_strict}")
print(f"  - Lacked sufficient evidence: {nei_no_evidence}")

# Summary
print("\n" + "="*70)
print("CONCLUSION")
print("="*70)
print(f"""
Based on analysis:

1. ANSWER DISTRIBUTION BIAS:
   - Nexus predicts NEI {nexus_nei} times (vs GT NEI={gt_nei})
   - ReAct predicts NEI {react_nei} times
   - Nexus is HEAVILY biased toward NEI

2. WHEN BOTH HAVE EVIDENCE:
   - {len(interesting_cases)} cases where Nexus=NEI but ReAct got it correct
   - This suggests Nexus IS stricter even with similar evidence

3. EVIDENCE vs STRICTNESS:
   - {nei_has_evidence_strict} failures: Had evidence but adjudicator was too strict
   - {nei_no_evidence} failures: Genuinely lacked evidence
""")
