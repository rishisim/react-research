import json

react = json.load(open('results/feverous/seed42_gemini-2.5-flash/react.json', encoding='utf-8'))
nexus = json.load(open('results/feverous/seed42_gemini-2.5-flash/nexus.json', encoding='utf-8'))

print("| Question Index | ReAct | Nexus | Reason for Nexus Failure |")
print("|---|---|---|---|")

for r, n in zip(react, nexus):
    idx = r.get('question_idx', 'N/A')
    rem = int(r.get('em', 0))
    nem = int(n.get('em', 0))
    
    react_status = "Success" if rem == 1 else "Fail"
    nexus_status = "Success" if nem == 1 else "Fail"
    
    # Determine failure reason for Nexus
    reason = "-"
    if nem == 0:
        traj = n.get('traj', '')
        gt = n.get('gt_answer', '')
        ans = n.get('answer', '')
        
        if ans == 'NOT ENOUGH INFO' and gt != 'NOT ENOUGH INFO':
            if 'not found' in traj.lower() or 'no tables' in traj.lower():
                reason = "TableLookup returned empty; insufficient evidence"
            elif 'Status: GAP' in traj:
                reason = "Identified gap but bridge query failed to retrieve needed data"
            elif 'Status: RESOLVED' in traj:
                reason = "Thought resolved but evidence was insufficient for correct answer"
            else:
                reason = "Scout/Architect phase missed key evidence"
        elif ans != gt:
            if 'Status: RESOLVED' in traj:
                reason = "Premature resolution; jumped to wrong conclusion"
            else:
                reason = "Misinterpreted available evidence"
    
    print(f"| {idx} | {react_status} | {nexus_status} | {reason} |")
