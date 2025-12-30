import json
import os

RESULTS_DIR = "results/feverous/seed42_gemini-2.5-flash"

def analyze():
    print(f"Loading results from {RESULTS_DIR}...")
    try:
        with open(os.path.join(RESULTS_DIR, "react.json"), 'r', encoding='utf-8') as f:
            react_data = json.load(f)
        with open(os.path.join(RESULTS_DIR, "nexus.json"), 'r', encoding='utf-8') as f:
            nexus_data = json.load(f)
    except FileNotFoundError:
        print("Error: Files not found.")
        return

    # Convert to dict for easy lookup by index
    r_map = {x['question_idx']: x for x in react_data}
    n_map = {x['question_idx']: x for x in nexus_data}
    
    indices = sorted(list(r_map.keys()))
    
    react_correct = 0
    nexus_correct = 0
    both_correct = 0
    both_wrong = 0
    react_only = 0
    nexus_only = 0
    
    print("\nCOMPARISON (ReAct vs Nexus):")
    print(f"{'Idx':<6} | {'GT':<15} | {'ReAct':<15} | {'Nexus':<15} | {'R ok':<4} | {'N ok':<4}")
    print("-" * 80)
    
    for idx in indices:
        r = r_map[idx]
        n = n_map.get(idx, {})
        
        gt = r['gt_answer']
        r_ans = r.get('answer', 'N/A')
        n_ans = n.get('answer', 'N/A')
        
        r_em = r.get('em', 0)
        n_em = n.get('em', 0)
        
        if r_em: react_correct += 1
        if n_em: nexus_correct += 1
        
        if r_em and n_em: both_correct += 1
        if not r_em and not n_em: both_wrong += 1
        if r_em and not n_em: react_only += 1
        if not r_em and n_em: nexus_only += 1
        
        print(f"{idx:<6} | {gt[:15]:<15} | {r_ans[:15]:<15} | {n_ans[:15]:<15} | {str(r_em):<4} | {str(n_em):<4}")

    print("-" * 80)
    print(f"Total: {len(indices)}")
    print(f"ReAct Accuracy: {react_correct}/{len(indices)} ({react_correct/len(indices):.1%})")
    print(f"Nexus Accuracy: {nexus_correct}/{len(indices)} ({nexus_correct/len(indices):.1%})")
    print("-" * 80)
    print(f"Both Correct: {both_correct}")
    print(f"Both Wrong: {both_wrong}")
    print(f"ReAct Wins (Nexus failed): {react_only}")
    print(f"Nexus Wins (ReAct failed): {nexus_only}")
    
    # Analyze Nexus Failures
    print("\n\nANALYSIS OF NEXUS FAILURES (Where ReAct succeeded):")
    for idx in indices:
        r = r_map[idx]
        n = n_map[idx]
        
        if r['em'] == 1 and n['em'] == 0:
            print(f"\nExample {idx}: GT={r['gt_answer']}")
            print(f"ReAct Answer: {r['answer']} (Correct)")
            print(f"Nexus Answer: {n['answer']} (Wrong)")
            print(f"Nexus Trace Summary:")
            # Extract key steps from trace if available
            traj = n.get('traj', '')
            if traj:
                print(traj[:1000] + "..." if len(traj) > 1000 else traj)
            else:
                print("No trace available.")

if __name__ == "__main__":
    analyze()
