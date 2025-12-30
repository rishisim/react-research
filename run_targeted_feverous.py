"""Run ReAct and Nexus on specific FEVEROUS question indices."""
import sys
import os

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'agents', 'feverous'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'temp_feverous', 'src'))

import json
from datasets import load_dataset
from react_agent import run_react
from nexus_wrapper import run_nexus

# Target question indices
TARGET_INDICES = [1914, 224, 901, 6948, 3902, 7684, 6981, 7298, 412, 7622, 3364, 4919, 5273]

def main():
    print("Loading FEVEROUS dataset...")
    dataset = load_dataset("feverous/feverous", split="dev", trust_remote_code=True)
    
    results = []
    
    for i, idx in enumerate(TARGET_INDICES):
        print(f"\n{'='*70}")
        print(f"[{i+1}/{len(TARGET_INDICES)}] Question Index: {idx}")
        print("="*70)
        
        # Get the question
        example = dataset[idx]
        claim = f"Claim: {example['claim']}"
        gt_label = example['label']
        gt_map = {0: 'NOT ENOUGH INFO', 1: 'SUPPORTS', 2: 'REFUTES'}
        gt_answer = gt_map.get(gt_label, str(gt_label))
        
        print(f"Claim: {example['claim'][:100]}...")
        print(f"Ground Truth: {gt_answer}")
        
        # Run ReAct
        print("\nRunning ReAct...")
        try:
            reward, react_info = run_react(idx=idx, to_print=False)
            react_answer = react_info.get('answer', 'ERROR')
            react_em = react_info.get('em', 0)
            print(f"  ReAct Answer: {react_answer} | EM: {react_em}")
        except Exception as e:
            react_answer = "ERROR"
            react_em = 0
            print(f"  ReAct Error: {e}")
        
        # Run Nexus
        print("Running Nexus...")
        try:
            nexus_result = run_nexus(claim)
            nexus_answer = nexus_result.get('answer', 'ERROR')
            nexus_em = 1 if nexus_answer.upper() == gt_answer.upper() else 0
            nexus_trace = nexus_result.get('traj', '')
            print(f"  Nexus Answer: {nexus_answer} | EM: {nexus_em}")
        except Exception as e:
            nexus_answer = "ERROR"
            nexus_em = 0
            nexus_trace = str(e)
            print(f"  Nexus Error: {e}")
        
        results.append({
            'question_idx': idx,
            'claim': example['claim'],
            'gt_answer': gt_answer,
            'react_answer': react_answer,
            'react_em': react_em,
            'nexus_answer': nexus_answer,
            'nexus_em': nexus_em,
            'nexus_trace': nexus_trace
        })
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    react_correct = sum(r['react_em'] for r in results)
    nexus_correct = sum(r['nexus_em'] for r in results)
    total = len(results)
    
    print(f"ReAct: {react_correct}/{total} ({100*react_correct/total:.1f}%)")
    print(f"Nexus: {nexus_correct}/{total} ({100*nexus_correct/total:.1f}%)")
    
    print("\nDetailed Results:")
    print(f"{'Idx':<6} {'GT':<20} {'ReAct':<20} {'Nexus':<20}")
    print("-"*70)
    for r in results:
        react_mark = "✓" if r['react_em'] else "✗"
        nexus_mark = "✓" if r['nexus_em'] else "✗"
        print(f"{r['question_idx']:<6} {r['gt_answer']:<20} {r['react_answer']:<15}{react_mark:<5} {r['nexus_answer']:<15}{nexus_mark}")
    
    # Save results
    output_file = 'targeted_feverous_results.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    main()

