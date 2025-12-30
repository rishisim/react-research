"""Run ReAct and Nexus on specific FEVEROUS question indices and save to dedicated file."""
import sys
import os
import json
from datetime import datetime

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'agents', 'feverous'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'temp_feverous', 'src'))

from react_agent import run_react
from nexus_wrapper import run_nexus

# Target question indices
TARGET_INDICES = [1914, 224, 901, 6948, 3902, 7684, 6981, 7298, 412, 7622, 3364, 4919, 5273]

def main():
    print("=" * 70)
    print("TARGETED FEVEROUS EXPERIMENT (Clean Run)")
    print("=" * 70)
    print(f"Questions: {TARGET_INDICES}")
    print(f"Total: {len(TARGET_INDICES)}")
    print("=" * 70)
    
    results = {
        'run_time': datetime.now().isoformat(),
        'questions': TARGET_INDICES,
        'react_results': [],
        'nexus_results': []
    }
    
    react_correct = 0
    nexus_correct = 0
    
    for i, idx in enumerate(TARGET_INDICES):
        print(f"\n[{i+1}/{len(TARGET_INDICES)}] Question Index: {idx}")
        print("-" * 50)
        
        # Run ReAct
        print("  Running ReAct...", end=" ", flush=True)
        try:
            reward, react_info = run_react(idx=idx, to_print=False)
            react_answer = react_info.get('answer', 'ERROR')
            react_em = react_info.get('em', 0)
            gt_answer = react_info.get('gt_answer', 'UNKNOWN')
            print(f"Answer={react_answer} | GT={gt_answer} | EM={react_em}")
            
            results['react_results'].append({
                'question_idx': idx,
                'answer': react_answer,
                'gt_answer': gt_answer,
                'em': react_em
            })
            react_correct += react_em
        except Exception as e:
            print(f"ERROR: {e}")
            results['react_results'].append({
                'question_idx': idx,
                'answer': 'ERROR',
                'gt_answer': 'UNKNOWN',
                'em': 0,
                'error': str(e)
            })
        
        # Run Nexus
        print("  Running Nexus...", end=" ", flush=True)
        try:
            reward_n, nexus_info = run_nexus(idx=idx, to_print=False)
            nexus_answer = nexus_info.get('answer', 'ERROR')
            nexus_gt = nexus_info.get('gt_answer', gt_answer)
            nexus_em = nexus_info.get('em', 0)
            print(f"Answer={nexus_answer} | GT={nexus_gt} | EM={nexus_em}")
            
            results['nexus_results'].append({
                'question_idx': idx,
                'answer': nexus_answer,
                'gt_answer': nexus_gt,
                'em': nexus_em,
                'trace': nexus_info.get('traj', '')
            })
            nexus_correct += nexus_em
        except Exception as e:
            print(f"ERROR: {e}")
            results['nexus_results'].append({
                'question_idx': idx,
                'answer': 'ERROR',
                'gt_answer': 'UNKNOWN',
                'em': 0,
                'error': str(e)
            })
    
    # Summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"ReAct:  {react_correct}/{len(TARGET_INDICES)} ({100*react_correct/len(TARGET_INDICES):.1f}%)")
    print(f"Nexus:  {nexus_correct}/{len(TARGET_INDICES)} ({100*nexus_correct/len(TARGET_INDICES):.1f}%)")
    
    results['summary'] = {
        'react_correct': react_correct,
        'nexus_correct': nexus_correct,
        'total': len(TARGET_INDICES)
    }
    
    # Save to dedicated file
    output_file = 'results/feverous/targeted_13_questions_v2.json'
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {output_file}")
    
    # Print detailed table
    print("\n" + "=" * 70)
    print("DETAILED RESULTS")
    print("=" * 70)
    print(f"{'Idx':<6} {'GT':<20} {'ReAct':<15} {'Nexus':<15}")
    print("-" * 70)
    for r_res, n_res in zip(results['react_results'], results['nexus_results']):
        r_mark = "✓" if r_res['em'] else "✗"
        n_mark = "✓" if n_res['em'] else "✗"
        print(f"{r_res['question_idx']:<6} {r_res['gt_answer']:<20} {r_res['answer']:<12}{r_mark:<3} {n_res['answer']:<12}{n_mark}")

if __name__ == "__main__":
    main()
