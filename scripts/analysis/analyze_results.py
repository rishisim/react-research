import json
import os
import sys

def load_json(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return []

def main():
    base_dir = "results/fever/20251126_201548_n15_gemini-2.5-flash"
    
    baseline = load_json(os.path.join(base_dir, "baseline.json"))
    multi_trace = load_json(os.path.join(base_dir, "multi_trace.json"))
    reflexion = load_json(os.path.join(base_dir, "reflexion.json"))
    
    # Create a map by question_idx
    data = {}
    
    for r in baseline:
        idx = r['question_idx']
        if idx not in data: data[idx] = {}
        data[idx]['gt'] = r.get('gt_answer')
        data[idx]['baseline'] = r.get('answer')
        data[idx]['baseline_em'] = r.get('em')

    for r in multi_trace:
        idx = r['question_idx']
        if idx not in data: data[idx] = {}
        data[idx]['multi_trace'] = r.get('synthesized_answer')
        data[idx]['multi_trace_em'] = r.get('em')

    for r in reflexion:
        idx = r['question_idx']
        if idx not in data: data[idx] = {}
        data[idx]['reflexion'] = r.get('answer')
        data[idx]['reflexion_em'] = r.get('em')
        data[idx]['reflexions'] = r.get('reflexions', [])

    print(f"{'Idx':<6} | {'GT':<15} | {'Baseline':<15} | {'Multi':<15} | {'Reflexion':<15} | {'Status'}")
    print("-" * 90)

    over_correction_count = 0
    fix_count = 0
    
    for idx, info in data.items():
        gt = info.get('gt', 'N/A')
        base = info.get('baseline', 'N/A')
        multi = info.get('multi_trace', 'N/A')
        ref = info.get('reflexion', 'N/A')
        
        base_em = info.get('baseline_em', 0)
        multi_em = info.get('multi_trace_em', 0)
        ref_em = info.get('reflexion_em', 0)
        
        status = ""
        if ref_em == 1 and base_em == 0:
            status = "FIXED"
            fix_count += 1
        elif ref_em == 0 and base_em == 1:
            status = "BROKE"
            over_correction_count += 1
        elif ref_em == 0 and multi_em == 1:
             status = "BROKE (vs Multi)"
             over_correction_count += 1
        
        print(f"{idx:<6} | {str(gt):<15} | {str(base):<15} | {str(multi):<15} | {str(ref):<15} | {status}")

    print("-" * 90)
    print(f"Total Fixed by Reflexion: {fix_count}")
    print(f"Total Broken by Reflexion: {over_correction_count}")

if __name__ == "__main__":
    main()
