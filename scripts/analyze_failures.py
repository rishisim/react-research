import json
import os

FEVER_PATH = r"c:\Users\rishi\Documents\summer-research\react-research\results\fever\seed112_gemini-2.5-flash\react.json"
HOTPOT_PATH = r"c:\Users\rishi\Documents\summer-research\react-research\results\hotpotqa\seed42_gemini-2.5-flash\react.json"

def analyze_fever(filepath):
    print(f"Analyzing FEVER: {filepath}")
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    failures = {
        "Retrieval Bottleneck": 0,
        "Cautiousness/NEI Fallback": 0,
        "Reasoning Error": 0,
        "Hallucination": 0,
        "Total Failures": 0
    }
    
    total_examples = len(data)
    
    for item in data:
        # Determine correctness (FEVER uses 0/1 for em/reward)
        is_correct = item.get('em') == 1 or item.get('f1') == 1
        if is_correct:
            continue
            
        failures["Total Failures"] += 1
        gt = str(item.get('gt_answer', '')).strip().upper()
        pred = str(item.get('answer', '')).strip().upper()
        trace = item.get('traj', '')
        
        # Heuristics
        if "Could not find" in trace:
             failures["Retrieval Bottleneck"] += 1
        elif pred == "NOT ENOUGH INFO" and gt != "NOT ENOUGH INFO":
             failures["Cautiousness/NEI Fallback"] += 1
        elif gt == "NOT ENOUGH INFO" and pred != "NOT ENOUGH INFO":
             failures["Hallucination"] += 1 
        else:
             failures["Reasoning Error"] += 1
             
    return failures, total_examples

def analyze_hotpot(filepath):
    print(f"Analyzing HotPotQA: {filepath}")
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    failures = {
        "Exact Match Sensitivity": 0,
        "Retrieval Failure": 0,
        "Premature Termination": 0,
        "Bridge Linkage Failure": 0, 
        "Total Failures": 0
    }
    
    total_examples = len(data)
    
    for item in data:
        is_correct = item.get('em') is True or item.get('f1') == 1.0 or item.get('reward') is True
        if is_correct:
            continue
            
        failures["Total Failures"] += 1
        gt = str(item.get('gt_answer', '')).strip().lower()
        pred = str(item.get('answer', '')).strip().lower()
        trace = item.get('traj', '')
        
        # Exact Match Sensitivity
        if pred and gt and (pred in gt or gt in pred) and pred != gt:
            failures["Exact Match Sensitivity"] += 1
            continue
            
        # Retrieval Failure
        if "Could not find" in trace:
             failures["Retrieval Failure"] += 1
             continue
             
        # Premature Termination
        steps = trace.count("Action ") 
        if steps <= 2: 
            failures["Premature Termination"] += 1
            continue
            
        # Default to Bridge Linkage Failure
        failures["Bridge Linkage Failure"] += 1

    return failures, total_examples

if __name__ == "__main__":
    if os.path.exists(FEVER_PATH):
        try:
            f_fails, f_total = analyze_fever(FEVER_PATH)
            print("\n=== FEVER RESULTS ===")
            print(f"Total Examples: {f_total}")
            print(f"Total Failures: {f_fails['Total Failures']}")
            for k, v in f_fails.items():
                if k == "Total Failures": continue
                pct = (v / f_fails["Total Failures"]) * 100 if f_fails["Total Failures"] > 0 else 0
                print(f"{k}: {v} ({pct:.1f}%)")
        except Exception as e:
            print(f"Error analyzing FEVER: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"FEVER file not found: {FEVER_PATH}")

    if os.path.exists(HOTPOT_PATH):
        try:
            h_fails, h_total = analyze_hotpot(HOTPOT_PATH)
            print("\n=== HOTPOTQA RESULTS ===")
            print(f"Total Examples: {h_total}")
            print(f"Total Failures: {h_fails['Total Failures']}")
            for k, v in h_fails.items():
                if k == "Total Failures": continue
                pct = (v / h_fails["Total Failures"]) * 100 if h_fails["Total Failures"] > 0 else 0
                print(f"{k}: {v} ({pct:.1f}%)")
        except Exception as e:
            print(f"Error analyzing HotPotQA: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"HotPotQA file not found: {HOTPOT_PATH}")
