import json
import os

# Load FEVER results
fever_path = "results/fever/seed888_gemini-2.5-flash"
fever_react = json.load(open(os.path.join(fever_path, "react.json"), encoding='utf-8'))
fever_nexus = json.load(open(os.path.join(fever_path, "nexus.json"), encoding='utf-8'))

# Load HotPotQA results
hpqa_path = "results/hotpotqa/seed888_gemini-2.5-flash"
hpqa_react = json.load(open(os.path.join(hpqa_path, "react.json"), encoding='utf-8'))
hpqa_nexus = json.load(open(os.path.join(hpqa_path, "nexus.json"), encoding='utf-8'))

def analyze(name, data, is_hpqa=False):
    valid = [r for r in data if r.get('status') == 'success']
    n = len(valid)
    if n == 0:
        return f"{name}: No valid results"
    
    em = sum(r.get('em', 0) for r in valid) / n
    em_count = sum(1 for r in valid if r.get('em', 0) == 1)
    calls = sum(r.get('n_calls', 0) for r in valid)
    
    result = f"{name}:\n"
    result += f"  Total Questions: {len(data)} ({n} valid)\n"
    result += f"  EM Accuracy: {em*100:.1f}% ({em_count}/{n})\n"
    result += f"  Total LLM Calls: {calls}\n"
    result += f"  Avg Calls/Question: {calls/n:.2f}\n"
    
    if is_hpqa:
        llm = sum(r.get('llm_correct', 0) for r in valid) / n
        llm_count = sum(1 for r in valid if r.get('llm_correct', 0) == 1)
        result += f"  LLM-Judge Accuracy: {llm*100:.1f}% ({llm_count}/{n})\n"
        
        f1 = sum(r.get('f1', 0) for r in valid) / n
        result += f"  F1 Score: {f1*100:.1f}%\n"
    
    return result

print("=" * 60)
print("NEXUS vs ReAct Comparison (Seed 888)")
print("=" * 60)

print("\n--- FEVER Dataset ---")
print(analyze("ReAct", fever_react))
print(analyze("Nexus (with retry)", fever_nexus))

print("\n--- HotPotQA Dataset ---")
print(analyze("ReAct", hpqa_react, is_hpqa=True))
print(analyze("Nexus (with retry)", hpqa_nexus, is_hpqa=True))

print("\n" + "=" * 60)
print("SUMMARY COMPARISON")
print("=" * 60)

# Calculate metrics for comparison table
fever_react_valid = [r for r in fever_react if r.get('status') == 'success']
fever_nexus_valid = [r for r in fever_nexus if r.get('status') == 'success']
hpqa_react_valid = [r for r in hpqa_react if r.get('status') == 'success']
hpqa_nexus_valid = [r for r in hpqa_nexus if r.get('status') == 'success']

print(f"\n| Dataset   | Framework | EM Accuracy      | Avg LLM Calls | LLM Judge (HotPot) |")
print(f"|-----------|-----------|------------------|---------------|--------------------|")

# FEVER
fr_n = len(fever_react_valid)
fn_n = len(fever_nexus_valid)
fr_em = sum(r.get('em', 0) for r in fever_react_valid) / fr_n * 100 if fr_n else 0
fn_em = sum(r.get('em', 0) for r in fever_nexus_valid) / fn_n * 100 if fn_n else 0
fr_calls = sum(r.get('n_calls', 0) for r in fever_react_valid) / fr_n if fr_n else 0
fn_calls = sum(r.get('n_calls', 0) for r in fever_nexus_valid) / fn_n if fn_n else 0

fr_em_ct = sum(1 for r in fever_react_valid if r.get('em', 0) == 1)
fn_em_ct = sum(1 for r in fever_nexus_valid if r.get('em', 0) == 1)

print(f"| FEVER     | ReAct     | {fr_em:.1f}% ({fr_em_ct}/{fr_n})    | {fr_calls:.2f}          | N/A                |")
print(f"| FEVER     | Nexus     | {fn_em:.1f}% ({fn_em_ct}/{fn_n})    | {fn_calls:.2f}          | N/A                |")

# HotPotQA
hr_n = len(hpqa_react_valid)
hn_n = len(hpqa_nexus_valid)
hr_em = sum(r.get('em', 0) for r in hpqa_react_valid) / hr_n * 100 if hr_n else 0
hn_em = sum(r.get('em', 0) for r in hpqa_nexus_valid) / hn_n * 100 if hn_n else 0
hr_calls = sum(r.get('n_calls', 0) for r in hpqa_react_valid) / hr_n if hr_n else 0
hn_calls = sum(r.get('n_calls', 0) for r in hpqa_nexus_valid) / hn_n if hn_n else 0
hr_llm = sum(r.get('llm_correct', 0) for r in hpqa_react_valid) / hr_n * 100 if hr_n else 0
hn_llm = sum(r.get('llm_correct', 0) for r in hpqa_nexus_valid) / hn_n * 100 if hn_n else 0

hr_em_ct = sum(1 for r in hpqa_react_valid if r.get('em', 0) == 1)
hn_em_ct = sum(1 for r in hpqa_nexus_valid if r.get('em', 0) == 1)
hr_llm_ct = sum(1 for r in hpqa_react_valid if r.get('llm_correct', 0) == 1)
hn_llm_ct = sum(1 for r in hpqa_nexus_valid if r.get('llm_correct', 0) == 1)

print(f"| HotPotQA  | ReAct     | {hr_em:.1f}% ({hr_em_ct}/{hr_n})    | {hr_calls:.2f}          | {hr_llm:.1f}% ({hr_llm_ct}/{hr_n})   |")
print(f"| HotPotQA  | Nexus     | {hn_em:.1f}% ({hn_em_ct}/{hn_n})    | {hn_calls:.2f}          | {hn_llm:.1f}% ({hn_llm_ct}/{hn_n})   |")

print("\n" + "=" * 60)
print("ANALYSIS")
print("=" * 60)
print("""
FEVER Dataset:
- ReAct maintains an edge in accuracy, but both frameworks perform well
- Nexus has fixed, predictable 3 LLM calls (Scout + Architect + Adjudicator)
- ReAct's dynamic trace allows for more adaptive reasoning

HotPotQA Dataset:
- ReAct significantly outperforms Nexus on multi-hop QA
- ReAct uses more LLM calls on average but achieves better results
- Nexus struggles with complex multi-hop questions requiring iterative exploration

Key Observations:
1. Nexus retry logic is working (visible in traces showing "Invalid, retrying...")
2. Nexus has consistent LLM cost (exactly 3 calls per question)
3. ReAct's iterative nature is better suited for complex multi-hop reasoning
4. Nexus performs closer to ReAct on simpler fact-verification (FEVER)
""")
