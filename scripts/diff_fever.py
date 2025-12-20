
import json
import sys

def compare_results(nexus_path, react_path):
    with open(nexus_path, 'r', encoding='utf-8') as f:
        nexus_data = {item['question_idx']: item for item in json.load(f)}
    with open(react_path, 'r', encoding='utf-8') as f:
        react_data = {item['question_idx']: item for item in json.load(f)}
        
    for idx_val in react_data:
        react_item = react_data.get(idx_val)
        nexus_item = nexus_data.get(idx_val, None)
        
        if not nexus_item:
            continue
            
        r_correct = react_item['em'] == 1
        n_correct = nexus_item['em'] == 1
        
        if r_correct and not n_correct:
            print(f"Index {idx_val}: ReAct Correct, Nexus Incorrect")
            print(f"Claim: {react_item['question_text']}")
            print(f"GT: {react_item['gt_answer']}")
            print(f"ReAct Answer: {react_item['answer']}")
            print(f"Nexus Answer: {nexus_item['answer']}")
            print("-" * 30)

if __name__ == "__main__":
    compare_results("results/fever/seed888_gemini-2.5-flash/nexus.json", "results/fever/seed888_gemini-2.5-flash/react.json")
