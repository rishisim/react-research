import json
import os
import sys

# Ensure we can import from src/agents/hotpotqa
sys.path.append(os.path.abspath("src/agents/hotpotqa"))
from hotpotqa_utils import llm_judge_answer

def add_judge_to_prog():
    json_path = "results/hotpotqa/prog_ca_pruning/seed42_gemini-2.5-flash/prog_ca_pruning.json"
    
    if not os.path.exists(json_path):
        print(f"File not found: {json_path}")
        return

    with open(json_path, 'r') as f:
        data = json.load(f)
        
    updated_count = 0
    
    print(f"Processing {len(data)} entries...")
    
    for entry in data:
        # Check if already has llm_correct
        if 'llm_correct' in entry:
            continue
            
        question = entry.get('question_text', '')
        # Fallback if question_text missing (wrapper usually adds it, but just in case)
        if not question and 'question' in entry:
            question = entry['question']
            
        answer = entry.get('answer', 'null')
        gt_answer = entry.get('gt_answer', '')
        
        print(f"Judging ID {entry.get('question_idx')}: Pred='{answer}' vs GT='{gt_answer}'")
        
        # Call judge
        eval_result = llm_judge_answer(question, answer, gt_answer)
        
        # Update entry
        entry['llm_correct'] = eval_result['llm_correct']
        entry['llm_explanation'] = eval_result['llm_explanation']
        
        # We optionally could log the extra tokens used for judging, but for now just saving the result is enough
        # entry['judge_tokens'] = ...
        
        updated_count += 1
        
    if updated_count > 0:
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"Updated {updated_count} entries in {json_path}")
    else:
        print("No entries needed updating.")

if __name__ == "__main__":
    add_judge_to_prog()
