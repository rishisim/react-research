
import json
import os

path = r"c:\Users\rishi\Documents\summer-research\react-research\results\feverous\seed42_gemini-2.5-flash\nexus.json"

try:
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    print(f"Loaded {len(data)} entries.")
    
    found = False
    for entry in data:
        if entry.get("question_idx") == 7684 or "Kougoure" in str(entry):
            print(f"FAILED ENTRY FOUND: {entry['question_idx']}")
            print(f"Question: {entry.get('question_text')}")
            print("--- TRAJ START ---")
            print(entry.get('traj'))
            print("--- TRAJ END ---")
            found = True
            
    if not found:
        print("Entry not found.")
        
except Exception as e:
    print(f"Error: {e}")
