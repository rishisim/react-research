import json
import os

fever_path = "/Users/rishisim/Documents/research/react-research/results/fever/reflexion/seed42_gemini-2.5-flash/reflexion.json"
hotpotqa_path = "/Users/rishisim/Documents/research/react-research/results/hotpotqa/reflexion/seed42_gemini-2.5-flash/reflexion.json"

def find_tasks(file_path, dataset_name):
    print(f"Searching in {dataset_name}...")
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return

    found_count = 0
    for item in data:
        # Check for multiple trials
        # Some items might not have 'num_trials', so default to 0
        num_trials = item.get('num_trials', 0)
        
        # Check for success
        # FEVER seems to use 1.0/0.0 for reward. HotPotQA uses boolean true/false or maybe 1/0.
        # We also check 'em' (exact match) just in case.
        reward = item.get('reward')
        em = item.get('em')
        
        is_success = False
        if isinstance(reward, bool):
            is_success = reward
        elif isinstance(reward, (int, float)):
            is_success = reward == 1.0
        elif em is True:
            is_success = True
        elif isinstance(em, (int, float)) and em == 1.0:
            is_success = True
            
        # Also check the status field if present
        if item.get('status') != 'success' and is_success:
             # Just noting this, but trust reward/em more
             pass

        if num_trials > 1 and is_success:
            print(f"Found Task in {dataset_name}:")
            print(f"  ID: {item.get('question_idx')}")
            print(f"  Question: {item.get('question_text')}")
            print(f"  Num Trials: {num_trials}")
            print(f"  Reward: {reward}")
            print("-" * 20)
            found_count += 1
            if found_count >= 2: # Stop after finding a few to avoid flooding, but user asked for 2 different tasks.
                 # The user asked for "two different tasks (where they have both multiple trials and succeeded in the end) from FEVER and HotPotQA".
                 # This implies 1 from FEVER and 1 from HotPotQA, or maybe 2 from each.
                 # I will print matching ones and I can select from output.
                 # I'll limit to 5 per dataset to give options.
                 if found_count >= 5:
                     break
    
    if found_count == 0:
        print(f"No tasks found matching criteria in {dataset_name}")

find_tasks(fever_path, "FEVER")
print("=" * 40)
find_tasks(hotpotqa_path, "HotPotQA")
