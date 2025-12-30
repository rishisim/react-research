import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

with open('results/feverous/seed42_gemini-2.5-flash/react.json', encoding='utf-8') as f:
    react_data = json.load(f)

# Find questions 1914 and 4299
for idx in [1914, 4299]:
    d = next((r for r in react_data if r['question_idx'] == idx), None)
    if d:
        print("="*70)
        print(f"ReAct: Question {idx}")
        print("="*70)
        print(f"Ground Truth: {d['gt_answer']}")
        print(f"ReAct Answer: {d['answer']} ({'CORRECT' if d['em'] == 1 else 'WRONG'})")
        print(f"\n--- TRAJECTORY ---")
        
        # Show just the reasoning part (skip the prompt template)
        traj = d.get('traj', '')
        if 'Claim:' in traj:
            # Get from the actual claim onwards
            parts = traj.split('Claim:')
            if len(parts) > 2:
                relevant = 'Claim:' + parts[-1]  # Last claim section
            else:
                relevant = 'Claim:' + parts[1]
            print(relevant[:2500])
        print("\n")
