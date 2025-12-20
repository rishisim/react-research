
import json

def extract_trace(path, indices):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    for item in data:
        if item['question_idx'] in indices:
            print(f"=== Trace for Index {item['question_idx']} ===")
            print(f"Question: {item['question_text']}")
            print(f"Prediction: {item['answer']}")
            print(f"Ground Truth: {item['gt_answer']}")
            print("--- TRAJECTORY ---")
            print(item['traj'])
            print("="*60)

if __name__ == "__main__":
    extract_trace("results/fever/seed888_gemini-2.5-flash/nexus.json", [240, 1574])
