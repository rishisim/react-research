import json
import os

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def print_trace(framework, data, idx):
    item = next((x for x in data if x['question_idx'] == idx), None)
    if not item:
        print(f"[{framework}] Index {idx} not found.")
        return

    print(f"\n--- {framework.upper()} (Index {idx}) ---")
    print(f"Claim: {item.get('question_text')}")
    print(f"Answer: {item.get('answer') or item.get('synthesized_answer')}")
    print(f"GT: {item.get('gt_answer')}")
    
    if framework == 'baseline':
        print(f"Trace:\n{item.get('traj')}")
    elif framework == 'multi_trace':
        print(f"Synthesized Answer: {item.get('synthesized_answer')}")
        print("Individual Trace Answers:", [t.get('answer') for t in item.get('full_traces', [])])
    elif framework == 'reflexion':
        print(f"Final Answer: {item.get('answer')}")
        print("Reflexions:")
        for i, r in enumerate(item.get('reflexions', [])):
            print(f"  R{i+1}: {r}")

def main():
    base_dir = "results/fever/20251126_201548_n15_gemini-2.5-flash"
    indices = [6566, 5398, 3410, 1981]
    
    baseline = load_json(os.path.join(base_dir, "baseline.json"))
    multi = load_json(os.path.join(base_dir, "multi_trace.json"))
    reflexion = load_json(os.path.join(base_dir, "reflexion.json"))
    
    with open('trace_analysis_output.txt', 'w', encoding='utf-8') as f:
        for idx in indices:
            f.write(f"\n{'='*80}\n")
            f.write(f"ANALYZING INDEX {idx}\n")
            f.write(f"{'='*80}\n")
            
            # Helper to redirect print to file
            def log(msg): f.write(str(msg) + "\n")
            
            # Modified print_trace to accept a logger
            def print_trace_file(framework, data, idx, logger):
                item = next((x for x in data if x['question_idx'] == idx), None)
                if not item:
                    logger(f"[{framework}] Index {idx} not found.")
                    return

                logger(f"\n--- {framework.upper()} (Index {idx}) ---")
                logger(f"Claim: {item.get('question_text')}")
                logger(f"Answer: {item.get('answer') or item.get('synthesized_answer')}")
                logger(f"GT: {item.get('gt_answer')}")
                
                if framework == 'baseline':
                    logger(f"Trace:\n{item.get('traj')}")
                elif framework == 'multi_trace':
                    logger(f"Synthesized Answer: {item.get('synthesized_answer')}")
                    logger(f"Individual Trace Answers: {[t.get('answer') for t in item.get('full_traces', [])]}")
                elif framework == 'reflexion':
                    logger(f"Final Answer: {item.get('answer')}")
                    logger("Reflexions:")
                    for i, r in enumerate(item.get('reflexions', [])):
                        logger(f"  R{i+1}: {r}")

            print_trace_file('baseline', baseline, idx, log)
            print_trace_file('multi_trace', multi, idx, log)
            print_trace_file('reflexion', reflexion, idx, log)

if __name__ == "__main__":
    main()
