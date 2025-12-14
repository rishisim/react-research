import sys
import os
import json

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src/agents/fever'))
import fever_agent as fa

def run_monk_test():
    idx = 478
    print(f"Running Reflexion test for Index {idx} (Monk Claim)...")
    
    try:
        _, reflexion_info = fa.webthink_multi_trace_reflexion(
            idx=idx,
            initial_prompt_template=fa.WEBTHINK_PROMPT_TEMPLATE,
            to_print=True
        )
        
        print("\n--- Final Result ---")
        print(f"Question: {reflexion_info.get('question_text')}")
        print(f"Answer: {reflexion_info.get('answer')}")
        print(f"GT Answer: {reflexion_info.get('gt_answer')}")
        print(f"EM: {reflexion_info.get('em')}")
        
        # Check individual traces
        print("\n--- Individual Traces ---")
        for i, t in enumerate(reflexion_info.get('individual_trajectories', [])):
            print(f"Trace {i+1} Answer: {t.get('answer')}")

        # Check reflexions
        print("\n--- Reflexions ---")
        for i, r in enumerate(reflexion_info.get('reflexions', [])):
            print(f"Reflexion {i+1}: {r}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    run_monk_test()
