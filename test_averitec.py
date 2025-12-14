"""
Quick test script for AVeriTeC integration
"""
import sys
sys.path.insert(0, 'src/agents/fever')
import fever_agent as fa

print("Testing AVeriTeC integration...")
print("="*60)

# Try to run a single trace on AVeriTeC data
try:
    reward, info = fa.webthink(idx=0, to_print=True, num_traces=1)
    print("\n" + "="*60)
    print("TEST SUCCESSFUL!")
    print(f"Claim: {info.get('question_text')}")
    print(f"Agent Answer: {info.get('answer')}")
    print(f"Ground Truth: {info.get('gt_answer')}")
    print(f"EM Score: {info.get('em')}")
    print("="*60)
except Exception as e:
    print(f"\nTEST FAILED! Error: {e}")
    import traceback
    traceback.print_exc()
