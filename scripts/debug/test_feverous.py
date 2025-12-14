"""
Quick test of FEVEROUS integration
"""
import sys
sys.path.append('src')

from agents.fever import fever_agent

# Test with a single example
print("Testing FEVEROUS integration...")
print("=" * 60)

reward, result = fever_agent.webthink(
    idx=10,  # Test with example 10
    to_print=True
)

print("\n" + "=" * 60)
print(f"Question: {result['question_text']}")
print(f"Ground Truth: {result.get('gt_answer', 'N/A')}")
print(f"Prediction: {result.get('answer', 'N/A')}")
print(f"EM Score: {result.get('em', reward)}")
print("=" * 60)
