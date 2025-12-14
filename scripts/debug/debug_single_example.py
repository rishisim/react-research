"""Debug script to run a single FEVER example with verbose output."""
import sys
import os
import numpy as np

# Add shared directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src/shared')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src/agents/fever')))

import fever_agent

# Set seed to match experiment
np.random.seed(2024)

# Create a temporary env to get indices
import wikienv
import wrappers
temp_env = wikienv.WikiEnv()
temp_env = wrappers.FeverWrapper(temp_env, split="dev")

# Get the same example indices as the experiment
n_examples = 15
indices = np.random.choice(len(temp_env.data), n_examples, replace=False).tolist()

print(f"Selected indices: {indices}")
print(f"\nRunning Example 1 (index {indices[0]})...")
print("="*80)

# Get ground truth
idx = indices[0]
question, answer = temp_env.reset(idx=idx, return_info=True)

print(f"\nQuestion: {question}")
print(f"Ground Truth: {answer}")
print("\n" + "="*80)
print("RUNNING BASELINE REACT...")
print("="*80 + "\n")

# Run baseline using fever_agent.webthink (num_traces=1)
reward, info = fever_agent.webthink(idx=idx, to_print=True, num_traces=1)

print("\n" + "="*80)
print("RESULT:")
print("="*80)
print(f"Model Answer: {info['answer']}")
print(f"Ground Truth: {answer}")
print(f"Correct (EM): {info['em']}")
print(f"Reward: {reward}")
print("="*80)

