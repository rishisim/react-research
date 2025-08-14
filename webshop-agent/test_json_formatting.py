#!/usr/bin/env python3

import json

# Test data that includes newlines
test_data = {
    "step": 1,
    "action": "search[test]",
    "observation": "WebShop \nInstruction: \nTest with newlines \n[Search]",
    "reward": 0.0
}

print("=== Current JSON formatting (standard) ===")
print(json.dumps(test_data, indent=2))

print("\n=== With ensure_ascii=False ===")
print(json.dumps(test_data, indent=2, ensure_ascii=False))

print("\n=== Loading and displaying the observation field ===")
obs = test_data["observation"]
print("Raw observation:")
print(repr(obs))
print("\nFormatted observation:")
print(obs)
