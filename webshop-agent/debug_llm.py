#!/usr/bin/env python3

from llm import call_llm

# Test different prompts to understand the issue

print("=== Testing LLM Response Formats ===\n")

# Test 1: Simple action completion
print("1. Testing action completion:")
result1 = call_llm('Complete this action: search[coffee', stop=[']'], num_traces=1)
print(f"Result 1: {repr(result1)}")
print(f"Length: {len(result1) if result1 else 'None'}\n")

# Test 2: WebShop context
print("2. Testing with WebShop context:")
prompt2 = """
You are a WebShop agent. Complete this action:
Action: search[coffee"""
result2 = call_llm(prompt2, stop=[']'], num_traces=1)
print(f"Result 2: {repr(result2)}")
print(f"Length: {len(result2) if result2 else 'None'}\n")

# Test 3: Full example context
print("3. Testing with full example:")
prompt3 = """
Action: search[matte lipstick multiple colors under 15]
Observation: [Back to Search] Page 1...

Action: search[coffee"""
result3 = call_llm(prompt3, stop=[']'], num_traces=1)
print(f"Result 3: {repr(result3)}")
print(f"Length: {len(result3) if result3 else 'None'}\n")

# Test 4: No stop sequence
print("4. Testing without stop sequence:")
result4 = call_llm('Complete this WebShop action: search[coffee', stop=[], num_traces=1)
print(f"Result 4: {repr(result4)}")
print(f"Length: {len(result4) if result4 else 'None'}\n")

# Test 5: Different stop sequence
print("5. Testing with newline stop:")
result5 = call_llm('Complete this WebShop action: search[coffee', stop=['\n'], num_traces=1)
print(f"Result 5: {repr(result5)}")
print(f"Length: {len(result5) if result5 else 'None'}\n")
