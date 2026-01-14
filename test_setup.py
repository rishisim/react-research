#!/usr/bin/env python3
"""Quick test to verify the environment is set up correctly."""

import os
from dotenv import load_dotenv

load_dotenv()

print("=" * 50)
print("Environment Test")
print("=" * 50)

# Test 1: Check imports
print("\n1. Testing imports...")
try:
    import torch
    import spacy
    import numpy as np
    from google import genai
    print("   ✅ All core imports successful")
except ImportError as e:
    print(f"   ❌ Import failed: {e}")
    exit(1)

# Test 2: Check API key
print("\n2. Checking API key...")
api_key = os.getenv("GEMINI_API_KEY")
if api_key:
    print(f"   ✅ GEMINI_API_KEY found (starts with: {api_key[:10]}...)")
else:
    print("   ❌ GEMINI_API_KEY not found in .env")
    exit(1)

# Test 3: Quick LLM call
print("\n3. Testing Gemini API connection...")
try:
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model="gemini-2.0-flash-lite",
        contents="What is 2+2? Reply with just the number."
    )
    answer = response.text.strip()
    print(f"   ✅ API call successful! Response: {answer}")
except Exception as e:
    print(f"   ❌ API call failed: {e}")
    exit(1)

# Test 4: Check HotPotQA data
print("\n4. Checking dataset availability...")
data_file = "data/hotpot_dev_distractor_v1.json"
if os.path.exists(data_file):
    size_mb = os.path.getsize(data_file) / (1024 * 1024)
    print(f"   ✅ HotPotQA dataset found ({size_mb:.1f} MB)")
else:
    print(f"   ⚠️  HotPotQA dataset not found at {data_file}")

print("\n" + "=" * 50)
print("All tests passed! Environment is ready. 🎉")
print("=" * 50)
