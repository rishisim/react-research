#!/usr/bin/env python3
"""
Test script to understand how WebShop loads goals/instructions
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'webshop'))

from web_agent_site.engine.engine import load_products
from web_agent_site.engine.goal import get_goals
from web_agent_site.utils import DEFAULT_FILE_PATH, DEBUG_PROD_SIZE
import json

print("Loading products...")
all_products, product_item_dict, product_prices, attribute_to_asins = load_products(
    filepath=DEFAULT_FILE_PATH,
    num_products=DEBUG_PROD_SIZE
)

print(f"Total products loaded: {len(all_products)}")

print("Getting goals...")
goals = get_goals(all_products, product_prices)
print(f"Total goals generated: {len(goals)}")

# Check the first few goals
print("\nFirst 5 goals:")
for i, goal in enumerate(goals[:5]):
    print(f"Goal {i}: {goal['instruction_text'][:100]}...")

print("\nLet's see if there are exactly 13 unique instructions:")
unique_instructions = set()
for goal in goals:
    unique_instructions.add(goal['instruction_text'])

print(f"Number of unique instruction texts: {len(unique_instructions)}")

if len(unique_instructions) <= 20:  # If small enough, print them all
    print("\nAll unique instruction texts:")
    for i, instruction in enumerate(sorted(unique_instructions)):
        print(f"{i}: {instruction}")

# Check what happens with fixed session IDs
print(f"\nChecking fixed session access:")
print(f"goals[0]: {goals[0]['instruction_text'][:100]}...")
print(f"goals[12]: {goals[12]['instruction_text'][:100]}...")
try:
    print(f"goals[13]: {goals[13]['instruction_text'][:100]}...")
except IndexError:
    print("goals[13]: INDEX ERROR - only 13 goals available!")
