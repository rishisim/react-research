import sys
import os

# Force UTF-8 output
sys.stdout.reconfigure(encoding='utf-8')

# Add paths
sys.path.append('temp_feverous/src')
sys.path.append('src/agents/feverous')

from feverous_env import FeverousEnv

env = FeverousEnv()

test_entities = ['Bangor City F.C.', 'Berlin', 'Kougoure']

for entity in test_entities:
    env.search_step(entity)
    if env.current_wiki_page:
        tables = env.current_wiki_page.get_tables()
        print(f"'{entity}': FOUND - {len(tables)} tables")
    else:
        print(f"'{entity}': NOT FOUND (fallback to Wikipedia API)")
