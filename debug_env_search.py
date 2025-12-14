"""
Minimal reproduction script to debug the WikiEnv search functionality.

This bypasses the agent entirely and tests the environment directly.
"""

import sys
import os

# Add shared directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'shared'))

import wikienv
import wrappers

def test_raw_wikienv():
    """Test the raw WikiEnv without any wrappers."""
    print("=" * 60)
    print("TEST 1: Raw WikiEnv (no wrappers)")
    print("=" * 60)
    
    env = wikienv.WikiEnv()
    env.reset()
    
    # Test search with lowercase action (correct format)
    action = "search[Paramore]"
    print(f"\nAction: {action}")
    obs, reward, done, info = env.step(action)
    print(f"Observation (len={len(obs)}): '{obs[:500]}...' " if len(obs) > 500 else f"Observation (len={len(obs)}): '{obs}'")
    print(f"Reward: {reward}, Done: {done}")
    print(f"Page content exists: {env.page is not None}")
    if env.page:
        print(f"Page length: {len(env.page)}")
    print()

def test_fever_wrapper():
    """Test with FeverWrapper."""
    print("=" * 60)
    print("TEST 2: WikiEnv with FeverWrapper")
    print("=" * 60)
    
    env = wikienv.WikiEnv()
    env = wrappers.FeverWrapper(env, split="dev")
    
    # Reset to a specific claim
    obs = env.reset(idx=3687)
    print(f"Reset observation: {obs}")
    
    # Test search
    action = "search[Paramore]"
    print(f"\nAction: {action}")
    obs, reward, done, info = env.step(action)
    print(f"Observation (len={len(obs)}): '{obs[:500]}...' " if len(obs) > 500 else f"Observation (len={len(obs)}): '{obs}'")
    print(f"Reward: {reward}, Done: {done}")
    print()

def test_full_stack():
    """Test with all wrappers (FeverWrapper + LoggingWrapper)."""
    print("=" * 60)
    print("TEST 3: Full stack (WikiEnv + FeverWrapper + LoggingWrapper)")
    print("=" * 60)
    
    env = wikienv.WikiEnv()
    env = wrappers.FeverWrapper(env, split="dev")
    env = wrappers.LoggingWrapper(env)
    
    # Reset to a specific claim
    obs = env.reset(idx=3687)
    print(f"Reset observation: {obs}")
    
    # Test search
    action = "search[Paramore]"
    print(f"\nAction: {action}")
    obs, reward, done, info = env.step(action)
    print(f"Observation (len={len(obs)}): '{obs[:500]}...' " if len(obs) > 500 else f"Observation (len={len(obs)}): '{obs}'")
    print(f"Reward: {reward}, Done: {done}")
    print()

def test_search_step_directly():
    """Test the search_step method directly to debug the HTTP request."""
    print("=" * 60)
    print("TEST 4: Direct search_step() debugging")
    print("=" * 60)
    
    import requests
    from bs4 import BeautifulSoup
    
    entity = "Paramore"
    entity_ = entity.replace(" ", "+")
    search_url = f"https://en.wikipedia.org/w/index.php?search={entity_}"
    
    # Wikipedia requires a User-Agent header
    headers = {
        "User-Agent": "FEVERResearchBot/1.0 (research purposes)"
    }
    
    print(f"Search URL: {search_url}")
    
    response = requests.get(search_url, headers=headers)
    print(f"Response status code: {response.status_code}")
    print(f"Response URL (after redirects): {response.url}")
    print(f"Response text length: {len(response.text)}")
    
    soup = BeautifulSoup(response.text, features="html.parser")
    
    # Check if there are search result divs (mismatch case)
    result_divs = soup.find_all("div", {"class": "mw-search-result-heading"})
    print(f"Number of search result divs: {len(result_divs)}")
    
    # Get paragraphs
    paragraphs = soup.find_all("p")
    ul_elements = soup.find_all("ul")
    print(f"Number of <p> elements: {len(paragraphs)}")
    print(f"Number of <ul> elements: {len(ul_elements)}")
    
    # Check for disambiguation page
    page = [p.get_text().strip() for p in paragraphs + ul_elements]
    if any("may refer to:" in p for p in page):
        print("DISAMBIGUATION PAGE DETECTED!")
    
    # Show first few paragraphs
    print("\nFirst 3 paragraphs content:")
    for i, p in enumerate(paragraphs[:3]):
        text = p.get_text().strip()
        print(f"  [{i}] (len={len(text)}): {text[:200]}...")
    
    # Now simulate what WikiEnv.search_step does
    print("\n--- Simulating WikiEnv.search_step ---")
    env = wikienv.WikiEnv()
    env.reset()
    env.search_step(entity)
    print(f"env.obs (len={len(env.obs)}): '{env.obs[:500]}...' " if len(env.obs) > 500 else f"env.obs (len={len(env.obs)}): '{env.obs}'")
    print(f"env.page exists: {env.page is not None}")
    if env.page:
        print(f"env.page length: {len(env.page)}")


def test_different_entities():
    """Test with different search entities."""
    print("=" * 60)
    print("TEST 5: Different search entities")
    print("=" * 60)
    
    entities = [
        "Paramore",
        "Tennessee",
        "Barack Obama",
        "Python programming language",
        "Nikolaj Coster-Waldau",  # This one works in the few-shot prompt
    ]
    
    env = wikienv.WikiEnv()
    
    for entity in entities:
        env.reset()
        action = f"search[{entity}]"
        obs, reward, done, info = env.step(action)
        obs_preview = obs[:100] + "..." if len(obs) > 100 else obs
        print(f"search[{entity}]: len={len(obs)}, preview='{obs_preview}'")
    print()


if __name__ == "__main__":
    print("Debug script for WikiEnv search functionality")
    print("=" * 60)
    print()
    
    test_raw_wikienv()
    test_fever_wrapper()
    test_full_stack()
    test_search_step_directly()
    test_different_entities()
