
import sys
import os

# Add src to path
sys.path.append(os.path.abspath("src"))
sys.path.append(os.path.abspath("src/agents/feverous"))

from agents.feverous.feverous_env import FeverousEnv

def main():
    print("Initializing FeverousEnv...")
    env = FeverousEnv()
    
    print("\n--- Step 1: Search for '550th Strategic Missile Squadron' ---")
    env.search_step("550th Strategic Missile Squadron")
    print(f"Current Page: {env.current_page_title}")
    
    if not env.page:
        print("ERROR: Page Content is empty/None!")
        return

    print(f"Page Text Length: {len(env.page)} chars")
    
    # Check for "air offensive" (which worked)
    print("\n--- Check: 'air offensive' ---")
    if "air offensive" in env.page.lower():
        print("FOUND 'air offensive' in page text.")
        # Find the context
        start = env.page.lower().find("air offensive")
        context = env.page[max(0, start-50):min(len(env.page), start+100)]
        print(f"Context: ...{context.replace(chr(10), ' ')}...")
    else:
        print("NOT FOUND: 'air offensive'")

    # Check for "deployment" (which failed)
    print("\n--- Check: 'deployment' ---")
    if "deployment" in env.page.lower():
        print("FOUND 'deployment' in page text.")
        start = env.page.lower().find("deployment")
        context = env.page[max(0, start-50):min(len(env.page), start+100)]
        print(f"Context: ...{context.replace(chr(10), ' ')}...")
    else:
        print("NOT FOUND: 'deployment'")

    # Check for related terms that might imply deployment but use different words
    print("\n--- Check: Related terms (station, base, assign) ---")
    for term in ["stationed", "based", "assigned", "location"]:
        if term in env.page.lower():
            print(f"Found related term '{term}'")

    # Dump a snippet of the page text to see structure (newlines, etc)
    print("\n--- Page Text Snippet (First 500 chars) ---")
    try:
        print(env.page[:500])
    except UnicodeEncodeError:
        print(env.page[:500].encode('ascii', 'replace').decode('ascii'))

if __name__ == "__main__":
    main()
