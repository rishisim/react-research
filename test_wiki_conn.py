
import time
import requests

HEADERS = {
    "User-Agent": "FEVERResearchBot/1.0 (https://github.com/rishisim/react-research; research purposes)"
}

def test_wiki():
    entity = "Indianapolis"
    search_url = f"https://en.wikipedia.org/w/index.php?search={entity}"
    print(f"Fetching {search_url}...")
    
    start = time.time()
    try:
        response = requests.get(search_url, headers=HEADERS, timeout=10)
        elapsed = time.time() - start
        print(f"Status Code: {response.status_code}")
        print(f"Time: {elapsed:.2f}s")
        if response.status_code != 200:
            print("Response content snippet:", response.text[:200])
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_wiki()
