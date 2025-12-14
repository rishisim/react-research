import requests
from bs4 import BeautifulSoup

def test_search(entity, headers=None):
    print(f"Testing search for '{entity}' with headers={headers}...")
    entity_ = entity.replace(" ", "+")
    search_url = f"https://en.wikipedia.org/w/index.php?search={entity_}"
    try:
        response = requests.get(search_url, headers=headers)
        print(f"Status Code: {response.status_code}")
        
        soup = BeautifulSoup(response.text, features="html.parser")
        result_divs = soup.find_all("div", {"class": "mw-search-result-heading"})
        
        if result_divs:
            titles = [div.get_text().strip() for div in result_divs]
            print(f"Found results: {titles[:3]}")
        else:
            # Check if it redirected to a page
            page_content = [p.get_text().strip() for p in soup.find_all("p")]
            if page_content:
                print(f"Found page content (first 100 chars): {page_content[0][:100]}")
            else:
                print("No results and no page content found.")
                
    except Exception as e:
        print(f"Error: {e}")
    print("-" * 50)

# Test 1: No Headers (Current implementation)
test_search("14th Dalai Lama")

# Test 2: With User-Agent
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}
test_search("14th Dalai Lama", headers=headers)
