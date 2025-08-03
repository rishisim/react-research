import requests

def get_webshop_page():
    url = "http://localhost:3000"
    try:
        response = requests.get(url)
        response.raise_for_status()
        print("[INFO]: Successfully fetched WebShop page.")
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"[ERROR]: Failed to fetch WebShop page. {e}")
        return None

def perform_action(action_url, payload):
    try:
        response = requests.post(action_url, json=payload)
        response.raise_for_status()
        print("[INFO]: Action performed successfully.")
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"[ERROR]: Failed to perform action. {e}")
        return None

if __name__ == "__main__":
    # Example usage
    page_content = get_webshop_page()
    if page_content:
        print(page_content[:500])  # Print first 500 characters of the page

    # Example action (replace with actual action URL and payload)
    action_url = "http://localhost:3000/action"
    payload = {"action": "search", "query": "laptop"}
    action_response = perform_action(action_url, payload)
    if action_response:
        print(action_response)
