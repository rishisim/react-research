import requests

urls = {
    "FEVER Train (fever.ai)": "https://fever.ai/download/fever/train.jsonl",
    "FEVER Shared Task Test (fever.ai)": "https://fever.ai/download/fever/shared_task_test.jsonl",
    "FEVER Shared Task Dev (fever.ai)": "https://fever.ai/download/fever/shared_task_dev.jsonl",
    "FEVER Paper Dev (fever.ai)": "https://fever.ai/download/fever/paper_dev.jsonl", 
    "HotPotQA Train": "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json",
    "HotPotQA Test (FullWiki)": "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_test_fullwiki_v1.json"
}

headers = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36",
    "Referer": "https://fever.ai/dataset/fever.html"
}

print("Verifying URLs...")
for name, url in urls.items():
    try:
        response = requests.head(url, headers=headers, allow_redirects=True, timeout=10)
        status = response.status_code
        if status == 405: 
             response = requests.get(url, headers=headers, stream=True, timeout=10)
             status = response.status_code
             response.close()
        
        print(f"[{status}] {name}: {url}")
        if status == 200 and response.history:
            print(f"  -> Redirected to: {response.url}")

    except Exception as e:
        print(f"[ERROR] {name}: {e}")
