import os
import requests
from pathlib import Path
import sys

def download_file(url, valid_path):
    """Download a file from a URL to a local path with progress indication."""
    try:
        print(f"Downloading {url} to {valid_path}...")
        
        # Ensure directory exists
        valid_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Headers to mimic a browser to avoid 403s
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36",
            "Referer": "https://fever.ai/"
        }
        
        response = requests.get(url, headers=headers, stream=True, timeout=30)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 8192
        downloaded = 0
        
        with open(valid_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=block_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = downloaded * 100 / total_size
                        sys.stdout.write(f"\rProgress: {percent:.1f}% ({downloaded}/{total_size} bytes)")
                        sys.stdout.flush()
        
        print(f"\n✓ Successfully downloaded to {valid_path}")
        return True
        
    except Exception as e:
        print(f"\n✗ Error downloading {url}: {e}")
        if valid_path.exists():
            valid_path.unlink() # Remove partial file
        return False

def main():
    project_root = Path(__file__).resolve().parents[2]
    data_dir = project_root / "data"
    
    datasets = [
        # FEVER
        {
            "name": "FEVER Train",
            "url": "https://fever.ai/download/fever/train.jsonl",
            "path": data_dir / "fever" / "train.jsonl"
        },
        {
            "name": "FEVER Test",
            "url": "https://fever.ai/download/fever/shared_task_test.jsonl",
            "path": data_dir / "fever" / "shared_task_test.jsonl"
        },
        # HotPotQA
        {
            "name": "HotPotQA Train",
            "url": "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json",
            "path": data_dir / "hotpotqa" / "hotpot_train_v1.1_simplified.json"
        },
        {
            "name": "HotPotQA Test (FullWiki)",
            "url": "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_test_fullwiki_v1.json",
            "path": data_dir / "hotpotqa" / "hotpot_test_v1_simplified.json"
        },
        # FEVEROUS
        {
            "name": "FEVEROUS Test (Unlabeled)",
            "url": "https://fever.ai/download/feverous/feverous_test_unlabeled.jsonl",
            "path": data_dir / "feverous" / "feverous_test_unlabeled.jsonl"
        }
    ]
    
    print(f"Starting download of {len(datasets)} files...")
    
    success_count = 0
    for ds in datasets:
        if ds["path"].exists() and ds["path"].stat().st_size > 0:
            print(f"Skipping {ds['name']} - already exists at {ds['path']}")
            success_count += 1
            continue
            
        if download_file(ds["url"], ds["path"]):
            success_count += 1
            
    print(f"\nFinished! Successfully downloaded {success_count}/{len(datasets)} files.")

if __name__ == "__main__":
    main()
