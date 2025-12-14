"""
Download FEVEROUS dataset using wget URLs from official repository.
"""
import json
import urllib.request
from pathlib import Path

def download_feverous():
    """Download FEVEROUS dataset and save as JSONL files."""
    print("Downloading FEVEROUS dataset...")
    
    # URLs from official download script
    files = {
        "train": "https://fever.ai/download/feverous/feverous_train_challenges.jsonl",
        "dev": "https://fever.ai/download/feverous/feverous_dev_challenges.jsonl",
    }
    
    # Create data directory if it doesn't exist
    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True)
    
    # Process each split
    for split_name, url in files.items():
        try:
            print(f"Downloading {split_name} from {url}...")
            
            # Download with proper headers
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response:
                content = response.read().decode('utf-8')
            
            lines = content.strip().split('\n')
            output_file = data_dir / f"feverous_{split_name}.jsonl"
            
            print(f"Processing {split_name} split ({len(lines)} examples)...")
            
            count = 0
            with open(output_file, 'w', encoding='utf-8') as f:
                for line in lines:
                    if not line.strip():
                        continue
                    try:
                        example = json.loads(line)
                        # Simplify to match FEVER format
                        entry = {
                            "claim": example.get("claim", ""),
                            "label": example.get("label", "NOT ENOUGH INFO"),
                            "id": example.get("id", ""),
                        }
                        f.write(json.dumps(entry, ensure_ascii=False) + '\n')
                        count += 1
                    except json.JSONDecodeError as e:
                        print(f"Skipping invalid JSON line: {e}")
                        continue
            
            print(f"✓ Saved {count} examples to {output_file}")
        
        except Exception as e:
            print(f"✗ Error downloading {split_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n✅ FEVEROUS dataset download complete!")
    print(f"📁 Location: {data_dir.absolute()}")

if __name__ == "__main__":
    download_feverous()
