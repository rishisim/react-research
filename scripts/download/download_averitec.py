"""
Download AVeriTeC dataset from official GitHub repository and convert to JSONL format.
"""
import json
import urllib.request
from pathlib import Path

def download_averitec():
    """Download AVeriTeC dataset from GitHub and save as JSONL files."""
    print("Downloading AVeriTeC dataset from GitHub...")
    
    # GitHub URLs for the dataset
    base_url = "https://raw.githubusercontent.com/MichSchli/AVeriTeC/main/data"
    files = {
        "train": f"{base_url}/train.json",
        "dev": f"{base_url}/dev.json",  
    }
    
    # Create data directory if it doesn't exist
    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True)
    
    # Process each split
    for split_name, url in files.items():
        try:
            print(f"Downloading {split_name} split from {url}...")
            with urllib.request.urlopen(url) as response:
                data = json.loads(response.read())
            
            output_file = data_dir / f"averitec_{split_name}.jsonl"
            
            print(f"Processing {split_name} split ({len(data)} examples)...")
            
            with open(output_file, 'w', encoding='utf-8') as f:
                for example in data:
                    # Convert to JSONL format matching FEVER structure
                    # AVeriTeC labels: "Supported", "Refuted", "Not Enough Evidence", "Conflicting Evidence/Cherry-picking"
                    entry = {
                        "claim": example.get("claim", ""),
                        "label": example.get("label", "NOT ENOUGH INFO"),
                        "claim_id": example.get("claim_id", ""),
                        "speaker": example.get("speaker", ""),
                        "claim_date": example.get("claim_date", ""),
                    }
                    
                    f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            
            print(f"✓ Saved {len(data)} examples to {output_file}")
        
        except Exception as e:
            print(f"✗ Error downloading {split_name}: {e}")
            continue
    
    print("\n✅ AVeriTeC dataset downloaded successfully!")
    print(f"📁 Location: {data_dir.absolute()}")

if __name__ == "__main__":
    download_averitec()
