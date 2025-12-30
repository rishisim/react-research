"""
Download SciFACT dataset directly from Allen AI S3.

Downloads:
- claims_train.jsonl, claims_dev.jsonl, claims_test.jsonl
- corpus.jsonl (5,183 scientific abstracts)

Saves to data/scifact/ directory.
"""

import os
import tarfile
import urllib.request
import shutil
from pathlib import Path

SCIFACT_URL = "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz"

def main():
    # Get project root (scripts is one level down)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    data_dir = project_root / "data" / "scifact"
    
    print(f"[INFO] Downloading SciFACT dataset...")
    print(f"[INFO] URL: {SCIFACT_URL}")
    print(f"[INFO] Save directory: {data_dir}")
    
    # Create directory
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Download tarball
    tar_path = data_dir / "data.tar.gz"
    print("\n[1/3] Downloading data.tar.gz...")
    
    urllib.request.urlretrieve(SCIFACT_URL, tar_path)
    print(f"    -> Downloaded {tar_path.stat().st_size / 1024 / 1024:.2f} MB")
    
    # Extract
    print("\n[2/3] Extracting archive...")
    with tarfile.open(tar_path, 'r:gz') as tar:
        tar.extractall(data_dir)
    
    # The tarball extracts to a 'data' subdirectory, move files up
    extracted_dir = data_dir / "data"
    if extracted_dir.exists():
        for item in extracted_dir.iterdir():
            if item.is_file():
                shutil.move(str(item), str(data_dir / item.name))
            elif item.is_dir() and item.name != "cross_validation":
                # Skip cross_validation folder
                shutil.move(str(item), str(data_dir / item.name))
        # Clean up extracted directory
        shutil.rmtree(extracted_dir)
    
    # Clean up tarball
    tar_path.unlink()
    
    print("\n[3/3] Verifying files...")
    
    # List files and count entries
    print("\n[FILES]")
    for file in sorted(data_dir.glob("*.jsonl")):
        size_kb = file.stat().st_size / 1024
        # Count lines
        with open(file, 'r', encoding='utf-8') as f:
            count = sum(1 for _ in f)
        print(f"  {file.name}: {size_kb:.1f} KB ({count} entries)")
    
    print("\n[DONE] SciFACT dataset downloaded successfully!")
    print(f"  Location: {data_dir}")

if __name__ == "__main__":
    main()
