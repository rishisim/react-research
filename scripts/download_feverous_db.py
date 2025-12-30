"""
FEVEROUS Wikipedia Database Download Script

Downloads the FEVEROUS Wikipedia SQLite database (~5GB compressed, ~13GB uncompressed).
The database contains all Wikipedia pages with structured table data.
"""

import os
import urllib.request
import zipfile
import sys

DB_URL = "https://fever.ai/download/feverous/feverous-wiki-pages-db.zip"
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
ZIP_PATH = os.path.join(DATA_DIR, "feverous-wiki-pages-db.zip")
DB_PATH = os.path.join(DATA_DIR, "feverous_wikiv1.db")


def download_with_progress(url, filepath):
    """Download a file with progress reporting."""
    print(f"Downloading from: {url}")
    print(f"Saving to: {filepath}")
    
    def reporthook(count, block_size, total_size):
        downloaded = count * block_size
        percent = min(100, downloaded * 100 / total_size)
        mb_downloaded = downloaded / (1024 * 1024)
        mb_total = total_size / (1024 * 1024)
        sys.stdout.write(f"\rProgress: {percent:.1f}% ({mb_downloaded:.1f} MB / {mb_total:.1f} MB)")
        sys.stdout.flush()
    
    urllib.request.urlretrieve(url, filepath, reporthook)
    print()  # New line after progress


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Check if database already exists
    if os.path.exists(DB_PATH):
        print(f"Database already exists at: {DB_PATH}")
        return
    
    # Download if zip doesn't exist
    if not os.path.exists(ZIP_PATH):
        print("Downloading FEVEROUS Wikipedia database...")
        print("Warning: This is a ~5GB download and may take a while.")
        download_with_progress(DB_URL, ZIP_PATH)
    else:
        print(f"Zip file already exists at: {ZIP_PATH}")
    
    # Extract
    print("Extracting database...")
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        # Extract only the db file
        for name in zip_ref.namelist():
            if name.endswith('.db'):
                print(f"Extracting: {name}")
                zip_ref.extract(name, DATA_DIR)
                # Rename if needed (though it usually matches the name in zip)
                extracted_path = os.path.join(DATA_DIR, name)
                if extracted_path != DB_PATH:
                    if os.path.exists(DB_PATH):
                        os.remove(DB_PATH)
                    os.rename(extracted_path, DB_PATH)
                break
    
    print(f"Database extracted to: {DB_PATH}")
    
    # Optionally remove zip file
    response = input("Remove zip file to save space? (y/n): ").strip().lower()
    if response == 'y':
        os.remove(ZIP_PATH)
        print("Zip file removed.")


if __name__ == "__main__":
    main()
