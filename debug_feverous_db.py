
import sys
import os
import sqlite3

# Mimic the path setup in feverous_env.py
script_dir = os.getcwd()
project_root = os.path.abspath(script_dir)
feverous_src = os.path.join(project_root, 'temp_feverous', 'src')
sys.path.append(feverous_src)

print(f"Added to path: {feverous_src}")
print(f"Checking if path exists: {os.path.exists(feverous_src)}")

try:
    from feverous.database.feverous_db import FeverousDB
    print("Successfully imported FeverousDB")
except ImportError as e:
    print(f"Failed to import FeverousDB: {e}")
    sys.exit(1)

db_path = os.path.join(project_root, 'data', 'feverous_wikiv1.db')
print(f"Checking DB path: {db_path}")
print(f"DB file exists: {os.path.exists(db_path)}")

if os.path.exists(db_path):
    try:
        db = FeverousDB(db_path)
        print("Successfully initialized FeverousDB")
        
        # Try a simple query
        print("Attempting a simple doc lookup for 'Berlin'...")
        doc = db.get_doc_json("Berlin")
        if doc:
            print("Successfully retrieved doc for 'Berlin'")
            print(f"Doc keys: {doc.keys()}")
        else:
            print("Doc 'Berlin' not found (might be expected if not in small DB)")
            
    except Exception as e:
        print(f"Error connecting to DB: {e}")
else:
    print("DB file does not exist, skipping connection test.")
