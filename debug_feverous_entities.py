
import sys
import os
import sqlite3

# Mimic the path setup in feverous_env.py
script_dir = os.getcwd()
project_root = os.path.abspath(script_dir)
feverous_src = os.path.join(project_root, 'temp_feverous', 'src')
sys.path.append(feverous_src)

try:
    from feverous.database.feverous_db import FeverousDB
except ImportError as e:
    print(f"Failed to import FeverousDB: {e}")
    sys.exit(1)

db_path = os.path.join(project_root, 'data', 'feverous_wikiv1.db')

if os.path.exists(db_path):
    try:
        db = FeverousDB(db_path)
        print("Successfully initialized FeverousDB")
        
        entities_to_check = [
            "Uparale", 
            "FC Lokomotiv Gorna Oryahovitsa", 
            "2014 Faroe Islands Cup", 
            "Kougoure",
            "Berlin", # Control
            "Bangor City F.C.",
            "Bangor Reserves"
        ]
        
        for entity in entities_to_check:
            normalized = entity.replace(" ", "_")
            print(f"Checking '{entity}' (normalized: '{normalized}')...")
            doc = db.get_doc_json(normalized)
            if doc:
                print(f"  FOUND: '{entity}'")
            else:
                print(f"  NOT FOUND: '{entity}'")
            
    except Exception as e:
        print(f"Error checking entities: {e}")
else:
    print("DB file does not exist.")
