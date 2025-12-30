
import sys
import os
import json

# Mimic the path setup
script_dir = os.getcwd()
project_root = os.path.abspath(script_dir)
feverous_src = os.path.join(project_root, 'temp_feverous', 'src')
sys.path.append(feverous_src)

try:
    from feverous.database.feverous_db import FeverousDB
    from feverous.utils.wiki_page import WikiPage
except ImportError as e:
    print(f"Failed to import: {e}")
    sys.exit(1)

db_path = os.path.join(project_root, 'data', 'feverous_wikiv1.db')

if os.path.exists(db_path):
    try:
        db = FeverousDB(db_path)
        print("Successfully initialized FeverousDB")
        
        entity = "Kougoure"
        normalized = entity.replace(" ", "_")
        print(f"Loading '{entity}'...")
        doc_json = db.get_doc_json(normalized)
        
        if doc_json:
            print(f"  JSON loaded (size: {len(str(doc_json))} chars)")
            try:
                page = WikiPage(entity, doc_json)
                print("  WikiPage object created successfully")
                print(f"  Title: {page.title}")
                tables = page.get_tables()
                print(f"  Tables found: {len(tables)}")
            except Exception as e:
                print(f"  ERROR creating WikiPage: {e}")
        else:
            print(f"  NOT FOUND in DB")
            
    except Exception as e:
        print(f"Error: {e}")
