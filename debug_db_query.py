import sys
import sqlite3

# Force UTF-8 output
sys.stdout.reconfigure(encoding='utf-8')

conn = sqlite3.connect('data/feverous_wikiv1.db')
cursor = conn.cursor()

# Get table names
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
print('Tables:', cursor.fetchall())

# Get total count
cursor.execute("SELECT COUNT(*) FROM wiki")
print('Total docs:', cursor.fetchone())

# Search for Lokomotiv
cursor.execute("SELECT id FROM wiki WHERE id LIKE '%Lokomotiv%' LIMIT 10")
print('Lokomotiv matches:', cursor.fetchall())

# Search for Faroe
cursor.execute("SELECT id FROM wiki WHERE id LIKE '%Faroe%' LIMIT 10")
print('Faroe matches:', cursor.fetchall())

# Search for Bangor
cursor.execute("SELECT id FROM wiki WHERE id LIKE '%Bangor%City%' LIMIT 10")
print('Bangor City matches:', cursor.fetchall())

conn.close()
