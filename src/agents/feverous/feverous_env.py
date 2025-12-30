"""
FEVEROUS Environment with Table Support

This environment extends the base WikiEnv to handle FEVEROUS-specific
table lookups using the FEVEROUS Wikipedia database.

Actions:
- search[entity]: Search for a Wikipedia page
- lookup[keyword]: Find text containing keyword on current page  
- table_lookup[table_id, cell]: Look up specific table cell
- finish[answer]: Submit final verdict (SUPPORTS/REFUTES/NOT ENOUGH INFO)
"""

import os
import sys
import gym
import sqlite3
import json

# Add parent paths for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
shared_dir = os.path.abspath(os.path.join(script_dir, '../../shared'))
sys.path.append(shared_dir)

# Add temp_feverous to path for FEVEROUS utilities
project_root = os.path.abspath(os.path.join(script_dir, '../../..'))
feverous_src = os.path.join(project_root, 'temp_feverous', 'src')
sys.path.append(feverous_src)

from wikienv import WikiEnv, textSpace

# Import FEVEROUS database utilities if available
try:
    from feverous.database.feverous_db import FeverousDB
    from feverous.utils.wiki_page import WikiPage
    FEVEROUS_DB_AVAILABLE = True
except ImportError:
    print("[WARNING] FEVEROUS database utilities not available. Table lookups will be limited.")
    FEVEROUS_DB_AVAILABLE = False


class FeverousEnv(WikiEnv):
    """
    Extended WikiEnv with FEVEROUS-specific table support.
    
    Inherits from WikiEnv for basic search/lookup/finish actions,
    and adds table_lookup action for structured table access.
    """
    
    def __init__(self, db_path=None):
        """
        Initialize the FEVEROUS environment.
        
        Args:
            db_path: Path to FEVEROUS SQLite database. If None, attempts
                     to find it in standard locations.
        """
        super().__init__()
        
        # Try to find database
        if db_path is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.abspath(os.path.join(script_dir, '../../..'))
            db_path = os.path.join(project_root, 'data', 'feverous_wikiv1.db')
        
        self.db_path = db_path
        self.db = None
        self.current_wiki_page = None  # WikiPage object for structured access
        self.current_page_title = None
        
        # Initialize database connection if available
        if FEVEROUS_DB_AVAILABLE and os.path.exists(db_path):
            try:
                self.db = FeverousDB(db_path)
                print(f"[FEVEROUS] Database connected: {db_path}")
            except Exception as e:
                print(f"[WARNING] Could not connect to FEVEROUS database: {e}")
                self.db = None
        elif not os.path.exists(db_path):
            print(f"[WARNING] FEVEROUS database not found at: {db_path}")
            print("Run scripts/download_feverous_db.py to download the database.")
    
    def reset(self, seed=None, return_info=False, options=None):
        """Reset the environment."""
        result = super().reset(seed=seed, return_info=return_info, options=options)
        self.current_wiki_page = None
        self.current_page_title = None
        return result
    
    def search_step(self, entity):
        """
        Search for a Wikipedia page.
        
        If FEVEROUS database is available, searches there first for
        structured table access. Falls back to live Wikipedia API.
        """
        # First try FEVEROUS database for structured access
        if self.db is not None:
            try:
                # Database stores entity IDs with spaces, not underscores
                page_json = self.db.get_doc_json(entity)
                
                if page_json is not None:
                    self.current_wiki_page = WikiPage(entity, page_json)
                    self.current_page_title = entity
                    self.page = str(self.current_wiki_page)
                    self.obs = self.get_page_obs(self.page)
                    self.lookup_keyword = self.lookup_list = self.lookup_cnt = None
                    return
            except Exception as e:
                print(f"[DEBUG] Database lookup failed for '{entity}': {e}")
        
        # Fall back to live Wikipedia API
        super().search_step(entity)
        self.current_wiki_page = None
        self.current_page_title = entity
    
    def table_lookup_step(self, table_query):
        """
        Look up table content from current page.
        
        Args:
            table_query: Can be:
                - table index (e.g., "0" for first table)
                - table_id (e.g., "table_0")
                - cell reference (e.g., "table_0_cell_0_0")
        """
        if self.current_wiki_page is None:
            if self.page is None:
                self.obs = "No page loaded. Use search[] first to find a Wikipedia page."
            else:
                self.obs = "Page loaded without structured table access. Try using lookup[] for text search."
            return
        
        try:
            tables = self.current_wiki_page.get_tables()
            if not tables:
                self.obs = "No tables found on current page."
                return
            
            # Handle different query formats
            table_query = table_query.strip()
            
            # If just a number, get that table
            if table_query.isdigit():
                table_idx = int(table_query)
                if table_idx >= len(tables):
                    self.obs = f"Table index {table_idx} out of range. Page has {len(tables)} tables (0 to {len(tables)-1})."
                    return
                table = tables[table_idx]
                self.obs = self._format_table(table)
                return
            
            # If table_id format (e.g., "table_0")
            if table_query.startswith("table_"):
                parts = table_query.split("_")
                if len(parts) == 2 and parts[1].isdigit():
                    table_idx = int(parts[1])
                    if table_idx >= len(tables):
                        self.obs = f"Table {table_query} not found. Page has {len(tables)} tables."
                        return
                    table = tables[table_idx]
                    self.obs = self._format_table(table)
                    return
                
                # Cell reference (e.g., "table_0_cell_0_1")
                if "cell" in table_query:
                    for table in tables:
                        cell = table.get_cell(table_query)
                        if cell:
                            self.obs = f"Cell content: {str(cell)}"
                            return
                    self.obs = f"Cell {table_query} not found."
                    return
            
            # Try to find table containing the keyword
            # Try to find best matching table using token overlap
            query_tokens = [t.lower() for t in table_query.split() if len(t) > 2] # Ignore short words
            if not query_tokens:
                query_tokens = [table_query.lower()]
            
            best_table = None
            best_score = 0
            
            for i, table in enumerate(tables):
                table_str = str(table).lower()
                score = sum(1 for token in query_tokens if token in table_str)
                
                if score > best_score:
                    best_score = score
                    best_table = (i, table)
            
            if best_table and best_score > 0:
                idx, table = best_table
                self.obs = f"Found in Table {idx} (Score: {best_score}/{len(query_tokens)}):\n{self._format_table(table, query_tokens=query_tokens)}"
                return
            
            self.obs = f"'{table_query}' not found in any table. Page has {len(tables)} tables."
            
        except Exception as e:
            self.obs = f"Error accessing table: {str(e)}"
    
    def _format_table(self, table, max_rows=15, query_tokens=None):
        """Format a WikiTable for display, prioritizing relevant rows."""
        try:
            rows = table.get_rows()
            output_lines = [f"Table: {table.name}"]
            
            if hasattr(table, 'caption') and table.caption:
                output_lines.append(f"Caption: {table.caption}")
            
            if not rows:
                output_lines.append("(Empty table)")
                return "\n".join(output_lines)
            
            # If we have query tokens, score rows and pick the most relevant
            if query_tokens and len(rows) > max_rows:
                row_scores = []
                for i, row in enumerate(rows):
                    row_cells = row.get_row_cells()
                    row_str = " ".join([str(cell).lower() for cell in row_cells])
                    score = sum(1 for token in query_tokens if token in row_str)
                    row_scores.append((i, score, row))
                
                # Always include first row (header) if it exists
                selected_rows = [(0, row_scores[0][2])] if row_scores else []
                
                # Sort remaining rows by score (descending) and add top ones
                sorted_rows = sorted(row_scores[1:], key=lambda x: x[1], reverse=True)
                for idx, score, row in sorted_rows[:max_rows-1]:
                    selected_rows.append((idx, row))
                
                # Sort by original index for coherent display
                selected_rows.sort(key=lambda x: x[0])
                
                for idx, row in selected_rows:
                    row_cells = row.get_row_cells()
                    row_str = " | ".join([str(cell).strip() for cell in row_cells])
                    output_lines.append(f"Row {idx}: {row_str}")
                
                if len(rows) > max_rows:
                    output_lines.append(f"... (showing {len(selected_rows)} most relevant of {len(rows)} total rows)")
            else:
                # Simple truncation if no query tokens
                for i, row in enumerate(rows[:max_rows]):
                    row_cells = row.get_row_cells()
                    row_str = " | ".join([str(cell).strip() for cell in row_cells])
                    output_lines.append(f"Row {i}: {row_str}")
                
                if len(rows) > max_rows:
                    output_lines.append(f"... ({len(rows) - max_rows} more rows)")
            
            return "\n".join(output_lines)
        except Exception as e:
            return f"[Error formatting table: {e}]"
    
    def step(self, action):
        """
        Execute an action in the environment.
        
        Extended to handle table_lookup action in addition to standard actions.
        """
        reward = 0
        done = False
        action = action.strip()
        
        if self.answer is not None:  # already finished
            done = True
            return self.obs, reward, done, self._get_info()
        
        # Handle table_lookup action
        if action.startswith("table_lookup[") and action.endswith("]"):
            query = action[len("table_lookup["):-1]
            self.table_lookup_step(query)
        # Handle table action (shorthand)
        elif action.startswith("table[") and action.endswith("]"):
            query = action[len("table["):-1]
            self.table_lookup_step(query)
        else:
            # Delegate to parent for standard actions
            return super().step(action)
        
        self.steps += 1
        return self.obs, reward, done, self._get_info()
    
    def get_table_summary(self):
        """Get a summary of tables on the current page."""
        if self.current_wiki_page is None:
            return "No structured page loaded."
        
        tables = self.current_wiki_page.get_tables()
        if not tables:
            return "No tables on current page."
        
        summaries = []
        for i, table in enumerate(tables):
            rows = table.get_rows()
            summaries.append(f"Table {i}: {len(rows)} rows")
        
        return "\n".join(summaries)


# Convenience function to get FEVEROUS environment
def get_feverous_env(db_path=None):
    """Get a configured FEVEROUS environment."""
    return FeverousEnv(db_path=db_path)
