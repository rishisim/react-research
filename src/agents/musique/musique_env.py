"""
Musique Environment with Iterative Retrieval.

This environment implements an iterative retrieval strategy for the Musique dataset,
filtered from a pool of ~20 candidate paragraphs per question.

Actions:
- search[query]: Search across the 20 candidate paragraphs (returns top matching snippets)
- select[title]: Read full content of a specific paragraph
- lookup[keyword]: Find text in currently selected paragraph
- finish[answer]: Submit final answer
"""

import os
import re
import json
import collections
import string
from typing import List, Dict, Any, Tuple

class MusiqueEnv:
    def __init__(self, data_path: str = None, split: str = 'dev'):
        """
        Initialize Musique environment.
        
        Args:
            data_path: Path to jsonl data file. If None, uses default path.
            split: 'train' or 'dev' (default: 'dev')
        """
        self.split = split
        if data_path is None:
            # Default path relative to this file
            base_dir = os.path.dirname(os.path.abspath(__file__))
            data_dir = os.path.join(base_dir, '../../../data/musique_v1.0')
            filename = f"musique_ans_v1.0_{split}.jsonl"
            data_path = os.path.join(data_dir, filename)
            
        self.data_path = data_path
        self.data = self._load_data(data_path)
        
        # State
        self.current_idx = None
        self.current_item = None
        self.current_paragraphs = [] # List of dicts {title, paragraph_text, is_supporting}
        self.selected_paragraph = None # The currently "read" paragraph
        self.steps = 0
        self.done = False
        self.answer = None
        
    def _load_data(self, path: str) -> List[Dict]:
        """Load JSONL data."""
        data = []
        try:
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data.append(json.loads(line))
        except FileNotFoundError:
            print(f"[ERROR] Data file not found at {path}")
            return []
        return data

    def reset(self, idx: int = None) -> str:
        """
        Reset environment to a specific question index.
        
        Args:
            idx: Index of question in dataset
            
        Returns:
            Question string
        """
        if idx is None:
            idx = 0
            
        if idx >= len(self.data):
             return f"Index {idx} out of range (max {len(self.data)-1})"
             
        self.current_idx = idx
        self.current_item = self.data[idx]
        self.current_paragraphs = self.current_item['paragraphs']
        self.selected_paragraph = None
        self.steps = 0
        self.done = False
        self.answer = None
        
        return f"Question: {self.current_item['question']}"
        
    def _normalize_text(self, s: str) -> str:
        """Lower text and remove punctuation, articles and extra whitespace."""
        def remove_articles(text):
            return re.sub(r'\b(a|an|the)\b', ' ', text)

        def white_space_fix(text):
            return ' '.join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))

    def _score_paragraph(self, query: str, paragraph: Dict) -> float:
        """Score a paragraph relevance to query using simple token overlap."""
        q_tokens = set(self._normalize_text(query).split())
        if not q_tokens:
            return 0.0
            
        # Check title and text
        title_tokens = set(self._normalize_text(paragraph['title']).split())
        text_tokens = set(self._normalize_text(paragraph['paragraph_text']).split())
        
        # Title match is weighted higher
        title_score = len(q_tokens.intersection(title_tokens)) * 2.0
        text_score = len(q_tokens.intersection(text_tokens))
        
        return title_score + text_score

    def search_step(self, query: str) -> str:
        """
        Search 20 candidate paragraphs.
        Returns top 3 matches with snippets.
        """
        if not self.current_paragraphs:
            return "No paragraphs available to search."
            
        scores = []
        for p in self.current_paragraphs:
            score = self._score_paragraph(query, p)
            scores.append((score, p))
            
        # Sort desc by score
        scores.sort(key=lambda x: x[0], reverse=True)
        
        # Get top 3
        top_k = scores[:3]
        
        results = []
        for score, p in top_k:
            if score > 0:
                # Return title and a snippet (first 100 chars)
                snippet = p['paragraph_text'][:100] + "..."
                results.append(f"Title: {p['title']}\nSnippet: {snippet}")
        
        if not results:
            return "No relevant paragraphs found."
            
        return "\n\n".join(results)

    def select_step(self, title_query: str) -> str:
        """
        Select a paragraph by title (fuzzy match) and read full content.
        Sets self.selected_paragraph.
        """
        # Exact match first
        for p in self.current_paragraphs:
            if p['title'].lower().strip() == title_query.lower().strip():
                self.selected_paragraph = p
                return f"Paragraph: {p['title']}\nContent: {p['paragraph_text']}"
        
        # Fuzzy match
        normalized_query = self._normalize_text(title_query)
        best_match = None
        best_score = 0
        
        for p in self.current_paragraphs:
            norm_title = self._normalize_text(p['title'])
            if normalized_query in norm_title or norm_title in normalized_query:
                # Simple containment score
                score = len(set(normalized_query.split()) & set(norm_title.split()))
                if score > best_score:
                    best_score = score
                    best_match = p
                    
        if best_match:
            self.selected_paragraph = best_match
            return f"Paragraph: {best_match['title']}\nContent: {best_match['paragraph_text']}"
            
        return f"Could not find paragraph with title '{title_query}'."

    def lookup_step(self, keyword: str) -> str:
        """Lookup keyword in currently selected paragraph."""
        if not self.selected_paragraph:
            return "No paragraph selected. Use Select[title] first."
            
        text = self.selected_paragraph['paragraph_text']
        sentences = text.split('. ')
        matches = []
        
        normalized_kw = self._normalize_text(keyword)
        
        for sent in sentences:
            if normalized_kw in self._normalize_text(sent):
                matches.append(sent)
                
        if not matches:
             return "Keyword not found in current paragraph."
             
        return "\n".join(matches)

    def step(self, action: str) -> Tuple[str, float, bool, Dict]:
        """
        Execute step.
        """
        if self.done:
            return self.answer, 0.0, True, self._get_info()
            
        self.steps += 1
        reward = 0.0
        obs = ""
        
        # Parse action
        action = action.strip()
        
        if action.lower().startswith("search[") and action.endswith("]"):
            query = action[7:-1]
            obs = self.search_step(query)
            
        elif action.lower().startswith("select[") and action.endswith("]"):
            title = action[7:-1]
            obs = self.select_step(title)
            
        elif action.lower().startswith("lookup[") and action.endswith("]"):
            keyword = action[7:-1]
            obs = self.lookup_step(keyword)
            
        elif action.lower().startswith("finish[") and action.endswith("]"):
            self.answer = action[7:-1]
            self.done = True
            obs = f"Episode finished. Answer: {self.answer}"
            # Compute reward (EM)
            em = self._compute_em(self.answer, self.current_item['answer'])
            reward = float(em)
            
        else:
            obs = f"Invalid action: {action}. Valid actions: search[query], select[title], lookup[keyword], finish[answer]"

        return obs, reward, self.done, self._get_info()

    def _compute_em(self, prediction, ground_truth):
        return self._normalize_text(prediction) == self._normalize_text(ground_truth)

    def _get_info(self):
        item = self.current_item
        return {
            'question_idx': self.current_idx,
            'question': item['question'],
            'gt_answer': item['answer'],
            'answerable': item['answerable'],
            'decomposition': item['question_decomposition'],
            'steps': self.steps,
            'answer': self.answer
        }
