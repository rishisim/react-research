"""
SciFACT Environment for Corpus-Based Search

Implements a gym-like environment for searching scientific abstracts
using TF-IDF retrieval (k=3), following the VERISCI baseline.

Actions:
- search[query] -> Returns top-3 matching abstracts
- lookup[keyword] -> Finds sentences containing keyword in current abstract
- finish[answer] -> Submits final verdict (SUPPORT/REFUTE/NOT ENOUGH INFO)
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class SciFACTEnv:
    """
    SciFACT environment for scientific claim verification.
    Uses TF-IDF search over a corpus of scientific abstracts.
    """
    
    def __init__(self, data_dir: str = None):
        """
        Initialize the SciFACT environment.
        
        Args:
            data_dir: Path to scifact data directory (default: data/scifact/)
        """
        if data_dir is None:
            # Default path relative to this file
            script_dir = Path(__file__).parent
            data_dir = script_dir.parent.parent.parent / "data" / "scifact"
        else:
            data_dir = Path(data_dir)
        
        self.data_dir = data_dir
        
        # Load corpus and build TF-IDF index
        self.corpus = self._load_corpus()
        self.claims = self._load_claims()
        self._build_tfidf_index()
        
        # Environment state
        self.current_claim_idx = None
        self.current_claim = None
        self.current_abstract = None  # Currently viewed abstract (for lookup)
        self.current_abstract_sentences = None
        self.lookup_keyword = None
        self.lookup_list = None
        self.lookup_cnt = 0
        self.steps = 0
        self.answer = None
        self.obs = None
    
    def _load_corpus(self) -> Dict[int, Dict]:
        """Load the corpus of scientific abstracts."""
        corpus_path = self.data_dir / "corpus.jsonl"
        corpus = {}
        
        with open(corpus_path, 'r', encoding='utf-8') as f:
            for line in f:
                doc = json.loads(line.strip())
                doc_id = doc['doc_id']
                corpus[doc_id] = {
                    'title': doc.get('title', ''),
                    'abstract': doc.get('abstract', []),  # List of sentences
                    'structured': doc.get('structured', False)
                }
        
        print(f"[SciFACTEnv] Loaded {len(corpus)} abstracts")
        return corpus
    
    def _load_claims(self) -> List[Dict]:
        """Load the claims from dev set."""
        claims_path = self.data_dir / "claims_dev.jsonl"
        claims = []
        
        with open(claims_path, 'r', encoding='utf-8') as f:
            for line in f:
                claim = json.loads(line.strip())
                claims.append(claim)
        
        print(f"[SciFACTEnv] Loaded {len(claims)} claims")
        return claims
    
    def _build_tfidf_index(self):
        """Build TF-IDF index over corpus abstracts."""
        # Create document texts (title + abstract sentences joined)
        self.doc_ids = list(self.corpus.keys())
        self.doc_texts = []
        
        for doc_id in self.doc_ids:
            doc = self.corpus[doc_id]
            text = doc['title'] + ' ' + ' '.join(doc['abstract'])
            self.doc_texts.append(text)
        
        # Build TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words='english',
            max_features=10000  # Limit vocabulary size
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(self.doc_texts)
        print(f"[SciFACTEnv] Built TF-IDF index with {self.tfidf_matrix.shape[1]} features")
    
    def reset(self, idx: int = None) -> str:
        """
        Reset the environment for a new claim.
        
        Args:
            idx: Claim index to use (0-indexed into claims list)
            
        Returns:
            The claim text
        """
        if idx is None:
            idx = 0
        
        self.current_claim_idx = idx
        self.current_claim = self.claims[idx]
        self.current_abstract = None
        self.current_abstract_sentences = None
        self.lookup_keyword = None
        self.lookup_list = None
        self.lookup_cnt = 0
        self.steps = 0
        self.answer = None
        self.obs = self.current_claim['claim']
        
        return self.current_claim['claim']
    
    def _search(self, query: str, k: int = 3) -> str:
        """
        Search for abstracts matching the query using TF-IDF.
        
        Args:
            query: Search query
            k: Number of results to return (default 3 per VERISCI paper)
            
        Returns:
            Observation string with search results
        """
        # Vectorize query
        query_vec = self.vectorizer.transform([query])
        
        # Compute cosine similarity
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[-k:][::-1]
        
        # Build results
        results = []
        for i, idx in enumerate(top_indices):
            doc_id = self.doc_ids[idx]
            doc = self.corpus[doc_id]
            score = similarities[idx]
            
            # Truncate abstract to first 3 sentences for readability
            abstract_preview = ' '.join(doc['abstract'][:3])
            if len(doc['abstract']) > 3:
                abstract_preview += ' ...'
            
            results.append(f"[{i+1}] {doc['title']}\n{abstract_preview}")
            
            # Store first result as current for potential lookup
            if i == 0:
                self.current_abstract = doc
                self.current_abstract_sentences = doc['abstract']
                self.lookup_keyword = None
                self.lookup_list = None
                self.lookup_cnt = 0
        
        if not results:
            return "No matching abstracts found."
        
        return "\n\n".join(results)
    
    def _lookup(self, keyword: str) -> str:
        """
        Look up a keyword in the current abstract.
        
        Args:
            keyword: Keyword to search for
            
        Returns:
            Matching sentence or status message
        """
        if self.current_abstract_sentences is None:
            return "No abstract loaded. Use search[] first."
        
        # Reset lookup if keyword changed
        if self.lookup_keyword != keyword.lower():
            self.lookup_keyword = keyword.lower()
            self.lookup_list = [
                sent for sent in self.current_abstract_sentences
                if keyword.lower() in sent.lower()
            ]
            self.lookup_cnt = 0
        
        if not self.lookup_list:
            return f"No sentences containing '{keyword}' found in current abstract."
        
        if self.lookup_cnt >= len(self.lookup_list):
            return "No more results."
        
        result = f"(Result {self.lookup_cnt + 1} / {len(self.lookup_list)}) {self.lookup_list[self.lookup_cnt]}"
        self.lookup_cnt += 1
        return result
    
    def step(self, action: str) -> Tuple[str, float, bool, Dict]:
        """
        Execute an action in the environment.
        
        Args:
            action: Action string (search[query], lookup[keyword], or finish[answer])
            
        Returns:
            Tuple of (observation, reward, done, info)
        """
        reward = 0
        done = False
        action = action.strip()
        
        if self.answer is not None:
            done = True
            return self.obs, reward, done, self._get_info()
        
        if action.startswith("search[") and action.endswith("]"):
            query = action[7:-1]
            self.obs = self._search(query)
            
        elif action.startswith("lookup[") and action.endswith("]"):
            keyword = action[7:-1]
            self.obs = self._lookup(keyword)
            
        elif action.startswith("finish[") and action.endswith("]"):
            answer = action[7:-1]
            self.answer = self._normalize_answer(answer)
            done = True
            reward = self._compute_reward()
            self.obs = f"Episode finished, reward = {reward}"
            
        elif action.startswith("think[") and action.endswith("]"):
            self.obs = "Nice thought."
            
        else:
            self.obs = f"Invalid action: {action}"
        
        self.steps += 1
        return self.obs, reward, done, self._get_info()
    
    def _normalize_answer(self, answer: str) -> str:
        """Normalize answer to standard format."""
        answer = answer.strip().upper()
        
        # Handle variations
        if answer in ["SUPPORT", "SUPPORTS"]:
            return "SUPPORT"
        elif answer in ["REFUTE", "REFUTES"]:
            return "REFUTE"
        elif answer in ["NOT ENOUGH INFO", "NEI", "NOINFO", "NOT_ENOUGH_INFO"]:
            return "NOT ENOUGH INFO"
        else:
            return answer
    
    def _get_ground_truth(self) -> str:
        """Get ground truth label for current claim."""
        if self.current_claim is None:
            return None
        
        # SciFACT claims have 'evidence' field with doc_id -> evidence mapping
        evidence = self.current_claim.get('evidence', {})
        
        if not evidence:
            return "NOT ENOUGH INFO"
        
        # Check labels in evidence
        for doc_id, doc_evidence in evidence.items():
            for ev in doc_evidence:
                label = ev.get('label', '').upper()
                if label in ['SUPPORT', 'SUPPORTS']:
                    return 'SUPPORT'
                elif label in ['REFUTE', 'REFUTES']:
                    return 'REFUTE'
        
        return "NOT ENOUGH INFO"
    
    def _compute_reward(self) -> float:
        """Compute reward based on answer correctness."""
        gt = self._get_ground_truth()
        if self.answer == gt:
            return 1.0
        return 0.0
    
    def _get_info(self) -> Dict:
        """Get info dictionary."""
        gt = self._get_ground_truth()
        em = 1.0 if self.answer == gt else 0.0
        
        return {
            "steps": self.steps,
            "answer": self.answer,
            "gt_answer": gt,
            "em": em,
            "f1": em,  # For classification, F1 = EM
            "claim_id": self.current_claim.get('id') if self.current_claim else None
        }


if __name__ == "__main__":
    # Test the environment
    print("\n[TEST] SciFACT Environment Test\n")
    
    env = SciFACTEnv()
    
    # Reset with first claim
    claim = env.reset(idx=0)
    print(f"Claim: {claim}")
    
    # Test search
    obs, _, _, _ = env.step("search[genetic mutation]")
    print(f"\nSearch result:\n{obs[:500]}...")
    
    # Test lookup
    obs, _, _, _ = env.step("lookup[mutation]")
    print(f"\nLookup result:\n{obs}")
    
    # Test finish
    obs, reward, done, info = env.step("finish[SUPPORT]")
    print(f"\nFinish: reward={reward}, info={info}")
