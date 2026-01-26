"""
Programmatic Action Pruning Module

Implements 6 action pruning techniques:
1. Loop Detection - prevent repeated same search/lookup
2. Success Gating - exit early when answer found with confidence
3. Query Deduplication - block near-identical queries
4. Cooldowns - prevent over-querying same entity
5. Failure Pattern Pruning - avoid redoing failed actions
6. Confidence Stabilization - stop when answer confidence plateaus

This is used in the ReAct step loop to gate actions before execution.
"""

import re
from typing import Tuple, Optional, Dict, List, Set
from collections import deque
from dataclasses import dataclass, field
from difflib import SequenceMatcher

@dataclass
class PrunerState:
    """Tracks state for pruning decisions."""
    # Recent actions: (action_type, args) tuples
    recent_action_keys: deque = field(default_factory=lambda: deque(maxlen=10))
    
    # Query tracking
    seen_queries: Set[str] = field(default_factory=set)
    query_history: deque = field(default_factory=lambda: deque(maxlen=20))
    
    # Entity/page lookups: {entity: count_in_recent_window}
    entity_lookup_counts: Dict[str, int] = field(default_factory=dict)
    
    # Recent failures: {(action, args): timestamp}
    recent_failures: deque = field(default_factory=lambda: deque(maxlen=5))
    
    # Answer tracking
    answer_candidate: Optional[str] = None
    answer_confidence: float = 0.0
    confidence_history: deque = field(default_factory=lambda: deque(maxlen=5))
    evidence_count: int = 0
    
    # Step tracking
    current_step: int = 0
    total_steps: int = 0
    
    # Pruning stats
    pruned_actions: List[Dict] = field(default_factory=list)


class ActionPruner:
    """
    Programmatic action pruning for ReAct agents.
    
    Provides pre-action and post-action gating to prevent redundant/failed actions.
    """
    
    def __init__(self, 
                 enable_loop_detection: bool = True,
                 enable_success_gating: bool = True,
                 enable_query_dedup: bool = True,
                 enable_cooldowns: bool = True,
                 enable_failure_patterns: bool = True,
                 enable_confidence_stabilization: bool = True,
                 max_search_steps: int = 4):
        """
        Initialize the pruner with all techniques enabled by default.
        
        Args:
            enable_*: Toggle individual pruning techniques
            max_search_steps: Step count after which 'search' actions are considered stagnation
        """
        self.state = PrunerState()
        
        # Feature flags
        self.enable_loop_detection = enable_loop_detection
        self.enable_success_gating = enable_success_gating
        self.enable_query_dedup = enable_query_dedup
        self.enable_cooldowns = enable_cooldowns
        self.enable_failure_patterns = enable_failure_patterns
        self.enable_confidence_stabilization = enable_confidence_stabilization
        
        # Thresholds
        self.loop_window_size = 10  # Recent N actions to check
        self.query_similarity_threshold = 0.80  # Tighten dedup to block near-duplicates sooner
        self.cooldown_lookup_limit = 1  # Hard cap repeated lookups of the same target
        self.cooldown_window_steps = 10  # Steps to look back
        self.confidence_delta_threshold = 0.05  # Confidence change threshold
        self.confidence_plateau_steps = 2  # Steps for stabilization check
        self.success_confidence_threshold = 0.80  # Still earlier than before, but less aggressive
        self.success_evidence_count = 2  # Min evidence items for finishing
        self.max_search_steps = max_search_steps # Configurable max steps for search
    
    def pre_action(self, action: str, args: str, step_num: int) -> Tuple[bool, Optional[str]]:
        """
        Gate an action before execution.
        
        Args:
            action: Action type (search, lookup, finish)
            args: Action arguments
            step_num: Current step number
            
        Returns:
            Tuple of (allow_action, reason_if_blocked)
            - allow_action: True if action should proceed
            - reason_if_blocked: String explaining pruning reason if blocked
        """
        self.state.current_step = step_num
        
        # 1. Loop Detection
        if self.enable_loop_detection:
            allow, reason = self._check_loop_detection(action, args)
            if not allow:
                self._log_prune_decision(action, args, reason)
                return False, reason
        
        # 2. Query Deduplication
        if self.enable_query_dedup and action in ['search', 'lookup']:
            allow, reason = self._check_query_dedup(action, args)
            if not allow:
                # Add specific redirect guidance
                redirect = " Search a specific, different entity title." if action == 'search' else " Lookup a different keyword."
                self._log_prune_decision(action, args, reason)
                return False, reason + redirect
        
        # 3. Cooldowns
        if self.enable_cooldowns and action == 'lookup':
            allow, reason = self._check_cooldown(args)
            if not allow:
                self._log_prune_decision(action, args, reason)
                return False, reason
        
        # 4. Failure Pattern Pruning
        if self.enable_failure_patterns:
            allow, reason = self._check_failure_pattern(action, args)
            if not allow:
                self._log_prune_decision(action, args, reason)
                return False, reason
        
        
        # 5. Stagnation / Depth Pruning
        # The base prompt often causes the agent to search endlessly or loop. 
        # We enforce a strict step limit for SEARCH actions.
        if step_num >= self.max_search_steps and action == 'search':
             reason = f"[PRUNE-STAGNATION] Search at step {step_num} is likely inefficient. Use existing knowledge or Finish."
             self._log_prune_decision(action, args, reason)
             return False, reason + " Return Finish[answer] or Finish[NOT ENOUGH INFO]."

        return True, None
    
    def post_action(self, action: str, args: str, observation: str, is_done: bool):
        """
        Update state after action execution.
        
        Args:
            action: Action type
            args: Action arguments
            observation: Observation from environment
            is_done: Whether task is complete
        """
        # Record this action in recent history
        action_key = self._canonicalize_action(action, args)
        self.state.recent_action_keys.append(action_key)
        
        # Track query if relevant
        if action in ['search', 'lookup']:
            normalized_query = self._normalize_query(args)
            self.state.query_history.append(normalized_query)
            self.state.seen_queries.add(normalized_query)
        
        # Track lookup counts
        if action == 'lookup':
            entity = args
            if entity not in self.state.entity_lookup_counts:
                self.state.entity_lookup_counts[entity] = 0
            self.state.entity_lookup_counts[entity] += 1
        
        # Check for failure pattern
        if self._is_failure_observation(observation):
            self.state.recent_failures.append((action, args))
        else:
            # Reset failure count for this action if successful
            self.state.entity_lookup_counts = {
                k: 0 for k in self.state.entity_lookup_counts
            }
    
    def set_answer_state(self, answer: str, confidence: float, evidence_count: int):
        """
        Update answer state for success gating.
        
        Args:
            answer: Current answer candidate (SUPPORTS/REFUTES/NOT ENOUGH INFO)
            confidence: Confidence in answer (0-1)
            evidence_count: Number of evidence items found
        """
        self.state.answer_candidate = answer
        self.state.answer_confidence = confidence
        self.state.evidence_count = evidence_count
        
        # Track confidence history
        self.state.confidence_history.append(confidence)
    
    def should_finish(self) -> Tuple[bool, Optional[str]]:
        """
        Check if success criteria are met.
        
        Returns:
            Tuple of (should_finish, reason)
        """
        if not self.enable_success_gating:
            return False, None
        
        if self.state.answer_confidence >= self.success_confidence_threshold and \
           self.state.evidence_count >= self.success_evidence_count:
            reason = f"[PRUNE] Success gate: confidence={self.state.answer_confidence:.2f}, evidence={self.state.evidence_count}"
            self._log_prune_decision('finish', '', reason)
            return True, reason
        
        # Check confidence stabilization
        if self.enable_confidence_stabilization:
            should_stabilize, reason = self._check_confidence_stabilization()
            if should_stabilize:
                self._log_prune_decision('finish', '', reason)
                return True, reason
        
        return False, None
    
    # --- Private helper methods ---
    
    def _canonicalize_action(self, action: str, args: str) -> str:
        """Canonicalize action for comparison (lowercase, remove punctuation)."""
        action_lower = action.lower().strip()
        args_lower = args.lower().strip()
        # Create hash for fast comparison
        key = f"{action_lower}:{args_lower}"
        return key
    
    def _normalize_query(self, query: str) -> str:
        """Normalize query for deduplication."""
        # Lowercase, remove punctuation, strip whitespace
        normalized = query.lower().strip()
        normalized = re.sub(r'[^\w\s]', '', normalized)
        normalized = ' '.join(normalized.split())
        return normalized
    
    def _query_similarity(self, q1: str, q2: str) -> float:
        """Compute string similarity between two queries (0-1)."""
        q1_norm = self._normalize_query(q1)
        q2_norm = self._normalize_query(q2)
        
        if q1_norm == q2_norm:
            return 1.0
        
        # Use SequenceMatcher for fuzzy matching
        similarity = SequenceMatcher(None, q1_norm, q2_norm).ratio()
        return similarity
    
    def _check_loop_detection(self, action: str, args: str) -> Tuple[bool, Optional[str]]:
        """
        Check if this action is a loop (repeated in recent window).
        
        Returns:
            (allow, reason) - allow=False if loop detected
        """
        action_key = self._canonicalize_action(action, args)
        
        # Check if exact same action in recent history
        if action_key in self.state.recent_action_keys:
            reason = f"[PRUNE-LOOP] Repeated action detected: {action}[{args}] in recent history"
            return False, reason
        
        # Check for action cycles (A -> B -> A)
        if len(self.state.recent_action_keys) >= 2:
            last_two = list(self.state.recent_action_keys)[-2:]
            if last_two[0] == action_key and len(self.state.recent_action_keys) >= 4:
                # Same action appeared 2 steps ago
                four_ago = list(self.state.recent_action_keys)[-4] if len(self.state.recent_action_keys) >= 4 else None
                if four_ago and four_ago == action_key:
                    reason = f"[PRUNE-LOOP] Action cycle detected: {action}[{args}]"
                    return False, reason
        
        return True, None
    
    def _check_query_dedup(self, action: str, args: str) -> Tuple[bool, Optional[str]]:
        """
        Check if query is a near-duplicate of recent queries.
        
        Returns:
            (allow, reason) - allow=False if duplicate found
        """
        query = args
        
        # Allow disambiguation search even if similar
        if action == 'search':
            if '(' in query or ')' in query:  # Explicit disambiguation like "Melancholia (film)"
                return True, None
            
            # For entities, use exact title caching only (no fuzzy matching)
            # This prevents "Melancholia" vs "Melancholia (film)" blocking
            normalized = self._normalize_query(query)
            if normalized in self.state.seen_queries:
                 return False, f"[PRUNE-DEDUP] Exact entity duplicate: '{query}'"
            return True, None

        # For Lookup, use similarity
        for recent_query in list(self.state.query_history)[-5:]:
            similarity = self._query_similarity(query, recent_query)
            if similarity >= self.query_similarity_threshold:
                reason = f"[PRUNE-DEDUP] Duplicate query: '{query}' similar to recent '{recent_query}' (sim={similarity:.2f})"
                return False, reason
        
        return True, None
    
    def _check_cooldown(self, entity: str) -> Tuple[bool, Optional[str]]:
        """
        Check if entity has been looked up too many times recently.
        
        Returns:
            (allow, reason) - allow=False if cooldown active
        """
        count = self.state.entity_lookup_counts.get(entity, 0)
        
        if count >= self.cooldown_lookup_limit:
            reason = f"[PRUNE-COOLDOWN] Entity '{entity}' already looked up {count} times in recent window"
            return False, reason
        
        return True, None
    
    def _check_failure_pattern(self, action: str, args: str) -> Tuple[bool, Optional[str]]:
        """
        Check if this action recently failed.
        
        Returns:
            (allow, reason) - allow=False if pattern matches recent failure
        """
        # Check if we've tried this exact action/args recently
        for recent_action, recent_args in list(self.state.recent_failures)[-3:]:
            if recent_action == action and recent_args == args:
                reason = f"[PRUNE-FAIL] Action failed recently: {action}[{args}]"
                return False, reason
        
        return True, None
    
    def _is_failure_observation(self, observation: str) -> bool:
        """Detect if observation indicates failure."""
        obs_lower = observation.lower()
        
        failure_patterns = [
            'not found',
            'could not find',
            'no results',
            'no match',
            'disambiguation',
            'error',
            'timeout',
            'failed',
        ]
        
        for pattern in failure_patterns:
            if pattern in obs_lower:
                return True
        
        return False
    
    def _check_confidence_stabilization(self) -> Tuple[bool, Optional[str]]:
        """
        Check if answer confidence has stabilized.
        
        Returns:
            (should_stop, reason)
        """
        if len(self.state.confidence_history) < self.confidence_plateau_steps:
            return False, None
        
        recent_confs = list(self.state.confidence_history)[-self.confidence_plateau_steps:]
        
        # Check if deltas are very small
        deltas = [abs(recent_confs[i] - recent_confs[i-1]) for i in range(1, len(recent_confs))]
        
        if all(d < self.confidence_delta_threshold for d in deltas):
            reason = f"[PRUNE-CONF] Confidence stabilized: {recent_confs[-1]:.2f} (deltas={[f'{d:.3f}' for d in deltas]})"
            return True, reason
        
        return False, None
    
    def _log_prune_decision(self, action: str, args: str, reason: str):
        """Log a pruning decision."""
        self.state.pruned_actions.append({
            'step': self.state.current_step,
            'action': action,
            'args': args,
            'reason': reason
        })
    
    def get_stats(self) -> Dict:
        """
        Get pruning statistics.
        
        Returns:
            Dictionary with pruning stats
        """
        return {
            'total_pruned': len(self.state.pruned_actions),
            'pruned_actions': self.state.pruned_actions,
            'final_confidence': self.state.answer_confidence,
            'final_evidence_count': self.state.evidence_count,
            'steps': self.state.current_step,
        }
