"""
FEVEROUS-Specific ReWOO Agent

Implements the ReWOO (Reasoning WithOut Observation) paradigm:
1. Planner: Generates complete execution plan with #E placeholders upfront
2. Worker: Executes tool calls and substitutes placeholders with results
3. Solver: Synthesizes final answer from collected evidence

This decouples planning from execution, unlike ReAct/Nexus which interleave them.
"""

import re
import logging
from typing import List, Dict, Any, Tuple

# Import FEVEROUS-specific prompts
from prompts.rewoo_feverous import (
    PLANNER_PROMPT_FEVEROUS,
    SOLVER_PROMPT_FEVEROUS
)

logger = logging.getLogger(__name__)


class FeverousReWOOAgent:
    """
    ReWOO Agent specialized for FEVEROUS dataset.
    
    Uses Planner -> Worker -> Solver paradigm with:
    - Search[entity]: Wikipedia page lookup
    - Lookup[keyword]: Text search on current page  
    - TableLookup[query]: Structured table access
    """
    
    def __init__(self, llm_func, env):
        self.llm = llm_func
        self.env = env
        self.framework = "rewoo_feverous"
        self.task_type = "feverous"
        self.n_calls = 0
    
    def _parse_plan(self, plan_text: str) -> List[Tuple[str, str, str]]:
        """
        Parse the planner output to extract evidence steps.
        
        Returns list of (evidence_id, tool_name, tool_input) tuples.
        Example: [("#E1", "Search", "Magnus Carlsen"), ("#E2", "Lookup", "champion")]
        """
        steps = []
        # Match patterns like: #E1 = Search[query], #E2 = TableLookup[query using #E1]
        pattern = r'(#E\d+)\s*=\s*(Search|Lookup|TableLookup|Table_Lookup)\s*\[([^\]]+)\]'
        
        for match in re.finditer(pattern, plan_text, re.IGNORECASE):
            evidence_id = match.group(1)
            tool_name = match.group(2).lower().replace('_', '')
            tool_input = match.group(3).strip()
            steps.append((evidence_id, tool_name, tool_input))
        
        return steps
    
    def _substitute_placeholders(self, text: str, evidence: Dict[str, str]) -> str:
        """
        Replace #E placeholders with actual evidence values.
        
        For example, "#E1" in tool input is replaced with the actual result from #E1.
        """
        result = text
        for eid, value in evidence.items():
            # Truncate very long evidence to avoid bloated prompts
            truncated = value[:500] if len(value) > 500 else value
            result = result.replace(eid, truncated)
        return result
    
    def _execute_action(self, action: str) -> str:
        """
        Execute an action and return the observation.
        Handles Search, Lookup, and TableLookup actions.
        """
        action_lower = action.lower().strip()
        
        # Normalize action format for environment
        if action_lower.startswith("search["):
            query = action[7:-1]
            full_action = f"search[{query}]"
        elif action_lower.startswith("lookup["):
            query = action[7:-1]
            full_action = f"lookup[{query}]"
        elif action_lower.startswith("tablelookup["):
            query = action[12:-1]
            full_action = f"table_lookup[{query}]"
        elif action_lower.startswith("table_lookup["):
            query = action[13:-1]
            full_action = f"table_lookup[{query}]"
        else:
            # Assume it's a search query directly
            full_action = f"search[{action}]"
        
        try:
            obs, _, _, _ = self.env.step(full_action)
            return obs
        except Exception as e:
            return f"Error executing {full_action}: {e}"
    
    def _is_invalid_result(self, obs: str) -> bool:
        """Check if observation indicates a failed action."""
        return (
            "Could not find" in obs 
            or "Invalid action" in obs 
            or "There were no results matching the query" in obs
            or not obs.strip()
        )
    
    def _parse_similar_results(self, obs: str) -> List[str]:
        """Parse 'Could not find X. Similar: [...]' responses."""
        if "Could not find" not in obs or "Similar:" not in obs:
            return []
        try:
            match = re.search(r"Similar:\s*(\[.*?\])", obs)
            if match:
                import ast
                return ast.literal_eval(match.group(1))
        except Exception:
            pass
        return []
    
    def solve(self, question: str) -> Tuple[str, Dict[str, Any]]:
        """
        Solve a FEVEROUS claim using the ReWOO approach.
        """
        clean_question = question.replace("Claim: ", "").strip()
        trace = f"Claim: {question}\n\n"
        
        # --- Phase 1: PLANNER ---
        plan = self.planner_phase(clean_question)
        trace += f"[Phase 1: Planner]\n{plan}\n\n"
        
        # Parse the plan into execution steps
        steps = self._parse_plan(plan)
        trace += f"Parsed {len(steps)} execution steps\n"
        
        # --- Phase 2: WORKER ---
        evidence, worker_trace = self.worker_phase(steps)
        trace += f"\n[Phase 2: Worker]\n{worker_trace}\n"
        
        # --- Phase 3: SOLVER ---
        answer, solver_trace = self.solver_phase(question, evidence)
        trace += f"\n[Phase 3: Solver]\n{solver_trace}"
        
        return answer, {"traj": trace, "plan": plan, "evidence": evidence}
    
    def planner_phase(self, question: str) -> str:
        """
        Generate a complete execution plan with #E placeholders.
        """
        prompt = PLANNER_PROMPT_FEVEROUS.format(question=question)
        
        self.n_calls += 1
        plan = self.llm(prompt, stop=["\n\nClaim:"])
        
        return plan
    
    def worker_phase(self, steps: List[Tuple[str, str, str]]) -> Tuple[Dict[str, str], str]:
        """
        Execute the plan steps, substituting placeholders with results.
        
        Args:
            steps: List of (evidence_id, tool_name, tool_input) tuples
            
        Returns:
            Tuple of (evidence_dict, trace_log)
        """
        evidence = {}
        trace_log = ""
        
        for evidence_id, tool_name, tool_input in steps:
            # Substitute any #E placeholders in the input
            resolved_input = self._substitute_placeholders(tool_input, evidence)
            
            # Construct the action
            if tool_name == "search":
                action = f"Search[{resolved_input}]"
            elif tool_name == "lookup":
                action = f"Lookup[{resolved_input}]"
            elif tool_name == "tablelookup":
                action = f"TableLookup[{resolved_input}]"
            else:
                action = f"Search[{resolved_input}]"
            
            trace_log += f"{evidence_id} = {action}\n"
            
            # Execute the action
            obs = self._execute_action(action)
            
            # Handle failed searches with retry
            if self._is_invalid_result(obs):
                similar = self._parse_similar_results(obs)
                if similar and tool_name == "search":
                    alt_entity = similar[0]
                    trace_log += f"  -> Failed, retrying with '{alt_entity}'\n"
                    obs = self._execute_action(f"Search[{alt_entity}]")
                else:
                    trace_log += f"  -> Failed: {obs[:100]}...\n"
            
            # Store result
            evidence[evidence_id] = obs
            trace_log += f"  -> Collected {len(obs)} chars\n"
        
        return evidence, trace_log
    
    def solver_phase(self, question: str, evidence: Dict[str, str]) -> Tuple[str, str]:
        """
        Synthesize the final answer from collected evidence.
        """
        # Build evidence log
        evidence_log = ""
        for eid, content in evidence.items():
            # Truncate very long evidence
            truncated = content[:1000] if len(content) > 1000 else content
            evidence_log += f"[{eid}]: {truncated}\n\n"
        
        prompt = SOLVER_PROMPT_FEVEROUS.format(
            question=question,
            evidence_log=evidence_log
        )
        
        self.n_calls += 1
        llm_output = self.llm(prompt, stop=[])
        
        # Parse answer
        match = re.search(r"^Answer:\s*(.+)$", llm_output, re.MULTILINE | re.IGNORECASE)
        if match:
            answer = match.group(1).strip()
        else:
            # Fallback: look for verdict keywords
            valid_labels = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]
            found_answer = None
            for line in reversed(llm_output.strip().split('\n')):
                clean_line = line.upper().strip()
                for label in valid_labels:
                    if label in clean_line:
                        found_answer = label
                        break
                if found_answer:
                    break
            answer = found_answer if found_answer else "NOT ENOUGH INFO"
        
        return answer, llm_output
