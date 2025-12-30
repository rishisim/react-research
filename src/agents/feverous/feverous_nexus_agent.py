"""
FEVEROUS-Specific Nexus Agent with TableLookup Support

This agent extends the base NexusAgent with:
1. FEVEROUS-specific prompts that include TableLookup examples
2. Support for table_lookup action in Scout and Architect phases
3. Better handling of structured data claims
"""

import re
import json
import logging
from typing import List, Dict, Any, Tuple

# Import FEVEROUS-specific prompts (relative import)
from prompts.nexus_feverous import (
    SCOUT_PROMPT_FEVEROUS,
    ARCHITECT_PROMPT_FEVEROUS,
    ADJUDICATOR_PROMPT_FEVEROUS
)

logger = logging.getLogger(__name__)

class FeverousNexusAgent:
    """
    Nexus Agent specialized for FEVEROUS dataset.
    
    Uses 3 phases (Scout -> Architect -> Adjudicator) with support for:
    - Search[entity]: Wikipedia page lookup
    - Lookup[keyword]: Text search on current page  
    - TableLookup[query]: Structured table access
    """
    
    def __init__(self, llm_func, env):
        self.llm = llm_func
        self.env = env
        self.framework = "nexus_feverous"
        self.task_type = "feverous"
        self.n_calls = 0
    
    def _parse_similar_results(self, obs: str) -> List[str]:
        """Parse 'Could not find X. Similar: [...]' responses to extract alternatives."""
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
    
    def _is_invalid_search(self, obs: str) -> bool:
        """Check if observation indicates a failed search."""
        return (
            "Could not find" in obs 
            or "Invalid action" in obs 
            or "There were no results matching the query" in obs
            or not obs.strip()
        )
    
    def _execute_action(self, action: str) -> str:
        """
        Execute an action and return the observation.
        Handles Search, Lookup, and TableLookup actions.
        """
        action_lower = action.lower().strip()
        
        # Normalize action format
        if action_lower.startswith("search["):
            query = action[7:-1]  # Extract query
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
        
    def solve(self, question: str) -> Tuple[str, Dict[str, Any]]:
        """
        Solve a FEVEROUS claim using the 3-phase Nexus approach.
        """
        # Clean question
        clean_question = question.replace("Claim: ", "").strip()
        
        # --- Phase 1: SCOUT ---
        scout_entities = self.scout_phase(clean_question)
        passports = {}
        trace = f"Claim: {question}\n\n[Phase 1: Scout]\n"
        
        for entity in scout_entities:
            if not entity.strip():
                continue
            
            obs = self._execute_action(f"Search[{entity}]")
            
            # Retry with similar if failed
            if self._is_invalid_search(obs):
                similar = self._parse_similar_results(obs)
                if similar:
                    alt_entity = similar[0]
                    trace += f"Scout Search[{entity}] -> Invalid, retrying with '{alt_entity}'\n"
                    obs = self._execute_action(f"Search[{alt_entity}]")
                    passports[alt_entity] = obs
                    trace += f"Scout Search[{alt_entity}] -> {str(obs)}\n"
                    continue
            
            passports[entity] = obs
            trace += f"Scout Search[{entity}] -> {str(obs)}\n"
        
        # --- Phase 2: ARCHITECT ---
        architect_result, bridge_trace = self.architect_phase(question, passports)
        trace += f"\n[Phase 2: Architect]\n{bridge_trace}\n"
        
        dossier = self.compile_dossier(passports, architect_result)
        
        # --- Phase 3: ADJUDICATOR ---
        answer, adj_trace = self.adjudicator_phase(question, dossier)
        trace += f"\n[Phase 3: Adjudicator]\n{adj_trace}"
        
        return answer, {"traj": trace, "dossier": dossier}

    def scout_phase(self, question: str) -> List[str]:
        """Identify key entities to search."""
        prompt = SCOUT_PROMPT_FEVEROUS.format(question=question)
        try:
            self.n_calls += 1
            llm_output = self.llm(prompt, stop=["\n\n"])
            
            # Parse entity list
            if "[" in llm_output and "]" in llm_output:
                start = llm_output.find("[")
                end = llm_output.rfind("]") + 1
                entities = json.loads(llm_output[start:end])
                return entities  # No limit - trust LLM judgment
            else:
                return [question]
        except Exception:
            return [question]

    def architect_phase(self, question: str, passports: Dict[str, str]) -> Tuple[Dict[str, str], str]:
        """Analyze passports and generate bridge queries if needed."""
        # Flatten passports
        passport_text = ""
        for entity, content in passports.items():
            passport_text += f"[Passport {entity}]: {content}\n"
        
        prompt = ARCHITECT_PROMPT_FEVEROUS.format(
            question=question, 
            passports=passport_text
        )
        
        self.n_calls += 1
        llm_output = self.llm(prompt, stop=[])
        
        trace_log = f"Architect Thought:\n{llm_output}\n"
        bridge_info = {}
        
        # Parse Bridge Actions (supports Search, Lookup, TableLookup)
        actions_match = re.search(r"Bridge Actions:\s*(\[.*?\])", llm_output, re.DOTALL | re.IGNORECASE)
        
        if actions_match:
            try:
                import ast
                actions_str = actions_match.group(1)
                actions = ast.literal_eval(actions_str)
            except:
                # Fallback regex for various action formats
                actions_matches = re.findall(r"(Search|Lookup|Table_?Lookup)\s*[\(\[]\s*[\"']?(.*?)[\"']?\s*[\)\]]", llm_output, re.IGNORECASE)
                actions = []
                for action_type, query in actions_matches:
                    clean_type = action_type.lower().replace('_', '')
                    if clean_type == "tablelookup":
                        actions.append(f"TableLookup[{query}]")
                    elif clean_type == "lookup":
                        actions.append(f"Lookup[{query}]")
                    else:
                        actions.append(f"Search[{query}]")
            
            trace_log += f"Generated Bridge Candidates: {actions}\n"
            
            # Execute bridge actions with fallback
            success = False
            for i, action_raw in enumerate(actions):
                if not action_raw:
                    continue
                
                # Parse action type and query - robust regex
                action_match = re.search(r"(Search|Lookup|Table_?Lookup)[\(\[]\s*[\"']?(.*?)[\"']?\s*[\)\]]", action_raw, re.IGNORECASE)
                if action_match:
                    action_type = action_match.group(1).lower().replace('_', '')
                    query = action_match.group(2)
                else:
                    # Treat as search query
                    action_type = "search"
                    query = action_raw.strip()
                
                if not query:
                    continue
                
                # Map action type
                if action_type == "tablelookup":
                    full_action = f"table_lookup[{query}]"
                elif action_type == "lookup":
                    full_action = f"lookup[{query}]"
                else:
                    full_action = f"search[{query}]"
                
                try:
                    obs, _, _, _ = self.env.step(full_action)
                    
                    if not self._is_invalid_search(obs):
                        bridge_info[f"Bridge_{action_type}_{query}"] = obs
                        trace_log += f"Bridge Attempt {i+1} ({full_action}) -> Success: {str(obs)}\n"
                        success = True
                        # break  <-- Removed to allow multiple actions
                    else:
                        # Try similar
                        similar = self._parse_similar_results(obs)
                        if similar:
                            alt_query = similar[0]
                            trace_log += f"Bridge Attempt {i+1} ({full_action}) -> Invalid, retrying with '{alt_query}'\n"
                            # Preserve action type for retry (not always search)
                            if action_type == "tablelookup":
                                alt_action = f"table_lookup[{alt_query}]"
                            elif action_type == "lookup":
                                alt_action = f"lookup[{alt_query}]"
                            else:
                                alt_action = f"search[{alt_query}]"
                            alt_obs, _, _, _ = self.env.step(alt_action)
                            
                            if not self._is_invalid_search(alt_obs):
                                bridge_info[f"Bridge_{action_type}_{alt_query}"] = alt_obs
                                trace_log += f"Bridge Retry ({alt_action}) -> Success\n"
                                success = True
                                # No break - continue to next bridge action
                            else:
                                trace_log += f"Bridge Retry ({alt_action}) -> Also failed\n"
                        else:
                            trace_log += f"Bridge Attempt {i+1} ({full_action}) -> Failed (No results)\n"
                
                except Exception as e:
                    trace_log += f"Bridge Attempt {i+1} ({full_action}) -> Error: {e}\n"
            
            if not success:
                trace_log += "All Bridge queries failed.\n"
        else:
            trace_log += "No Bridge Query needed (or failed parsing).\n"
        
        return bridge_info, trace_log

    def adjudicator_phase(self, question: str, dossier: str) -> Tuple[str, str]:
        """Synthesize evidence and produce final answer."""
        prompt = ADJUDICATOR_PROMPT_FEVEROUS.format(question=question, dossier=dossier)
        
        self.n_calls += 1
        llm_output = self.llm(prompt, stop=[])
        
        # Parse answer - strict format first
        match = re.search(r"^Answer:\s*(.+)$", llm_output, re.MULTILINE | re.IGNORECASE)
        if match:
            answer = match.group(1).strip()
        else:
            # Fallback: look for verdict keywords in output
            lines = llm_output.strip().split('\n')
            found_answer = None
            
            valid_labels = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]
            for line in reversed(lines):
                clean_line = line.upper().strip()
                for label in valid_labels:
                    if label in clean_line:
                        found_answer = label
                        break 
                if found_answer:
                    break
            
            answer = found_answer if found_answer else "NOT ENOUGH INFO"
        
        return answer, llm_output

    def compile_dossier(self, passports: Dict[str, str], bridge_info: Dict[str, str]) -> str:
        """Compile all evidence into a dossier."""
        dossier = ""
        for ent, text in passports.items():
            dossier += f"--- Information on {ent} ---\n{text}\n\n"
        for ent, text in bridge_info.items():
            dossier += f"--- {ent} ---\n{text}\n\n"
        return dossier
