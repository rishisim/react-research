import re
import json
import logging
from typing import List, Dict, Any, Tuple
from src.agents.nexus.nexus_prompts import SCOUT_PROMPT, ARCHITECT_PROMPT, ADJUDICATOR_PROMPT

logger = logging.getLogger(__name__)

class NexusAgent:
    """
    Nexus Agent: A graph-based agent that uses 3 phases (Scout -> Architect -> Adjudicator)
    to solve multi-hop reasoning and retrieval bottleneck problems.
    """
    def __init__(self, llm_func, env, task_type="fever"):
        self.llm = llm_func
        self.env = env
        self.framework = "nexus"
        self.task_type = task_type  # "fever" or "hotpotqa"
        self.n_calls = 0  # LLM call tracking
        
    def solve(self, question: str) -> Tuple[str, Dict[str, Any]]:
        # Reset environment is NOT done here because we might receive an already reset env or we do it outside.
        # But usually we need to reset for the specific question if the env is stateful per question.
        # In FEVER runner, env.reset(idx=idx) is called. 
        # Here we assume the env is ready or we handle it via the wrapper.
        pass
        
        # Actually, for the generic solve, we just take the question string?
        # The environment `step` needs to work.
        
        # --- Phase 1: SCOUT ---
        # Clean question prompt
        clean_question = question.replace("Question: ", "").strip()
        scout_entities = self.scout_phase(clean_question)
        passports = {}
        trace = f"Question: {question}\n\n[Phase 1: Scout]\n"
        
        for entity in scout_entities:
            # Check for empty entity
            if not entity.strip(): continue
            
            # Using env.step which returns (obs, reward, done, info)
            # We assume the env wrapper handles Search[Entity]
            try:
                # The action must correspond to what the environment expects.
                # For FEVER/HotPot, it expects "search[Entity]" (lowercase 's')
                action = f"search[{entity}]"
                obs, _, _, _ = self.env.step(action)
                passports[entity] = obs
                trace += f"Scout search[{entity}] -> {str(obs)[:200]}...\n"
            except Exception as e:
                passports[entity] = f"Error searching {entity}: {e}"
                trace += f"Scout search[{entity}] -> Error\n"
            
        # --- Phase 2: ARCHITECT ---
        # Goal: Link Analysis and Bridge Query
        architect_result, bridge_trace = self.architect_phase(question, passports)
        trace += f"\n[Phase 2: Architect]\n{bridge_trace}\n"
        
        dossier = self.compile_dossier(passports, architect_result)
        
        # --- Phase 3: ADJUDICATOR ---
        # Goal: Synthesis
        answer, adj_trace = self.adjudicator_phase(question, dossier)
        trace += f"\n[Phase 3: Adjudicator]\n{adj_trace}"
        
        return answer, {"traj": trace, "dossier": dossier}

    def scout_phase(self, question: str) -> List[str]:
        prompt = SCOUT_PROMPT.format(question=question)
        try:
            # llm_func(prompt, stop)
            self.n_calls += 1
            llm_output = self.llm(prompt, stop=["\n\n"])
            # Simple parsing attempt
            if "[" in llm_output and "]" in llm_output:
                start = llm_output.find("[")
                end = llm_output.rfind("]") + 1
                entities = json.loads(llm_output[start:end])
                return entities[:3] # Limit to top 3
            else:
                 return [question]
        except Exception:
            return [question]

    def architect_phase(self, question: str, passports: Dict[str, str]) -> Tuple[Dict[str, str], str]:
        # Flatten passports for prompt
        passport_text = ""
        for entity, content in passports.items():
            passport_text += f"[Passport {entity}]: {content}\n"
            
        prompt = ARCHITECT_PROMPT.format(question=question, passport_a=passport_text, passport_b="(See above)")
        self.n_calls += 1
        llm_output = self.llm(prompt, stop=[])
        
        trace_log = f"Architect Thought:\n{llm_output}\n"
        bridge_info = {}
        
        # Check for Bridge Actions (Multiple)
        # Regex to capture the list inside ["..."]
        actions_match = re.search(r"Bridge Actions:\s*(\[.*?\])", llm_output, re.DOTALL | re.IGNORECASE)
        
        if actions_match:
            try:
                # Parse list using ast.literal_eval for safety, or simple json if strict
                actions_str = actions_match.group(1)
                # Cleaning quotes if needed or relying on AST
                import ast
                actions = ast.literal_eval(actions_str)
            except:
                # Fallback regex if eval fails (e.g. malformed list)
                actions = re.findall(r"Search\((.*?)\)", actions_match.group(1), re.IGNORECASE)
                # Convert to standard format
                actions = [f"search[{a}]" for a in actions]

            trace_log += f"Generated Bridge Candidates: {actions}\n"

            # Fallback Loop
            success = False
            for i, action_raw in enumerate(actions):
                # Clean up action string to match Search[...] or search[...]
                # The prompt might output "Search(Query)" or "search[Query]"
                # OR it might just output the raw query "Query" inside the list.
                
                # Extract query content
                q_match = re.search(r"Search[\(\[]\s*[\"']?(.*?)[\"']?\s*[\]\)]", action_raw, re.IGNORECASE)
                if q_match:
                    query = q_match.group(1)
                else:
                    # Assume the whole string is the query if it's not wrapped
                    query = action_raw.strip()
                
                # Sanity check: Ensure we aren't executing empty queries
                if not query: 
                    continue

                full_command = f"search[{query}]"
                
                try:
                    obs, _, _, _ = self.env.step(full_command)
                    
                    # Heuristic for "Valid Result"
                    # WikiEnv returns "Could not find..." or "Invalid action" on failure.
                    # We also want to avoid empty results or "Main page..." generic dumps if possible.
                    is_failure = (
                        "Could not find" in obs 
                        or "Invalid action" in obs 
                        or "There were no results matching the query" in obs
                        or not obs.strip()
                    )
                    
                    if not is_failure:
                        bridge_info[f"Bridge_{query}"] = obs
                        trace_log += f"Bridge Attempt {i+1} ({full_command}) -> Success: {str(obs)[:100]}...\n"
                        success = True
                        break # Stop after first success
                    else:
                        trace_log += f"Bridge Attempt {i+1} ({full_command}) -> Failed (No results)\n"
                        
                except Exception as e:
                    trace_log += f"Bridge Attempt {i+1} ({full_command}) -> Error: {e}\n"
            
            if not success:
                 trace_log += "All Bridge queries failed.\n"

        else:
            trace_log += "No Bridge Query needed (or failed parsing).\n"
            
        return bridge_info, trace_log

    def adjudicator_phase(self, question: str, dossier: str) -> Tuple[str, str]:
        # Use task-aware prompt suffix
        task_instructions = ""
        if self.task_type == "fever":
            task_instructions = "For FEVER: Output SUPPORTS, REFUTES, or NOT ENOUGH INFO."
        else:
            task_instructions = "For QA: Output only the short answer. Do NOT say 'not enough info' - make your best guess from the evidence."
        
        prompt = ADJUDICATOR_PROMPT.format(question=question, dossier=dossier) + "\n" + task_instructions
        self.n_calls += 1
        llm_output = self.llm(prompt, stop=[])
        
        # Extract Answer
        # Parse strict "Answer: ..." format
        match = re.search(r"^Answer:\s*(.+)$", llm_output, re.MULTILINE | re.IGNORECASE)
        if match:
             answer = match.group(1).strip()
        else:
             # Fallback: try to find the last line or just return output
             lines = llm_output.strip().split('\n')
             # Default depends on task type
             default_answer = "NOT ENOUGH INFO" if self.task_type == "fever" else "null"
             answer = lines[-1] if lines else default_answer
            
        return answer, llm_output
        
    def compile_dossier(self, passports: Dict[str, str], bridge_info: Dict[str, str]) -> str:
        dossier = ""
        for ent, text in passports.items():
            dossier += f"--- Information on {ent} ---\n{text}\n\n"
        for ent, text in bridge_info.items():
             dossier += f"--- Bridge Info: {ent} ---\n{text}\n\n"
        return dossier
