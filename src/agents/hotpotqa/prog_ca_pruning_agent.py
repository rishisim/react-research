"""
Programmatic Combined Action and Context Pruning ReAct Agent for HotPotQA

Combines:
1. Programmatic Action Pruning:
   - Loop detection
   - Success gating
   - Query deduplication
   - Cooldowns
   - Failure pattern pruning
   - Confidence stabilization

2. Context Pruning:
   - Evidence state instead of full observations
   - Drop old thoughts
   - Compact running summary
   - Keep last 1-2 observations
   - Track visited pages, failures

Maintains same or better accuracy through smart pruning.
"""

import sys
import os
import re
import json

# Add src to path to allow absolute imports
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, '../../../'))
if src_dir not in sys.path:
    sys.path.append(src_dir)

# Import shared pruning modules
from src.agents.pruning.action_pruner import ActionPruner
from src.agents.pruning.context_pruner import ContextPruner, build_compact_prompt

# Import HotPotQA utilities
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import hotpotqa_utils

# Base instruction for HotPotQA
# (Adapted from hotpotqa_utils.py instruction)
instruction = """Solve a question answering task with interleaving Thought, Action, Observation steps. Thought can reason about the current situation, and Action can be three types: 
(1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.
(2) Lookup[keyword], which returns the next sentence containing keyword in the current passage.
(3) Finish[answer], which returns the answer and finishes the task.
Here are some examples.
If the question is based on a false premise or the information needed to answer is not available, answer "null"."""

# We can reuse the examples from hotpotqa_utils if needed, or stick to a simple prompt
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    prompt_path = os.path.join(script_dir, 'prompts_naive.json')
    with open(prompt_path, 'r') as f:
        prompt_dict = json.load(f)
    webthink_examples = prompt_dict.get('webthink_simple6', "")
except:
    webthink_examples = ""

PROG_CA_PRUNING_PROMPT_TEMPLATE = instruction + webthink_examples

def run_prog_ca_pruning_hotpot(idx, prompt_template=None, to_print=True):
    """
    Run Programmatic Combined Action & Context Pruning ReAct agent for HotPotQA.
    
    Args:
        idx: Question index from HotPotQA dataset
        prompt_template: Custom prompt template (uses pruning template if None)
        to_print: Whether to print progress during execution
        
    Returns:
        Tuple of (reward, info_dict)
    """
    if prompt_template is None:
        prompt_template = PROG_CA_PRUNING_PROMPT_TEMPLATE
    
    if to_print:
        print("="*70)
        print("[FRAMEWORK] Programmatic Combined Action & Context Pruning ReAct (HotPotQA)")
        print("="*70)
    
    # Initialize pruners
    # HotPotQA often needs more flexibility than FEVER, so we might adjust params slightly if needed.
    # Default params: search_steps=4 (stagnation). HotPotQA might need 5-6 hops sometimes, but usually 2-3 searches.
    action_pruner = ActionPruner(
        enable_loop_detection=True,
        enable_success_gating=False, # HotPotQA is harder to determine "success" purely by evidence count compared to FEVER
        enable_query_dedup=True,
        enable_cooldowns=True,
        enable_failure_patterns=True,
        enable_confidence_stabilization=False, # Disable for now as we don't extract confidence easily from HotPotQA model logic
        max_search_steps=6 # Allow more search steps for HotPotQA
    )
    
    context_pruner = ContextPruner(
        max_evidence_items=15,
        max_observations_kept=2,
        max_failures_kept=3,
        enable_evidence_dedup=True
    )
    
    # Initialize environment
    hotpot_env = hotpotqa_utils.get_hotpotqa_env()
    question = hotpot_env.reset(idx=idx)
    
    if to_print:
        print(f"[TRACE] Index: {idx}")
        print(f"[QUESTION] {question}")
    
    n_calls, n_badcalls = 0, 0
    total_input_tokens, total_output_tokens = 0, 0
    llm_calls = []
    current_trace_steps = []
    pruned_action_count = 0
    
    # Build initial prompt with template + question
    current_prompt = prompt_template + question + "\n"
    
    # Main step loop
    for i in range(1, 8):  # Max 7 steps per trace
        context_pruner.increment_step()
        
        # Build prompt using compressed context if we have steps
        if i > 1:
            current_prompt_input = build_compact_prompt(prompt_template, question, context_pruner)
            current_prompt_input += f"Thought {i}:"
        else:
             current_prompt_input = current_prompt + f"Thought {i}:"

        
        # Generate thought+action
        n_calls += 1
        thought_action, token_usage = hotpotqa_utils.llm(
            current_prompt_input,
            stop=[f"\nObservation {i}:"],
            num_traces=1
        )
        total_input_tokens += token_usage['input_tokens']
        total_output_tokens += token_usage['output_tokens']
        context_pruner.add_tokens(token_usage['input_tokens'], token_usage['output_tokens'])
        
        llm_calls.append({
            'call_num': n_calls,
            'step': i,
            'type': 'thought_action',
            'input_tokens': token_usage['input_tokens'],
            'output_tokens': token_usage['output_tokens'],
            'total_tokens': token_usage['total_tokens']
        })
        
        # Parse thought and action
        try:
            thought, action = thought_action.strip().split(f"\nAction {i}: ")
        except:
            if to_print:
                print(f"[ERROR] Parsing thought/action: '{thought_action}'")
            n_badcalls += 1
            thought = thought_action.strip().split('\n')[0] if thought_action else "Error in thought generation"
            
            # Recovery call
            # For recovery, we probably want to use the same compressed prompt context
            action_prompt = current_prompt_input + f" {thought}\nAction {i}:"
            
            action, recovery_token_usage = hotpotqa_utils.llm(action_prompt, stop=["\n"], num_traces=1)
            action = action.strip()
            total_input_tokens += recovery_token_usage['input_tokens']
            total_output_tokens += recovery_token_usage['output_tokens']
            context_pruner.add_tokens(recovery_token_usage['input_tokens'], recovery_token_usage['output_tokens'])
            n_calls += 1
            
            llm_calls.append({
                'call_num': n_calls,
                'step': i,
                'type': 'action_recovery',
                'input_tokens': recovery_token_usage['input_tokens'],
                'output_tokens': recovery_token_usage['output_tokens'],
                'total_tokens': recovery_token_usage['total_tokens']
            })
            
            if not action or ("Finish[" not in action and "Search[" not in action and "Lookup[" not in action):
                action = "Finish[null]"
                if to_print:
                    print(f"[RECOVERY] Using default action: {action}")
        
        # Parse action type and args
        action_type = None
        action_args = None
        
        if "Search[" in action:
            action_type = "search"
            match = re.search(r'Search\[([^\]]+)\]', action)
            action_args = match.group(1) if match else ""
        elif "Lookup[" in action:
            action_type = "lookup"
            match = re.search(r'Lookup\[([^\]]+)\]', action)
            action_args = match.group(1) if match else ""
        elif "Finish[" in action:
            action_type = "finish"
            match = re.search(r'Finish\[([^\]]+)\]', action)
            action_args = match.group(1) if match else "null"
        else:
            action_type = "unknown"
            action_args = action
        
        # --- ACTION PRUNING: Pre-action gate ---
        if action_type != "finish":
            allow, prune_reason = action_pruner.pre_action(action_type, action_args, i)
            
            if not allow:
                pruned_action_count += 1
                if to_print:
                    print(f"[PRUNE-PRE] {prune_reason}")
                
                # Force redirect
                if "[PRUNE-STAGNATION]" in prune_reason:
                     action = "Finish[null]"
                     action_type = "finish"
                     action_args = "null"
                elif action_type == "search":
                    # Switch to lookup if available
                    action = "Lookup[previous term]"  # Heuristic fallback
                    action_type = "lookup"
                    action_args = "previous term"
                else:
                    action = "Finish[null]"
                    action_type = "finish"
                    action_args = "null"
        
        # Execute action
        # Don't capitalize; WikiEnv expects lowercase
        action_lowercase = action[0].lower() + action[1:] if action else action
        
        obs, r, done, info = hotpotqa_utils.step(hotpot_env, action_lowercase)
        obs = obs.replace('\\n', '') if isinstance(obs, str) else str(obs)
        
        # --- POST-ACTION: Update state ---
        action_pruner.post_action(action_type, action_args, obs, done)
        context_pruner.add_observation(obs, source=f"{action_type}({action_args})")
        
        # Extract and add evidence from observation
        if action_type in ["search", "lookup"]:
            context_pruner.extract_and_add_evidence(obs, source=action_args, query=action_args)
        
        # Track visited pages
        if action_type == "search":
            context_pruner.add_visited_page(action_args)
        
        # Update focus based on thought
        context_pruner.update_focus(thought[:100])
        
        # Build trajectory string for display/logging
        step_str = f"Thought {i}: {thought}\nAction {i}: {action}\nObservation {i}: {obs}\n"
        current_prompt += step_str
        current_trace_steps.append(step_str)
        
        if to_print:
            print(step_str)
        
        if done:
            # Extract final answer
            answer = info.get('answer', 'null')
            # Update final state in pruner
            # (We don't strictly use it for success gating as disabled, but good for tracking)
            action_pruner.set_answer_state(answer, 1.0, len(context_pruner.state.evidence))
            context_pruner.update_answer(answer, 1.0)
            break
            
        # --- SUCCESS GATING (If enabled, checking here) ---
        should_finish, finish_reason = action_pruner.should_finish()
        if should_finish:
             if to_print:
                 print(f"[GATE-SUCCESS] {finish_reason}")
             
             final_answer = context_pruner.state.answer_candidate or 'null'
             obs, r, done, info = hotpotqa_utils.step(hotpot_env, f"finish[{final_answer}]")
             context_pruner.update_answer(final_answer, 1.0)
             break

    # Handle case where agent didn't finish
    if not done:
        if to_print:
            print(f"[WARNING] Agent did not finish in {i} steps. Forcing Finish[null].")
        obs_finish, r_finish, done_finish, info_finish = hotpotqa_utils.step(hotpot_env, "finish[null]")
        info = info_finish if isinstance(info_finish, dict) else {}
        if 'answer' not in info:
            info['answer'] = 'null'
    
    if not isinstance(info, dict):
        info = {}
    
    # Prepare result
    trace_info = info.copy()
    
    action_prune_stats = action_pruner.get_stats()
    context_prune_stats = context_pruner.get_stats()
    
    trace_info.update({
        'n_calls': n_calls,
        'n_badcalls': n_badcalls,
        'input_tokens': total_input_tokens,
        'output_tokens': total_output_tokens,
        'total_tokens': total_input_tokens + total_output_tokens,
        'llm_calls': llm_calls,
        'traj': current_prompt, # Return full path
        'question_idx': idx,
        'question_text': question,
        'answer': info.get('answer', 'null'),
        'framework': 'prog_CA_pruning_hotpot',
        
        # Pruning statistics
        'pruning': {
            'action_pruning': {
                'total_pruned': action_prune_stats.get('total_pruned', 0),
                'pruned_actions': action_prune_stats.get('pruned_actions', [])[:5],
            },
            'context_pruning': context_prune_stats,
        }
    })
    
    reward = trace_info.get('em', 0.0)
    
    if to_print:
        print("="*70)
        print(f"[FINAL] Answer: {trace_info['answer']} | GT: {trace_info.get('gt_answer')} | EM: {trace_info.get('em')}")
        print(f"[STATS] Calls: {n_calls} | Tokens: {total_input_tokens + total_output_tokens} | Pruned: {pruned_action_count}")
        print("="*70)
    
    return reward, trace_info

if __name__ == '__main__':
    # Test with a sample HotPotQA example
    print("\n[TEST] Running Programmatic CA Pruning HotPotQA agent test\n")
    # Using a known idx (0 or something available)
    try:
        reward, info = run_prog_ca_pruning_hotpot(idx=0, to_print=True)
    except Exception as e:
        print(f"[TEST ERROR] {str(e)}")
