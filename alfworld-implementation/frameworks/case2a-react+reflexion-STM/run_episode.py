"""CASE 2A: ReAct with in-trajectory self-reflection (short-term memory) for ALFWorld.

This adapts the pattern from case3 (across-trial memory) to insert reflection checkpoints
within a single episode to influence subsequent actions.
"""

from __future__ import annotations

import os
import sys
import json
import time
import importlib
from typing import Any, Dict, List, Tuple, Optional

import yaml

# Resolve repo root for potential relative imports if needed
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, '..', '..', '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

try:
    # When running as a package
    from .utils_case2a import llm_call  # type: ignore
except Exception:
    # When running as a script
    from utils_case2a import llm_call  # type: ignore


# Load prompts similar to case3
_PROMPTS_DIR = os.path.join(_SCRIPT_DIR, 'prompts')

_PROMPTS_PATH = os.path.join(_PROMPTS_DIR, 'alfworld_3prompts.json')
if not os.path.exists(_PROMPTS_PATH):
    # Fallback to case3 prompt file if present
    _CASE3_DIR = os.path.join(os.path.dirname(_SCRIPT_DIR), 'case3-react+reflexion', 'prompts')
    _PROMPTS_PATH = os.path.join(_CASE3_DIR, 'alfworld_3prompts.json')
with open(_PROMPTS_PATH, 'r') as f:
    PROMPTS = json.load(f)


def process_ob(ob: str) -> str:
    if ob.startswith('You arrive at loc '):
        ob = ob[ob.find('. ') + 2:]
    return ob


PREFIXES = {
    'pick_and_place': 'put',
    'pick_clean_then_place': 'clean',
    'pick_heat_then_place': 'heat',
    'pick_cool_then_place': 'cool',
    'look_at_obj': 'examine',
    'pick_two_obj': 'puttwo'
}


def build_base_prompt(task_key: str) -> str:
    # Use two few-shot exemplars like case3
    return (
        'Interact with a household to solve a task. Here are two examples.\n'
        + PROMPTS[f'react_{task_key}_1']
        + PROMPTS[f'react_{task_key}_0']
    )


REFLECTION_INSTRUCTIONS = (
    "You are partway through solving your current ALFWorld task. Do not describe the environment; "
    "instead analyze your progress so far. Identify: \n"
    "- Helpful steps that moved you toward the goal.\n"
    "- Missteps or loops and how to avoid them next.\n"
    "- If you've seen similar issues, recall what worked and adapt it.\n"
    "- Otherwise, propose a concise, concrete next-step plan.\n"
    "End with the single next action you will take. Keep it concise."
)


def maybe_reflect(step_idx: int, trigger_steps: List[int], history_text: str, instruction: str, num_traces: int, model: str) -> str:
    """If this step is a trigger, generate a reflection string."""
    if (step_idx + 1) not in trigger_steps:
        return ""

    reflection_prompt = f"""
{REFLECTION_INSTRUCTIONS}

---
Instruction:
{instruction}

Trajectory so far:
{history_text}

---
Reflection:
"""
    reflection = llm_call(reflection_prompt, stop=["Action:"], num_traces=max(1, num_traces), model=model).strip()
    return reflection


def _sanitize_action(raw_action: str) -> str:
    action = raw_action.strip()
    if action.startswith('>'):
        action = action.lstrip('> ').strip()
    if action.lower().startswith('action:'):
        action = action.split(':', 1)[1].strip()
    if action.startswith('>'):
        action = action.lstrip('> ').strip()
    return action


def alfworld_episode(env, base_prompt: str, instruction: str, trigger_steps: List[int], model: str, num_traces: int = 1, to_print: bool = True) -> Tuple[List[Dict[str, Any]], bool]:
    """
    Runs a single episode with ReAct + in-trajectory reflections.
    Returns the trajectory list and success flag.
    """
    trajectory: List[Dict[str, Any]] = []

    prompt_prefix = (
        base_prompt
        + "\nHere is the task.\n"
        + instruction.strip()
        + "\n"
    )

    history_block: str = ""
    reflections_block: str = ""

    def build_action_prompt() -> str:
        prompt = prompt_prefix
        if reflections_block:
            prompt += "\nReflections:\n" + reflections_block
        prompt += history_block
        if not prompt.endswith('\n'):
            prompt += "\n"
        prompt += "> "
        return prompt

    cur_step = 0
    while cur_step < 49:
        # Ask for next action
        full_prompt = build_action_prompt()
        raw_action = llm_call(full_prompt, stop=['\n'], num_traces=num_traces, model=model).strip()
        action = _sanitize_action(raw_action)

        if not action:
            action = 'look'

        # Step environment
        try:
            observation, reward, done, info = env.step([action])
            observation, reward, done = process_ob(observation[0]), info['won'][0], done[0]
            if action.startswith('think:'):
                observation = 'OK.'
        except AssertionError:
            observation, reward, done = 'Invalid action!', 0.0, False

        # Record step
        step_data = {
            'step': cur_step,
            'action': action,
            'observation': observation.strip(),
            'reward': reward,
            'done': done,
        }
        trajectory.append(step_data)
        history_block += f"> {action}\n{observation.strip()}\n"

        if to_print:
            print(f"\n--- Step {cur_step+1}/49 ---")
            print(f"Action: {action}")
            print("Observation:\n    " + "\n    ".join(observation.strip().split('\n')))

        if done:
            return trajectory, True

        # In-trajectory reflection at trigger steps
        reflection = maybe_reflect(cur_step, trigger_steps, history_block, instruction, num_traces, model)
        if reflection:
            reflections_block += f"Reflection (step {cur_step + 1}): {reflection}\n"
            trajectory.append({
                'step': cur_step + 0.5,
                'type': 'reflexion',
                'reflexion': reflection,
                'trigger_step': cur_step + 1,
            })
            if to_print:
                print("\n--- In-Trajectory Reflection ---")
                print(reflection)

        cur_step += 1

    return trajectory, False


def run_trial(
    trial_log_path: str,
    world_log_path: str,
    trial_idx: int,
    env_configs: List[Dict[str, Any]],
    model: str,
    trigger_steps: Optional[List[int]] = None,
) -> List[Dict[str, Any]]:
    """Run a pass over num_envs with in-trajectory STM reflections only."""
    # Import ALFWorld lazily to avoid import-time issues in static analysis
    import alfworld  # type: ignore
    import alfworld.agents.environment  # type: ignore
    importlib.reload(alfworld)  # type: ignore
    importlib.reload(alfworld.agents.environment)  # type: ignore

    with open(os.path.join(_SCRIPT_DIR, 'base_config.yaml')) as reader:
        config = yaml.safe_load(reader)
    split = "eval_out_of_distribution"

    env = alfworld.agents.environment.get_environment(config["env"]["type"])(config, train_eval=split)
    env = env.init_env(batch_size=1)

    num_successes = 0
    num_envs = len(env_configs)

    # Choose reflection trigger steps (configurable)
    trigger_steps_used: List[int] = trigger_steps if trigger_steps is not None else DEFAULT_TRIGGER_STEPS

    for z, env_config in enumerate(env_configs):
        ob, info = env.reset()
        ob = '\n'.join(ob[0].split('\n\n')[1:])
        name = '/'.join(info['extra.gamefile'][0].split('/')[-3:-1])

        print(f"using {name}")

        if env_config.get("is_success"):
            num_successes += 1
            with open(world_log_path, 'a') as wf:
                wf.write(f'Environment #{z} Trial #{trial_idx}: SUCCESS\n')
            with open(trial_log_path, 'a') as wf:
                wf.write(f'\n#####\n\nEnvironment #{z}: Success (already solved)\n\n#####\n')
            continue

        # Build few-shot base prompt for this task prefix
        base_prompt = None
        for prefix, key in PREFIXES.items():
            if name.startswith(prefix):
                base_prompt = build_base_prompt(key)
                break
        if base_prompt is None:
            # Fallback to a generic prompt if prefix not matched
            # Use 'put' examples as generic
            base_prompt = build_base_prompt('put')

        # Instruction visible to the agent is the initial observation from ALFWorld
        instruction = ob

        trajectory, is_success = alfworld_episode(
            env,
            base_prompt=base_prompt,
            instruction=instruction,
            trigger_steps=trigger_steps_used,
            model=model,
            num_traces=1,
            to_print=True,
        )

        if is_success:
            env_configs[z]['is_success'] = True
            num_successes += 1
            status_str = f'Environment #{z} Trial #{trial_idx}: SUCCESS'
        else:
            status_str = f'Environment #{z} Trial #{trial_idx}: FAIL'

        with open(world_log_path, 'a') as wf:
            wf.write(status_str + '\n')

        # Log trajectory to trial log
        with open(trial_log_path, 'a') as wf:
            wf.write(f"\n#####\n\nEnvironment #{z}:\n")
            for step in trajectory:
                if step.get('type') == 'reflexion':
                    wf.write(f"[Reflection @ step {step['trigger_step']}] {step['reflexion']}\n")
                else:
                    wf.write(f"> {step['action']}\n{step['observation']}\n")
            wf.write(f"\nSTATUS: {'OK' if is_success else 'FAIL'}\n\n#####\n")

    env.close()

    # Summarize trial
    log_str = f"""
-----
SUCCESS: {num_successes}
FAIL: {num_envs - num_successes}
TOTAL: {num_envs}
ACCURACY: {round(num_successes / max(1, num_envs), 2)}
-----"""
    with open(trial_log_path, 'a') as wf:
        wf.write(log_str)
    with open(world_log_path, 'a') as wf:
        wf.write(log_str + '\n')

    return env_configs


# ===== User-configurable defaults for direct execution =====
DEFAULT_NUM_TRIALS: int = 1
DEFAULT_NUM_ENVS: int = 2
DEFAULT_RUN_NAME: str = "case2a_run"
DEFAULT_MODEL: str = "gemini-2.5-flash-lite"
DEFAULT_TRIGGER_STEPS: List[int] = [19, 30, 37, 41, 44]


if __name__ == "__main__":
    # You can modify the defaults above to change run behavior when invoking:
    #   python run_episode.py
    run_name = DEFAULT_RUN_NAME
    if not os.path.exists(run_name):
        os.makedirs(run_name)

    # initialize environment configs
    env_configs: List[Dict[str, Any]] = []
    for i in range(DEFAULT_NUM_ENVS):
        env_configs.append({
            'name': f'env_{i}',
            'is_success': False,
            'skip': False,
        })

    world_log_path: str = os.path.join(run_name, 'world.log')
    print(f"\n-----\nStarting CASE 2A (in-trajectory STM Reflexion)\nRun: {run_name}\nTrials: {DEFAULT_NUM_TRIALS}\nEnvs per trial: {DEFAULT_NUM_ENVS}\nModel: {DEFAULT_MODEL}\nLogs in: {run_name}\n-----\n")

    for trial_idx in range(DEFAULT_NUM_TRIALS):
        with open(world_log_path, 'a') as wf:
            wf.write(f"\n\n***** Start Trial #{trial_idx} *****\n\n")

        trial_log_path = os.path.join(run_name, f'trial_{trial_idx}.log')
        trial_env_configs_log_path = os.path.join(run_name, f'env_results_trial_{trial_idx}.json')
        if os.path.exists(trial_log_path):
            open(trial_log_path, 'w').close()
        if os.path.exists(trial_env_configs_log_path):
            open(trial_env_configs_log_path, 'w').close()

        env_configs = run_trial(
            trial_log_path=trial_log_path,
            world_log_path=world_log_path,
            trial_idx=trial_idx,
            env_configs=env_configs,
            model=DEFAULT_MODEL,
            trigger_steps=DEFAULT_TRIGGER_STEPS,
        )

        with open(trial_env_configs_log_path, 'w') as wf:
            json.dump(env_configs, wf, indent=2)

        with open(world_log_path, 'a') as wf:
            wf.write(f"\n\n***** End Trial #{trial_idx} *****\n\n")
