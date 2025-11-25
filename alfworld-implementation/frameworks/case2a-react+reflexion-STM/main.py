import os
import json
import argparse

from typing import Any, List, Dict

try:
    from .run_episode import run_trial  # type: ignore
except Exception:
    from run_episode import run_trial  # type: ignore


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_trials", type=int, default=1, help="Number of trials (episodes per env config dump)")
    parser.add_argument("--num_envs", type=int, default=10, help="Number of environments to run per trial")
    parser.add_argument("--run_name", type=str, default="case2a_run", help="Directory to write logs/results")
    parser.add_argument("--model", type=str, default="gemini-2.5-flash-lite", help="Model name for LLM calls")
    args = parser.parse_args()
    assert args.num_trials > 0 and args.num_envs > 0
    return args


def main(args) -> None:
    # Prepare logging dir
    if not os.path.exists(args.run_name):
        os.makedirs(args.run_name)

    # Initialize per-env configs
    env_configs: List[Dict[str, Any]] = []
    for i in range(args.num_envs):
        env_configs.append({
            "name": f"env_{i}",
            "is_success": False,
            "skip": False,
        })

    world_log_path = os.path.join(args.run_name, 'world.log')

    print(f"\n-----\nStarting CASE 2A (in-trajectory STM Reflexion)\nRun: {args.run_name}\nTrials: {args.num_trials}\nEnvs per trial: {args.num_envs}\nModel: {args.model}\nLogs in: {args.run_name}\n-----\n")

    for trial_idx in range(args.num_trials):
        with open(world_log_path, 'a') as wf:
            wf.write(f"\n\n***** Start Trial #{trial_idx} *****\n\n")

        trial_log_path = os.path.join(args.run_name, f'trial_{trial_idx}.log')
        trial_env_configs_log_path = os.path.join(args.run_name, f'env_results_trial_{trial_idx}.json')
        if os.path.exists(trial_log_path):
            open(trial_log_path, 'w').close()
        if os.path.exists(trial_env_configs_log_path):
            open(trial_env_configs_log_path, 'w').close()

        # Run one pass over N envs
        env_configs = run_trial(
            trial_log_path=trial_log_path,
            world_log_path=world_log_path,
            trial_idx=trial_idx,
            env_configs=env_configs,
            model=args.model,
        )

        # Persist env config snapshot for this trial
        with open(trial_env_configs_log_path, 'w') as wf:
            json.dump(env_configs, wf, indent=2)

        with open(world_log_path, 'a') as wf:
            wf.write(f"\n\n***** End Trial #{trial_idx} *****\n\n")


if __name__ == "__main__":
    main(get_args())
