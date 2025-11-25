# CASE 2A — ReAct + In‑Trajectory Reflexion (STM)

This framework runs ALFWorld tasks with a ReAct agent that performs short‑term, in‑trajectory self‑reflections at specific step indices (default: 6, 9, 11, 13). Reflections are appended to an in‑memory buffer and injected back into the prompt to steer subsequent actions.

## Files
- `main.py` — CLI entry point that manages trials, logging, and calls `run_episode.run_trial`.
- `run_episode.py` — Core loop for ALFWorld with ReAct + in‑trajectory reflection.
- `utils_case2a.py` — Gemini LLM wrapper.
- `base_config.yaml` — ALFWorld environment config (path variables rely on `$ALFWORLD_DATA`).
- `prompts/alfworld_3prompts.json` — Few‑shot prompts (fallbacks to the `case3` folder if missing).
- `run.sh` — Convenience script.

## Requirements
- Environment variable `GEMINI_API_KEY` in a `.env` at repository root (same as case3).
- `ALFWORLD_DATA` pointing to ALFWorld dataset directory.
- Python deps (install in your environment):

```
pip install -r requirements.txt
```

You also need ALFWorld installed and its dependencies available (refer to repo instructions).

## Run
From this folder:

```
python -u main.py --num_envs 10 --num_trials 1 --run_name case2a_run
```

Or use the helper script:

```
./run.sh 10 1 case2a_run
```

Logs are written to `run_name` with `trial_{i}.log` and `world.log`. Each environment log lists action/observation pairs and any in‑trajectory reflections.

## Notes
- Reflection trigger steps are currently `[6, 9, 11, 13]`. Adjust in `run_episode.py` if desired.
- The action format and environment stepping mirror case3 for consistency.
