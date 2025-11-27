# FEVER Agent Frameworks

This directory contains 4 distinct agent frameworks for FEVER fact verification:

## Agent Frameworks

### 1. **ReAct** (`react_agent.py`)
Standard single-trace ReAct agent.
- **Execution**: One reasoning trace with temperature=0.0 (deterministic)
- **Output**: Direct answer from the single trace
- **Usage**: `from react_agent import run_react`

### 2. **Reflexion ReAct** (`reflexion_react_agent.py`)
Two-trace approach with reflexion.
- **Execution**:
  1. Initial trace (temperature=0.0)
  2. Generate reflexion from trace 1
  3. Second trace with reflexion as context (temperature=0.0)
- **Output**: Answer from second trace (no synthesis)
- **Usage**: `from reflexion_react_agent import run_reflexion_react`

### 3. **Majority Voting** (`majority_voting_agent.py`)
Three independent traces with simple voting.
- **Execution**: 3 traces with temperature=0.7 (diversity)
- **Output**: Majority vote (2/3), ties default to "NOT ENOUGH INFO"
- **Usage**: `from majority_voting_agent import run_majority_voting`

### 4. **CoT-SC** (`cot_sc_agent.py`)
Multi-trace with Chain-of-Thought Self-Consistency.
- **Execution**: 3 traces with temperature=0.7
- **Output**: LLM synthesis after reviewing all reasoning trajectories
- **Usage**: `from cot_sc_agent import run_cot_sc`

## Shared Utilities

**`fever_utils.py`** - Common functions used by all agents:
- LLM interaction (`llm()`)
- Environment setup (`get_fever_env()`, `step()`)
- Prompt loading
- Answer extraction and synthesis
- Core single-trace execution (`run_single_trace()`)

## Running Experiments

### Basic Usage

```bash
# Run single framework
python run_fever_experiments.py --frameworks react --num-examples 10

# Run multiple frameworks
python run_fever_experiments.py --frameworks react reflexion majority_voting cot_sc --num-examples 15

# Continue previous run (same seed)
python run_fever_experiments.py --frameworks react --num-examples 20 --seed 42

# Retry failed questions
python run_fever_experiments.py --frameworks react --num-examples 10 --seed 42 --retry-failed
```

### Command-Line Arguments

- `--model`: Gemini model to use (default: `gemini-2.5-flash-lite`)
- `--num-examples`: Number of examples to run (default: 5)
- `--frameworks`: Space-separated list of frameworks to run
- `--seed`: Random seed for reproducibility (default: 42)
- `--retry-failed`: Retry previously failed questions

## Continuation System

The experiment runner uses a **seed-based continuation system**:

1. **Seed-based directories**: Results accumulate in `seed{N}_{model}/`
2. **Index tracking**: 
   - `processed_indices.json` - Successfully completed questions
   - `failed_indices.json` - Failed questions (can retry)
3. **Resumption**: Next run with same seed skips processed indices
4. **No duplicates**: Each question is only run once per seed

### Example Workflow

```bash
# Run 1: Process 15 examples with seed=42
python run_fever_experiments.py --frameworks react --num-examples 15 --seed 42
# Creates: seed42_gemini-2.5-flash-lite/
# Processes indices: [3421, 891, 5523, ...]

# Run 2: Process 10 MORE examples with seed=42
python run_fever_experiments.py --frameworks react --num-examples 10 --seed 42
# Uses: seed42_gemini-2.5-flash-lite/
# Skips: [3421, 891, 5523, ...]
# Processes: NEW indices only

# Result: Total of 25 examples in seed42_gemini-2.5-flash-lite/react.json
```

## Results Structure

```
results/fever/seed42_gemini-2.5-flash-lite/
├── config.json                  # Experiment configuration
├── processed_indices.json       # Successfully processed question indices
├── failed_indices.json          # Failed question indices
├── run_history.json             # History of all runs
├── summary.json                 # Aggregate statistics
├── react.json                   # Results for ReAct framework
├── reflexion.json               # Results for Reflexion framework
├── majority_voting.json         # Results for Majority Voting framework
└── cot_sc.json                  # Results for CoT-SC framework
```

## Testing

Run the test suite to verify all agents work:

```bash
python test_agents.py
```

This tests all 4 frameworks with a sample FEVER example (index 3687).

## Error Handling

- **Successful runs**: Saved to framework results + `processed_indices.json`
- **Failed runs**: Saved to framework results (with error) + `failed_indices.json`
- **Retry**: Use `--retry-failed` flag to retry previously failed questions
- **Analysis**: Filter results by `status == 'success'` for accuracy calculation

## Output Format

Clean, emoji-free output:

```
======================================================================
[EXAMPLE 1/15] Index: 3421
----------------------------------------------------------------------
  Running react...
  > react: Answer=SUPPORTS | GT=SUPPORTS | EM=1.0
  [STATUS] Successfully processed
======================================================================
```
