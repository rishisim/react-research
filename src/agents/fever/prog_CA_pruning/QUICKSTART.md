# Quick Start Guide: prog_CA_pruning

## Installation

All files are already in place:
```
/Users/rishisim/Documents/research/react-research/src/agents/fever/prog_CA_pruning/
```

No additional dependencies needed beyond existing FEVER setup.

## Quick Test (30 seconds)

```bash
cd /Users/rishisim/Documents/research/react-research/src/agents/fever/prog_CA_pruning

python3 -c "
from prog_ca_pruning_agent import run_prog_ca_pruning_react

reward, info = run_prog_ca_pruning_react(idx=3687, to_print=True)
print(f'\\n[RESULT] Answer={info[\"answer\"]} EM={info[\"em\"]} Tokens={info[\"total_tokens\"]}')
"
```

## Run Batch Tests

```bash
cd /Users/rishisim/Documents/research/react-research/src/agents/fever/prog_CA_pruning
python3 test_prog_ca.py
```

Tests 3 examples with aggregate stats.

## Run Full Experiments

```bash
# Single command to process N examples
python3 run_prog_ca_experiments.py --num 10

# For 50 examples
python3 run_prog_ca_experiments.py --num 50
```

Results saved to:
- `../../../results/fever/prog_CA_pruning/prog_CA_pruning_results.json`
- `../../../results/fever/prog_CA_pruning/prog_CA_pruning_experiment.log`

## Integration into Existing Runners

To add to existing FEVER experiment runner:

```python
# In src/agents/fever/run_fever_experiments.py

import sys
sys.path.append(os.path.dirname(__file__))
from prog_CA_pruning.prog_ca_pruning_agent import run_prog_ca_pruning_react

# Add to your experiment loop:
reward, info = run_prog_ca_pruning_react(idx=idx, to_print=False)
append_to_json(info, output_file_path)
```

## What Gets Pruned

### Action Pruning (reduces steps)
- ❌ **Repeated searches**: `search("Einstein")` after `search("Einstein")`
- ❌ **Excessive lookups**: >2 lookups on same entity within 10 steps
- ❌ **Failed actions**: Don't retry `search("Xyzzy")` after "not found"
- ❌ **Similar queries**: `search("Marie Curie")` vs `search("Curie Marie")`
- ✅ **Early finishing**: Stop when confident answer found + 2+ evidence

### Context Pruning (reduces tokens/step)
- ✂️ **Old thoughts**: Don't feed back verbose thinking
- ✂️ **Old observations**: Keep only last 1-2 tool outputs
- ✂️ **Redundant evidence**: Deduplicate similar facts
- ✅ **Evidence state**: Keep 5-15 key facts with sources
- ✅ **Summary**: Compact: visited pages, focus, answer, confidence

## Interpreting Results

```json
{
  "em": 1.0,                          // Exact match (0 or 1)
  "n_calls": 5,                       // LLM calls made
  "total_tokens": 1587,               // Total tokens used
  "pruning": {
    "action_pruning": {
      "total_pruned": 2                // # actions blocked by pruning
    },
    "context_pruning": {
      "evidence_items": 4              // # facts kept
    }
  }
}
```

**Good signs**:
- ✅ `em=1.0`: Correct answer
- ✅ `total_pruned > 0`: Pruning is active
- ✅ `total_tokens < 2000`: Effective compression
- ✅ `n_calls < 6`: Efficient steps

## Thresholds

All tunable - see `action_pruner.py` for defaults:

```python
pruner.query_similarity_threshold = 0.85      # Query dedup strictness
pruner.cooldown_lookup_limit = 2              # Max lookups per entity
pruner.confidence_delta_threshold = 0.05      # Stabilization sensitivity
pruner.success_confidence_threshold = 0.85    # Finishing threshold
```

Higher values = less pruning (safer), Lower = more pruning (faster).

## Troubleshooting

**Issue**: Import errors
```bash
# Make sure you're in the right directory
cd /Users/rishisim/Documents/research/react-research/src/agents/fever
python3 -c "from prog_CA_pruning import *"
```

**Issue**: Timeout on examples
- Increase timeout in `fever_utils.step()`
- Set `num_examples=1` to test single example first

**Issue**: Results file grows too large
- Results are JSON, can be split by date
- Use `--num 10` for smaller batches

## Comparing Against Baselines

To compare token usage:

```bash
# Baseline ReAct (from existing code)
grep "total_tokens" results/fever/react/*/results.json

# prog_CA_pruning
grep "total_tokens" results/fever/prog_CA_pruning/prog_CA_pruning_results.json

# Compare EM scores
jq 'map(.em) | add / length' results/fever/react/*/results.json
jq 'map(.em) | add / length' results/fever/prog_CA_pruning/prog_CA_pruning_results.json
```

## Expected Performance

| Metric | Typical |
|--------|---------|
| Steps | 3-4 (vs 5-6 baseline) |
| Total Tokens | 1000-1500 (vs 2500-3000 baseline) |
| Token Savings | 60-70% |
| EM Score | 70-75% (similar to baseline) |
| Time per Example | 8-12 sec (with API delays) |

## File Locations

| File | What |
|------|------|
| `action_pruner.py` | 6 action pruning techniques |
| `context_pruner.py` | State compression |
| `prog_ca_pruning_agent.py` | Main agent (use this) |
| `README.md` | Full documentation |
| `test_prog_ca.py` | Test on 3 examples |
| `run_prog_ca_experiments.py` | Batch experiment runner |
| `IMPLEMENTATION_SUMMARY.md` | Implementation details |

## Next Steps

1. Run on 10 examples: `python3 run_prog_ca_experiments.py --num 10`
2. Check results: `cat ../../../results/fever/prog_CA_pruning/prog_CA_pruning_experiment.log`
3. Compare tokens: Check `total_tokens` in results JSON
4. Tune thresholds if needed (see README.md)
5. Run on full dev set (7405 examples)

---

For full details, see [README.md](README.md) in this directory.
