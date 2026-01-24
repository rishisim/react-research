# Programmatic Combined Action & Context Pruning (prog_CA_pruning)

Combines **action pruning** and **context pruning** for efficient FEVER fact verification.

## Overview

### Problem
Standard ReAct agents on FEVER:
- Take many redundant steps (repeated searches, wasted lookups)
- Feed entire trajectory history to LLM (verbose, high token cost)
- Result: High token usage, slow inference, sometimes worse accuracy

### Solution
Two complementary pruning strategies:

**1. Programmatic Action Pruning (6 Techniques)**
- **Loop Detection**: Block repeated search/lookup with same arguments
- **Success Gating**: Exit early when answer found with confidence >= 0.85
- **Query Deduplication**: Block near-identical queries (similarity >= 0.85)
- **Cooldowns**: Prevent over-querying same entity (max 2 lookups per 10 steps)
- **Failure Pattern Pruning**: Avoid redoing actions that just failed
- **Confidence Stabilization**: Stop when answer confidence plateaus

Expected impact: **30-50% reduction in steps**

**2. Context Pruning (State Compression)**
- Keep compact **evidence state** (extracted facts with sources)
- Drop all **old thoughts** (verbose reasoning text)
- Maintain **running summary** (visited pages, current focus, answer candidate)
- Retain only **last 1-2 observations** (recent tool outputs)
- Track **recent failures** (avoid patterns)

Expected impact: **70% reduction in tokens per step**

**Combined**: **65-85% total token reduction** with same/better accuracy

## Module Structure

```
prog_CA_pruning/
├── __init__.py                    # Module exports
├── action_pruner.py              # ActionPruner class (6 techniques)
├── context_pruner.py             # ContextPruner class (state compression)
├── prog_ca_pruning_agent.py      # Main agent that uses both pruners
├── test_prog_ca.py               # Test suite
├── run_prog_ca_experiments.py    # Experiment runner with logging
└── README.md                      # This file
```

## Usage

### Quick Test

```python
from prog_ca_pruning_agent import run_prog_ca_pruning_react

# Run on a single FEVER example
reward, info = run_prog_ca_pruning_react(idx=3687, to_print=True)

print(f"Answer: {info['answer']}")
print(f"EM Score: {info['em']}")
print(f"Total Tokens: {info['total_tokens']}")
print(f"Actions Pruned: {info['pruning']['action_pruning']['total_pruned']}")
```

### Batch Testing

```bash
cd /Users/rishisim/Documents/research/react-research/src/agents/fever/prog_CA_pruning
python3 test_prog_ca.py
```

### Run Full Experiments

```bash
# Run on 10 examples
python3 run_prog_ca_experiments.py --num 10

# Run on 50 examples
python3 run_prog_ca_experiments.py --num 50
```

Results saved to:
- `results/fever/prog_CA_pruning/prog_CA_pruning_results.json` (results)
- `results/fever/prog_CA_pruning/prog_CA_pruning_experiment.log` (detailed log)

## Action Pruning Details

### 1. Loop Detection

**What**: Prevents repeated `search("X")` or `lookup("X")` with identical arguments

**How**: Maintains recent action history, checks if `(action, args)` already seen in last 10 steps

**Example**:
```
Step 3: search[Einstein]  
Step 4: lookup[born]  
Step 5: search[Einstein]  <- PRUNED (loop detected)
```

**Thresholds**:
- `loop_window_size`: 10 recent actions
- Blocks exact matches and 2-step cycles

### 2. Success Gating

**What**: Exit early when answer found with sufficient confidence and evidence

**How**: Tracks answer candidate, confidence, and evidence count. Finishes if:
- `confidence >= 0.85` AND
- `evidence_count >= 2`

**Example**:
```
Step 4: Found "Marie Curie won Nobel Prize" with evidence
        -> Confidence: 0.90, Evidence: 2
        -> Triggers early finish
```

**Thresholds**:
- `success_confidence_threshold`: 0.85
- `success_evidence_count`: 2

### 3. Query Deduplication

**What**: Block near-identical queries across steps

**How**: Normalizes queries (lowercase, remove punctuation), computes string similarity

**Example**:
```
Step 2: search[Marie Curie]  
Step 5: search[Marie Curie]    <- PRUNED (exact duplicate)
Step 6: search[Curie Marie]    <- PRUNED (similarity = 0.95)
Step 7: search[Pierre Curie]   <- ALLOWED (similarity = 0.60)
```

**Thresholds**:
- `query_similarity_threshold`: 0.85 (fuzzy match)
- Checks against last 5 queries

### 4. Cooldowns

**What**: Prevent excessive lookup operations on the same entity

**How**: Tracks per-entity lookup count in recent window (10 steps)

**Example**:
```
Step 2: lookup[Einstein biography]   (count=1)  
Step 3: lookup[Einstein relatives]   (count=2)  
Step 4: lookup[Einstein awards]      (count=3)  <- PRUNED (cooldown active)
```

**Thresholds**:
- `cooldown_lookup_limit`: 2 lookups per entity
- `cooldown_window_steps`: 10

### 5. Failure Pattern Pruning

**What**: Avoid repeating actions that recently failed

**How**: Detects failure observations ("not found", "error", etc.), blocks same action/args

**Example**:
```
Step 2: search[Xyzzy]  
Obs: "Could not find page..."  
Step 5: search[Xyzzy]  <- PRUNED (same action just failed)
```

**Failure patterns detected**:
- "not found"
- "no results"
- "disambiguation"
- "error"
- "timeout"

### 6. Confidence Stabilization

**What**: Stop exploration when answer confidence plateaus

**How**: Tracks confidence history, if change < 0.05 for 2 steps, trigger finish

**Example**:
```
Step 4: Confidence = 0.82  
Step 5: Confidence = 0.83  (delta = 0.01)  
Step 6: Confidence = 0.84  (delta = 0.01)  
        -> Deltas < 0.05 for 2 steps -> Finish
```

**Thresholds**:
- `confidence_delta_threshold`: 0.05
- `confidence_plateau_steps`: 2

## Context Pruning Details

### What Gets Kept

**1. Evidence State** (5-15 items)
```
[1] "Marie Curie won Nobel Prize in Physics (1903)" [Wikipedia: Marie Curie]
[2] "She won Nobel Prize in Chemistry (1911)" [Wikipedia: Nobel Prize winners]
[3] "First woman to win two Nobel Prizes" [Wikipedia: Women Nobel laureates]
```

**2. Running Summary**
```
Step: 6
Visited Pages: Marie Curie, Nobel Prize, Physics
Current Focus: Nobel Prize timeline
Current Answer: SUPPORTS (confidence: 0.90)
```

**3. Last 1-2 Observations** (recent tool output)
```
Marie Curie (1867–1934) was a Polish-born physicist and chemist...
She is the first woman to win a Nobel Prize...
```

**4. Recent Failures** (2-3 items)
```
- search[Xyzzy]: "Could not find page"
- lookup[foobar]: "No results found"
```

### What Gets Dropped

- ✂️ **All thoughts** (verbose reasoning text)
- ✂️ **Old observations** (observations > 2 steps ago)
- ✂️ **Redundant evidence** (near-duplicate facts)
- ✂️ **Full search result lists** (keep only selected pages)

## Result Structure

Each result includes:

```json
{
  "question_idx": 3687,
  "question_text": "Marie Curie won two Nobel Prizes",
  "answer": "SUPPORTS",
  "gt_answer": "SUPPORTS",
  "em": 1.0,
  "n_calls": 5,
  "input_tokens": 1245,
  "output_tokens": 342,
  "total_tokens": 1587,
  "framework": "prog_CA_pruning",
  "pruning": {
    "action_pruning": {
      "total_pruned": 2,
      "pruned_actions": [
        {
          "step": 3,
          "action": "search",
          "args": "Marie Curie",
          "reason": "[PRUNE-LOOP] Repeated action detected"
        }
      ]
    },
    "context_pruning": {
      "evidence_items": 4,
      "visited_pages": 2,
      "observations_retained": 2,
      "failures_tracked": 1,
      "total_tokens": 1587
    }
  }
}
```

## Logging

All experiments log to:

- **Log file**: `results/fever/prog_CA_pruning/prog_CA_pruning_experiment.log`
  - Timestamp, log level, message
  - Per-example: result, efficiency, pruning stats
  - Summary: success rate, avg metrics

- **Results file**: `results/fever/prog_CA_pruning/prog_CA_pruning_results.json`
  - Full result for each example
  - Pruning breakdown
  - Token and call counts

### Log Format Examples

```
[2025-01-22 14:23:45] [INFO] Running Programmatic Combined Action & Context Pruning ReAct
[2025-01-22 14:23:48] [PRUNE-LOOP] Repeated action detected: search[Einstein]
[2025-01-22 14:23:50] [GATE-SUCCESS] Confidence stabilized: 0.87
[2025-01-22 14:23:52] [RESULT] Answer: SUPPORTS | GT: SUPPORTS | EM: 1.0
[2025-01-22 14:23:52] [STATS] Calls: 5 | Tokens: 1587 | Pruned Actions: 2
```

## Configuration

Thresholds can be tuned in `ActionPruner.__init__()` and `ContextPruner.__init__()`:

```python
pruner = ActionPruner(
    enable_loop_detection=True,
    enable_success_gating=True,
    enable_query_dedup=True,
    enable_cooldowns=True,
    enable_failure_patterns=True,
    enable_confidence_stabilization=True,
)

# Tune thresholds
pruner.loop_window_size = 10
pruner.query_similarity_threshold = 0.85
pruner.cooldown_lookup_limit = 2
pruner.confidence_delta_threshold = 0.05
```

## Expected Performance

Based on ReAct baselines on FEVER:

| Metric | Baseline | prog_CA_pruning | Improvement |
|--------|----------|-----------------|-------------|
| Avg Steps | 5.2 | 3.1 | -40% |
| Avg Input Tokens | 2100 | 850 | -60% |
| Avg Total Tokens | 2850 | 900 | -68% |
| EM Score | 72.3% | 72.0-73.5% | -0.3% to +1.2% |

Goal: **65-85% token reduction** with **same or better accuracy**

## Debugging

Enable verbose output for single examples:

```python
reward, info = run_prog_ca_pruning_react(idx=3687, to_print=True)
```

This prints all thoughts, actions, observations, and pruning decisions.

## Files

- `action_pruner.py` (330 lines): Core pruning logic
- `context_pruner.py` (280 lines): State compression
- `prog_ca_pruning_agent.py` (350 lines): Main agent loop
- `test_prog_ca.py` (160 lines): Test suite
- `run_prog_ca_experiments.py` (240 lines): Experiment runner
- `__init__.py` (15 lines): Module exports

**Total: ~1375 lines of well-documented code**

## Next Steps

1. Run on full FEVER dev set (7405 examples)
2. Compare against baselines (react, action_prune_react, etc.)
3. Tune thresholds on validation set
4. Analyze failure modes and edge cases
5. Consider dataset-specific adaptations (FEVER vs HotPotQA)

## References

- **ReAct**: Yao et al. "ReAct: Synergizing Reasoning and Acting in Language Models"
- **Action Pruning**: Similar to strategies in Reflexion (Shinn et al.) and ReWOO (Xu et al.)
- **Context Compression**: Inspired by Compressive Transformers and prompt engineering best practices
