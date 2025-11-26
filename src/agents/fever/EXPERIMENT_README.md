# FEVER Experiment Runner

A structured system for running FEVER fact verification experiments with organized, timestamped results.

## Features

✨ **Organized Results**: Timestamped directories with clean separation  
📊 **Auto-Summary**: Generates accuracy metrics and statistics  
🔧 **Configurable**: Easy to switch models, adjust sample sizes  
💾 **Progressive Saving**: Results saved after each example (safe for interruptions)  
📝 **Complete Logging**: Stores config, full trajectories, and summaries

## Quick Start

### Test Run (5 examples with gemini-2.5-flash-lite)
```bash
cd src/agents/fever
python experiment_runner.py --num-examples 5
```

### Full Run (custom configuration)
```bash
python experiment_runner.py \
  --model gemini-2.5-flash \
  --num-examples 15 \
  --frameworks baseline multi_trace
```

### Include Reflexion Framework
```bash
python experiment_runner.py \
  --num-examples 10 \
  --frameworks baseline multi_trace reflexion
```

## Command-Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | `gemini-2.5-flash-lite` | Gemini model to use |
| `--num-examples` | `5` | Number of FEVER examples |
| `--frameworks` | `baseline multi_trace` | Which frameworks to run |
| `--seed` | `42` | Random seed for reproducibility |

## Results Structure

Results are saved in `results/fever/TIMESTAMP_nN_MODEL/`:

```
results/fever/20251124_194800_n5_gemini-2.5-flash-lite/
├── config.json           # Run configuration and metadata
├── baseline.json         # Baseline ReAct results (num_traces=1)
├── multi_trace.json      # Multi-trace ReAct results (num_traces=3)
├── reflexion.json        # Reflexion framework results (if run)
└── summary.json          # Aggregate statistics
```

### File Contents

#### config.json
```json
{
  "timestamp": "20251124_194800",
  "model": "gemini-2.5-flash-lite",
  "num_examples": 5,
  "frameworks": ["baseline", "multi_trace"],
  "seed": 42,
  "indices": [1234, 5678, 910, ...],
  "rate_limit_sleep": 4.1
}
```

#### baseline.json / multi_trace.json
```json
[
  {
    "question_idx": 1234,
    "question_text": "The claim to verify...",
    "answer": "SUPPORTS",
    "gt_answer": "SUPPORTS",
    "em": 1.0,
    "f1": 1.0,
    "reward": 1.0,
    "n_calls": 6,
    "n_badcalls": 0,
    "traj": "Thought 1: ...\nAction 1: ..."
  }
]
```

#### summary.json
```json
{
  "baseline": {
    "total_examples": 5,
    "valid_examples": 5,
    "accuracy_em": 0.6000,
    "accuracy_f1": 0.6000,
    "success_count": 3,
    "total_llm_calls": 32,
    "avg_calls_per_example": 6.4
  },
  "multi_trace": {
    "total_examples": 5,
    "valid_examples": 5,
    "accuracy_em": 0.8000,
    "accuracy_f1": 0.8000,
    "success_count": 4,
    "total_llm_calls": 96,
    "avg_calls_per_example": 19.2
  }
}
```

## Rate Limiting

The system respects Gemini API rate limits:

- **gemini-2.5-flash-lite**: 15 RPM (4.1s sleep between calls)
- **gemini-2.5-flash**: 10 RPM (6s sleep - needs update in `fever_agent.py`)

## Estimating Run Time

**Baseline ReAct** (num_traces=1):
- ~6-7 LLM calls per example
- With 4.1s sleep: ~30-35 seconds per example
- 5 examples: ~2.5-3 minutes

**Multi-Trace ReAct** (num_traces=3):
- ~18-21 LLM calls per example  
- With 4.1s sleep: ~80-90 seconds per example
- 5 examples: ~7-8 minutes

**Both frameworks**: ~10-11 minutes for 5 examples

## Environment Setup

Ensure your `.env` file contains:
```
GEMINI_API_KEY="your-api-key-here"
```

## Next Steps

1. ✅ Run test with 5 examples
2. 📊 Review results in `results/fever/TIMESTAMP_...`
3. ⚙️ Switch to `gemini-2.5-flash` for production
4. 🚀 Run larger batch (10-15 examples)
