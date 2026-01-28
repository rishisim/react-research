# Action Prune ReAct Agent - Quick Start

## Setup

1. **Create .env file with your API key:**
   ```bash
   cp .env.example .env
   # Then edit .env and add your Gemini API key
   # Get it from: https://aistudio.google.com/app/apikey
   ```

2. **Run the tests:**
   ```bash
   ./run_action_prune_tests.sh
   ```

## Manual Testing

If you prefer to run tests individually:

```bash
# Activate virtual environment
source venv/bin/activate

# Export API key
export GEMINI_API_KEY="your_key_here"

# Run FEVER test
cd src/agents/fever
python test_action_prune.py

# Run HotPotQA test  
cd ../hotpotqa
python test_action_prune.py
```

## Results

Test results are saved to:
- `results/fever/action_prune/`
- `results/hotpotqa/action_prune/`

Each test creates a JSON file with:
- Full trajectory
- Answer and ground truth
- Evaluation metrics (EM, F1, LLM-judge)
- Number of LLM calls

## What is Action Prune ReAct?

Action Prune ReAct is a ReAct agent with strict rules to prevent common failure modes:

- **No repeats**: Don't use the same Search query twice; don't use the same Lookup keyword twice in a row
- **No search-spam**: Max 2 Search actions in a row; then Lookup required
- **Be specific**: Search must be a concrete entity (not generic words like "born", "author", "city")
- **Evidence-first**: Only Finish if you saw the supporting fact in Observations
- **Multi-hop**: Find bridge entities first, then search and lookup final facts
