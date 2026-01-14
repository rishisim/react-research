# Project State

> **Last Updated**: 2025-12-29 by Claude
> 
> This document describes the current state of the project. AI agents should read this file when starting work and update it when making significant changes.

## Quick Start

```bash
# Activate environment
source nexus_env/bin/activate

# Run a quick test
python test_setup.py
```

---

## Project Structure

```
react-research/
├── src/
│   ├── agents/           # Agent implementations
│   │   ├── fever/        # FEVER fact verification (4 frameworks)
│   │   ├── feverous/     # FEVEROUS with table support
│   │   ├── hotpotqa/     # Multi-hop QA (4 frameworks)
│   │   ├── musique/      # Multi-hop reasoning
│   │   ├── nexus/        # Nexus agent core
│   │   └── scifact/      # Scientific fact verification
│   └── shared/           # Shared utilities (wikienv, wrappers)
├── scripts/              # Utility scripts (analysis, downloads, debug)
├── data/                 # Datasets (see below)
├── results/              # Experiment results
├── progress_notes/       # Research notes and documentation
├── nexus_env/            # Python 3.10 virtual environment
├── requirements.txt      # Python dependencies
├── SETUP.md              # Setup instructions
└── AGENTS.md             # Agent framework documentation
```

---

## Environment

| Component | Version/Status |
|-----------|----------------|
| Python | 3.10 (via Homebrew) |
| Virtual Environment | `nexus_env/` |
| PyTorch | 2.9.1 |
| spaCy | 3.8.11 |
| NumPy | 2.2.6 |
| Gemini API | ✅ Configured in `.env` |

### API Keys
- `GEMINI_API_KEY` — Required, stored in `.env`

---

## Datasets

| Dataset | File(s) | Status | Size |
|---------|---------|--------|------|
| HotPotQA (dev) | `data/hotpot_dev_distractor_v1.json` | ✅ Ready | 44.2 MB |
| FEVER (train) | `data/train.jsonl` | ✅ Ready | 31.4 MB |
| FEVER (dev) | `data/paper_dev.jsonl` | ✅ Ready | 2.1 MB |
| FEVEROUS (train) | `data/feverous_train.jsonl` | ✅ Ready | 13.7 MB |
| FEVEROUS (dev) | `data/feverous_dev.jsonl` | ✅ Ready | 1.5 MB |
| FEVEROUS DB | `data/feverous_wikiv1.db` | ✅ Ready | ~13 GB |
| SciFact | `data/scifact/` | ✅ Ready | 8.4 MB |
| AVeriTeC | `data/averitec_*.jsonl` | ⚠️ Present but not used | 0.7 MB |

---

## Agent Frameworks

Four agent frameworks are implemented for each task:

1. **ReAct** — Standard single-trace reasoning
2. **CoT-SC** — Chain-of-thought with self-consistency  
3. **Nexus** — Scout → Architect → Adjudicator pipeline
4. **Reflexion** — Self-reflection with iterative improvement

See [`AGENTS.md`](AGENTS.md) and [`progress_notes/frameworks.md`](progress_notes/frameworks.md) for details.

---

## Recent Changes

### 2025-12-29
- Migrated project from Windows to macOS
- Installed Python 3.10 via Homebrew
- Created `nexus_env` virtual environment
- Fixed `requirements.txt` (UTF-16 → UTF-8, removed `pywin32`)
- Moved 19 loose scripts from root to `scripts/`
- Downloaded FEVER train.jsonl and SciFact dataset
- Downloaded FEVEROUS Wikipedia database (~13GB)
- Created `SETUP.md` with setup instructions
- Tested FEVEROUS agents (ReAct + Nexus) successfully

---

## Known Issues

1. **Gym deprecation warning** — Gym is unmaintained; consider migrating to Gymnasium
2. **FEVEROUS table lookups limited** — Need to verify DB utilities are working
3. **Rate limiting** — Gemini API has 15 RPM limit; agents include delays

---

## Running Experiments

### Quick Tests
```bash
# Test environment setup
python test_setup.py

# Test FEVEROUS agents  
python test_feverous.py
```

### Full Experiments
```bash
# FEVER experiments
cd src/agents/fever
python run_reflexion_experiments.py

# HotPotQA experiments
cd src/agents/hotpotqa
python run_hotpotqa_experiments.py
```

---

## Notes for AI Agents

When making changes to this repository:

### Keep the Project Clean

1. **No loose files in root** — Scripts go in `scripts/`, agents go in `src/agents/`
2. **No temporary files** — Delete any temp files, debug outputs, or test artifacts after use
3. **No duplicate code** — Use shared utilities in `src/shared/` when possible
4. **Clean imports** — Remove unused imports before committing
5. **Meaningful names** — Use descriptive file and function names
6. **Delete before creating** — If replacing a file, delete the old one first

### Before Making Changes

1. **Read `PROJECT_STATE.md`** to understand current state
2. **Check dataset availability** before running experiments
3. **Respect API rate limits** (Gemini: 15 RPM)

### After Making Changes

1. **Update `PROJECT_STATE.md`** with what you changed (see `/update-project-state` workflow)
2. **Test changes** using the test scripts before committing
3. **Clean up** any temporary files or test outputs you created
4. **Verify structure** — Run `ls` on root to ensure no new loose files were added
