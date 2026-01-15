# Dataset layout

This folder is organized by dataset. Each subfolder contains the local splits currently available.

## Current structure

- averitec/
  - averitec_train.jsonl
  - averitec_dev.jsonl
- fever/
  - paper_dev.jsonl
  - train.jsonl (missing)
- feverous/
  - feverous_train.jsonl
  - feverous_dev.jsonl
- hotpotqa/
  - hotpot_dev_distractor_v1.json
  - hotpot_train_v1.1_simplified.json (missing)
  - hotpot_test_v1_simplified.json (missing)

## Gaps to fill

- FEVER train split: data/fever/train.jsonl
- HotPotQA train/test splits: data/hotpotqa/hotpot_train_v1.1_simplified.json and data/hotpotqa/hotpot_test_v1_simplified.json

## Download helpers

- AVeriTeC: scripts/download/download_averitec.py
- FEVEROUS: scripts/download/download_feverous.py

If you want me to add download scripts for FEVER and HotPotQA, tell me which source URLs to use and I’ll wire them in.
