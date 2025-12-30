# Setup Guide (macOS)

This guide walks you through setting up the development environment on macOS.

## 1. Create a Virtual Environment

```bash
# Navigate to the project directory
cd /Users/rishisim/Documents/research/react-research

# Create a virtual environment
python3 -m venv .venv

# Activate the virtual environment
source .venv/bin/activate
```

> **Note**: You only need to run `python3 -m venv .venv` once. After that, just use `source .venv/bin/activate` each time you start working.

## 2. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## 3. Download Datasets

The following scripts in `scripts/download/` can be used to download datasets:

```bash
# Download the FEVEROUS database
python scripts/download_feverous_db.py

# Download SciFact dataset
python scripts/download_scifact.py
```

Check `scripts/download/` for any additional dataset download scripts.

## 4. Set Up API Keys

Store your OpenAI API key in the environment:

```bash
export OPENAI_API_KEY="your-api-key-here"
```

For persistence, add this line to your `~/.zshrc` file.

## 5. Verify Installation

```bash
# Test that imports work
python -c "import torch; import spacy; print('All good!')"
```

---

You're ready to go! 🎉
