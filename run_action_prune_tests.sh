#!/bin/bash
# Setup script for Action Prune tests

echo "=================================================================="
echo "Action Prune ReAct Agent - Setup & Test Runner"
echo "=================================================================="
echo ""

# Check if .env exists
if [ ! -f .env ]; then
    echo "⚠️  .env file not found!"
    echo ""
    echo "Please create a .env file with your Gemini API key:"
    echo "  1. Copy .env.example to .env"
    echo "  2. Get your API key from: https://aistudio.google.com/app/apikey"
    echo "  3. Replace 'your_api_key_here' with your actual key"
    echo ""
    echo "Then run this script again."
    exit 1
fi

# Check if API key is set
source .env
if [ -z "$GEMINI_API_KEY" ] || [ "$GEMINI_API_KEY" = "your_api_key_here" ]; then
    echo "⚠️  GEMINI_API_KEY not properly configured in .env file"
    echo ""
    echo "Please edit .env and add your API key"
    echo "Get your API key from: https://aistudio.google.com/app/apikey"
    exit 1
fi

echo "✓ API key configured"
echo ""

# Activate virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate
echo "✓ Virtual environment activated"
echo ""

# Install dependencies if needed
if ! python -c "import google.genai" 2>/dev/null; then
    echo "Installing dependencies..."
    pip install -q beautifulsoup4 bs4 google-genai gym python-dotenv requests
    echo "✓ Dependencies installed"
    echo ""
fi

# Run tests
echo "=================================================================="
echo "Running FEVER Action Prune Test"
echo "=================================================================="
cd src/agents/fever
python test_action_prune.py
cd ../../..

echo ""
echo "=================================================================="
echo "Running HotPotQA Action Prune Test"
echo "=================================================================="
cd src/agents/hotpotqa
python test_action_prune.py
cd ../../..

echo ""
echo "=================================================================="
echo "Tests Complete!"
echo "=================================================================="
echo "Results saved in:"
echo "  - results/fever/action_prune/"
echo "  - results/hotpotqa/action_prune/"
echo "=================================================================="
