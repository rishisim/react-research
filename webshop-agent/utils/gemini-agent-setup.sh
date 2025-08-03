#!/bin/bash

# Displays information on how to use script
helpFunction()
{
  echo "Usage: $0 [-e gemini-agent-env]"
  echo -e "\t-e gemini-agent-env - Specify the name of the virtual environment for the Gemini Agent"
  exit 1 # Exit script after printing help
}

# Get values of command line flags
while getopts e: flag
do
  case "${flag}" in
    e) env_name=${OPTARG};;
  esac
done

if [ -z "$env_name" ]; then
  echo "[ERROR]: Missing -e flag"
  helpFunction
fi

# Create and activate Python 3.11 virtual environment
python3.11 -m venv $env_name
source $env_name/bin/activate

# Upgrade pip and install dependencies
pip install --upgrade pip
pip install google-genai requests

# Deactivate environment
deactivate

echo "[INFO]: Gemini Agent environment setup complete. Activate it using 'source $env_name/bin/activate'."
