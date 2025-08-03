# Specifies the name of the virtual environment for the Gemini Agent
param(
  [string]$env_name = "gemini-agent-env"
)

Write-Host "[INFO]: Setting up Gemini Agent environment: $env_name"

# Create Python virtual environment
python -m venv $env_name
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR]: Failed to create virtual environment. Make sure Python 3.11+ is installed and accessible."
    exit 1
}

# Activate the environment
. .\$env_name\Scripts\Activate.ps1

# Upgrade pip and install dependencies
Write-Host "[INFO]: Installing dependencies..."
pip install --upgrade pip
pip install google-generativeai python-dotenv requests beautifulsoup4
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR]: Failed to install dependencies."
    deactivate
    exit 1
}

# Deactivate environment
deactivate

Write-Host "[INFO]: Gemini Agent environment setup complete. Activate it using '.\$env_name\Scripts\Activate.ps1'."
