# PowerShell script to set up a Python environment using uv's native approach
Write-Host "Setting up Python environment for RAG evaluation project..." -ForegroundColor Green

# Check if uv is installed
try {
    $uvVersion = uv --version
    Write-Host "Found uv version: $uvVersion" -ForegroundColor Green
} catch {
    Write-Host "uv is not installed. Please install it first with:" -ForegroundColor Red
    Write-Host "pip install uv" -ForegroundColor Yellow
    exit 1
}

# Create a virtual environment directly with uv
Write-Host "Creating virtual environment..." -ForegroundColor Green
uv venv

# Install all dependencies from requirements.txt
Write-Host "Installing dependencies from requirements.txt..." -ForegroundColor Green
uv pip install -r requirements.txt

# Activate the virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Green
& .\.venv\Scripts\Activate.ps1

Write-Host "Environment setup complete!" -ForegroundColor Green
Write-Host "To activate this environment in the future, run: .\.venv\Scripts\Activate.ps1" -ForegroundColor Cyan 