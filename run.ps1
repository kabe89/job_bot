# Quick-start launcher for Windows PowerShell
param([string]$cmd = "serve")

if (-not (Test-Path ".venv")) {
    Write-Host "Creating venv..." -ForegroundColor Cyan
    python -m venv .venv
}
& .\.venv\Scripts\Activate.ps1
pip install -q -r requirements.txt

if (-not (Test-Path ".env")) {
    Copy-Item .env.example .env
    Write-Host "Created .env from template — edit it with your API keys, then re-run." -ForegroundColor Yellow
    exit 0
}

python -m jobbot $cmd
