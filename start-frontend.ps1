$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

if (-not (Test-Path "node_modules")) {
    Write-Host "Installing node modules..." -ForegroundColor Yellow
    npm install
}

Write-Host "🚀 Starting React Frontend on http://localhost:3000" -ForegroundColor Green
npm run dev
