$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

$venvPath = ".venv"
$pythonExe = "$venvPath\Scripts\python.exe"
$pipExe = "$venvPath\Scripts\pip.exe"

if (-not (Test-Path $pythonExe)) {
    Write-Host "Creating virtual environment..." -ForegroundColor Yellow

    # Find Python installation
    $python = $null
    if (Test-Path "C:\Python314\python.exe") { $python = "C:\Python314\python.exe" }
    elseif (Test-Path "C:\Python313\python.exe") { $python = "C:\Python313\python.exe" }
    elseif (Test-Path "C:\Python312\python.exe") { $python = "C:\Python312\python.exe" }
    elseif (Test-Path "C:\Python311\python.exe") { $python = "C:\Python311\python.exe" }
    else {
        Write-Host "Python not found! Please install Python first." -ForegroundColor Red
        exit 1
    }

    & $python -m venv $venvPath
    Write-Host "Installing dependencies..." -ForegroundColor Yellow
    & $pipExe install -r requirements.txt
}
else {
    Write-Host "Virtual environment found." -ForegroundColor Green
}

Write-Host "Starting Flask Backend on http://localhost:5000" -ForegroundColor Green
$env:PYTHONPATH = (Get-Location).Path
$env:FLASK_DEBUG = "0"
& $pythonExe backend\flask_api.py
