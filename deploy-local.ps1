# Local Deployment Script for Atmospheric Aviation Generator
# This script runs the application locally without Docker

Write-Host "Starting Local Deployment of Atmospheric Aviation Generator" -ForegroundColor Green
Write-Host "=============================================================" -ForegroundColor Green

# Check if virtual environment exists, if not create it
if (-Not (Test-Path ".\.venv")) {
    Write-Host "Creating Python virtual environment..." -ForegroundColor Yellow
    python -m venv .venv
}

# Activate virtual environment
Write-Host "Activating Python virtual environment..." -ForegroundColor Yellow
& .\.venv\Scripts\Activate.ps1

# Install Python dependencies
Write-Host "Installing Python dependencies..." -ForegroundColor Yellow
pip install -r requirements.txt

# Install Node.js dependencies if not already installed
if (-Not (Test-Path ".\node_modules")) {
    Write-Host "Installing Node.js dependencies..." -ForegroundColor Yellow
    npm install
}

Write-Host "=============================================================" -ForegroundColor Green
Write-Host "Setup Complete! Starting services..." -ForegroundColor Green
Write-Host "=============================================================" -ForegroundColor Green

Write-Host "Backend API will run on: http://localhost:5000" -ForegroundColor Cyan
Write-Host "Frontend will run on: http://localhost:5173" -ForegroundColor Cyan
Write-Host "" 
Write-Host "To stop the services, press Ctrl+C in both terminal windows" -ForegroundColor Yellow
Write-Host "" 

# Start backend in new PowerShell window
Write-Host "Starting Flask Backend..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PWD'; .\.venv\Scripts\Activate.ps1; python flask_api.py"

# Wait a moment for backend to start
Start-Sleep -Seconds 3

# Start frontend in new PowerShell window  
Write-Host "Starting React Frontend..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PWD'; npm run dev"

Write-Host "=============================================================" -ForegroundColor Green
Write-Host "Deployment Complete!" -ForegroundColor Green
Write-Host "Visit http://localhost:5173 to access your application" -ForegroundColor Cyan
Write-Host "API documentation available at http://localhost:5000" -ForegroundColor Cyan
Write-Host "=============================================================" -ForegroundColor Green
