# Complete Deployment Script for Atmospheric Aviation Generator
# Handles both Docker and local deployment options

param(
    [string]$DeploymentType = "local"  # Options: "docker", "local"
)

Write-Host "=========================================================================================================" -ForegroundColor Green
Write-Host "                    ATMOSPHERIC AVIATION GENERATOR - DEPLOYMENT SCRIPT" -ForegroundColor Green
Write-Host "=========================================================================================================" -ForegroundColor Green
Write-Host ""

# Function to check if a port is available
function Test-Port {
    param([int]$Port)
    try {
        $tcpConnection = Get-NetTCPConnection -LocalPort $Port -ErrorAction SilentlyContinue
        return $tcpConnection.Count -eq 0
    }
    catch {
        return $true
    }
}

# Function to stop processes on specific ports
function Stop-PortProcesses {
    param([int]$Port)
    Write-Host "Checking for processes on port $Port..." -ForegroundColor Yellow
    $processes = netstat -ano | findstr ":$Port"
    if ($processes) {
        Write-Host "Found processes using port $Port. Attempting to stop..." -ForegroundColor Yellow
        $pids = $processes | ForEach-Object { ($_ -split '\s+')[-1] } | Sort-Object -Unique
        foreach ($pid in $pids) {
            if ($pid -ne "0") {
                try {
                    Stop-Process -Id $pid -Force -ErrorAction SilentlyContinue
                    Write-Host "Stopped process with PID: $pid" -ForegroundColor Green
                }
                catch {
                    Write-Host "Could not stop process with PID: $pid" -ForegroundColor Red
                }
            }
        }
        Start-Sleep -Seconds 2
    }
}

if ($DeploymentType -eq "docker") {
    Write-Host "🐳 DOCKER DEPLOYMENT SELECTED" -ForegroundColor Cyan
    Write-Host "=========================================================================================================" -ForegroundColor Green
    
    # Check Docker
    try {
        docker --version | Out-Null
        Write-Host "✅ Docker is installed" -ForegroundColor Green
    }
    catch {
        Write-Host "❌ Docker is not installed or not accessible" -ForegroundColor Red
        exit 1
    }
    
    # Stop any existing containers
    Write-Host "Stopping existing containers..." -ForegroundColor Yellow
    docker-compose down --remove-orphans 2>$null
    
    # Clean up ports
    Stop-PortProcesses -Port 5000
    Stop-PortProcesses -Port 80
    Stop-PortProcesses -Port 3000
    
    # Build and run with Docker Compose
    Write-Host "Building and starting Docker containers..." -ForegroundColor Yellow
    $dockerResult = docker-compose up --build -d
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Docker deployment successful!" -ForegroundColor Green
        Write-Host ""
        Write-Host "🌐 Your application is available at:" -ForegroundColor Cyan
        Write-Host "   Frontend: http://localhost" -ForegroundColor Yellow
        Write-Host "   Backend API: http://localhost:5000" -ForegroundColor Yellow
        Write-Host ""
        Write-Host "📋 Container status:" -ForegroundColor Cyan
        docker ps
    }
    else {
        Write-Host "❌ Docker deployment failed. Falling back to local deployment..." -ForegroundColor Red
        $DeploymentType = "local"
    }
}

if ($DeploymentType -eq "local") {
    Write-Host "🏠 LOCAL DEPLOYMENT SELECTED" -ForegroundColor Cyan
    Write-Host "=========================================================================================================" -ForegroundColor Green
    
    # Check Python
    try {
        $pythonVersion = python --version 2>&1
        Write-Host "✅ Python is installed: $pythonVersion" -ForegroundColor Green
    }
    catch {
        Write-Host "❌ Python is not installed or not accessible" -ForegroundColor Red
        exit 1
    }
    
    # Check Node.js
    try {
        $nodeVersion = node --version
        Write-Host "✅ Node.js is installed: $nodeVersion" -ForegroundColor Green
    }
    catch {
        Write-Host "❌ Node.js is not installed or not accessible" -ForegroundColor Red
        exit 1
    }
    
    # Clean up ports
    Stop-PortProcesses -Port 5000
    Stop-PortProcesses -Port 5173
    
    # Setup Python environment
    Write-Host "Setting up Python environment..." -ForegroundColor Yellow
    if (-Not (Test-Path ".\.venv")) {
        python -m venv .venv
        Write-Host "✅ Virtual environment created" -ForegroundColor Green
    }
    
    # Activate virtual environment and install dependencies
    Write-Host "Installing Python dependencies..." -ForegroundColor Yellow
    & .\.venv\Scripts\Activate.ps1
    
    # Use the compatible requirements file
    if (Test-Path "requirements-py313.txt") {
        pip install -r requirements-py313.txt --quiet
    } else {
        pip install -r requirements.txt --quiet
    }
    Write-Host "✅ Python dependencies installed" -ForegroundColor Green
    
    # Install Node.js dependencies
    Write-Host "Installing Node.js dependencies..." -ForegroundColor Yellow
    npm install --silent
    Write-Host "✅ Node.js dependencies installed" -ForegroundColor Green
    
    # Create start scripts for background processes
    Write-Host "Creating service scripts..." -ForegroundColor Yellow
    
    # Backend start script
    @"
cd "$PWD"
& .\.venv\Scripts\Activate.ps1
Write-Host "🚀 Starting Flask Backend on http://localhost:5000" -ForegroundColor Green
python flask_api.py
"@ | Out-File -FilePath "start-backend.ps1" -Encoding UTF8
    
    # Frontend start script
    @"
cd "$PWD"
Write-Host "🚀 Starting React Frontend on http://localhost:5173" -ForegroundColor Green
npm run dev
"@ | Out-File -FilePath "start-frontend.ps1" -Encoding UTF8
    
    # Start backend service
    Write-Host "Starting backend service..." -ForegroundColor Yellow
    Start-Process powershell -ArgumentList "-NoExit", "-ExecutionPolicy", "Bypass", "-File", "$PWD\start-backend.ps1"
    
    # Wait for backend to start
    Start-Sleep -Seconds 5
    
    # Start frontend service
    Write-Host "Starting frontend service..." -ForegroundColor Yellow
    Start-Process powershell -ArgumentList "-NoExit", "-ExecutionPolicy", "Bypass", "-File", "$PWD\start-frontend.ps1"
    
    # Wait for services to start
    Write-Host "Waiting for services to initialize..." -ForegroundColor Yellow
    Start-Sleep -Seconds 8
    
    # Test services
    Write-Host "Testing services..." -ForegroundColor Yellow
    
    # Test backend
    try {
        $backendResponse = Invoke-RestMethod -Uri "http://localhost:5000" -Method GET -TimeoutSec 5
        Write-Host "✅ Backend is responding" -ForegroundColor Green
        Write-Host "   API: $($backendResponse.project)" -ForegroundColor Gray
    }
    catch {
        Write-Host "⚠️  Backend may still be starting up..." -ForegroundColor Yellow
    }
    
    # Test frontend
    try {
        $frontendResponse = Invoke-WebRequest -Uri "http://localhost:5173" -UseBasicParsing -TimeoutSec 5
        Write-Host "✅ Frontend is responding" -ForegroundColor Green
    }
    catch {
        Write-Host "⚠️  Frontend may still be starting up..." -ForegroundColor Yellow
    }
    
    Write-Host ""
    Write-Host "=========================================================================================================" -ForegroundColor Green
    Write-Host "✅ LOCAL DEPLOYMENT COMPLETE!" -ForegroundColor Green
    Write-Host "=========================================================================================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "🌐 Your application should be available at:" -ForegroundColor Cyan
    Write-Host "   🎨 Frontend (React Dashboard): http://localhost:5173" -ForegroundColor Yellow
    Write-Host "   🔧 Backend API (Flask):        http://localhost:5000" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "📚 API Endpoints:" -ForegroundColor Cyan
    Write-Host "   GET  /                          - API documentation" -ForegroundColor Gray
    Write-Host "   POST /api/generate/radiosonde   - Generate atmospheric data" -ForegroundColor Gray
    Write-Host "   POST /api/generate/aviation     - Generate aviation data" -ForegroundColor Gray
    Write-Host "   POST /api/generate/combined     - Generate combined dataset" -ForegroundColor Gray
    Write-Host ""
    Write-Host "🔧 Management:" -ForegroundColor Cyan
    Write-Host "   - Two PowerShell windows opened for backend and frontend" -ForegroundColor Gray
    Write-Host "   - Press Ctrl+C in each window to stop the services" -ForegroundColor Gray
    Write-Host "   - Or close the terminal windows to stop the services" -ForegroundColor Gray
    Write-Host ""
    Write-Host "📁 Additional files created:" -ForegroundColor Cyan
    Write-Host "   - start-backend.ps1    (Backend start script)" -ForegroundColor Gray
    Write-Host "   - start-frontend.ps1   (Frontend start script)" -ForegroundColor Gray
    Write-Host "   - requirements-py313.txt (Python 3.13 compatible requirements)" -ForegroundColor Gray
    Write-Host ""
    Write-Host "🎯 Next Steps:" -ForegroundColor Cyan
    Write-Host "   1. Open http://localhost:5173 in your browser" -ForegroundColor Gray
    Write-Host "   2. Try generating some synthetic atmospheric or aviation data" -ForegroundColor Gray
    Write-Host "   3. Use the API endpoints to integrate with other applications" -ForegroundColor Gray
    Write-Host ""
}
