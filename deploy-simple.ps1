# Simple Deployment Script for Atmospheric Aviation Generator
# Local deployment only

Write-Host "=========================================================================================================" -ForegroundColor Green
Write-Host "                    ATMOSPHERIC AVIATION GENERATOR - DEPLOYMENT" -ForegroundColor Green
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

# Setup Python environment
Write-Host "Setting up Python environment..." -ForegroundColor Yellow
if (-Not (Test-Path ".\.venv")) {
    python -m venv .venv
    Write-Host "✅ Virtual environment created" -ForegroundColor Green
}

# Install Python dependencies
Write-Host "Installing Python dependencies..." -ForegroundColor Yellow
& .\.venv\Scripts\Activate.ps1
if (Test-Path "requirements-py313.txt") {
    pip install -r requirements-py313.txt --quiet
} else {
    pip install flask flask-cors pandas numpy requests --quiet
}
Write-Host "✅ Python dependencies installed" -ForegroundColor Green

# Install Node.js dependencies
Write-Host "Installing Node.js dependencies..." -ForegroundColor Yellow
npm install --silent
Write-Host "✅ Node.js dependencies installed" -ForegroundColor Green

# Create backend start script
$backendScript = @"
cd "$PWD"
& .\.venv\Scripts\Activate.ps1
Write-Host "🚀 Starting Flask Backend on http://localhost:5000" -ForegroundColor Green
python flask_api.py
"@
$backendScript | Out-File -FilePath "start-backend.ps1" -Encoding UTF8

# Create frontend start script  
$frontendScript = @"
cd "$PWD"
Write-Host "🚀 Starting React Frontend on http://localhost:5173" -ForegroundColor Green
npm run dev
"@
$frontendScript | Out-File -FilePath "start-frontend.ps1" -Encoding UTF8

# Start services
Write-Host "Starting backend service..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-ExecutionPolicy", "Bypass", "-File", "$PWD\start-backend.ps1"

Start-Sleep -Seconds 3

Write-Host "Starting frontend service..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-ExecutionPolicy", "Bypass", "-File", "$PWD\start-frontend.ps1"

Write-Host "Waiting for services to start..." -ForegroundColor Yellow
Start-Sleep -Seconds 8

Write-Host ""
Write-Host "=========================================================================================================" -ForegroundColor Green
Write-Host "✅ DEPLOYMENT COMPLETE!" -ForegroundColor Green
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
Write-Host ""
Write-Host "🎯 Next Steps:" -ForegroundColor Cyan
Write-Host "   1. Open http://localhost:5173 in your browser" -ForegroundColor Gray
Write-Host "   2. Try generating some synthetic atmospheric or aviation data" -ForegroundColor Gray
Write-Host "   3. Use the API endpoints to integrate with other applications" -ForegroundColor Gray
Write-Host ""
