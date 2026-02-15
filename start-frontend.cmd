@echo off
cd /d "%~dp0"

echo 🚀 Starting React Frontend on http://localhost:3000
cd frontend

if not exist "node_modules" (
    echo Installing node modules...
    call npm install
)

call npm run dev
pause
