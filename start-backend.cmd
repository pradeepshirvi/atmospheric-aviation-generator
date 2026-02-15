@echo off
cd /d "%~dp0"

if not exist "backend\.venv" (
    echo Creating virtual environment...
    cd backend
    python -m venv .venv
    cd ..
)

if exist "backend\.venv\Scripts\activate.bat" (
    call backend\.venv\Scripts\activate.bat
) else (
    echo Virtual environment activation script not found.
    pause
    exit /b 1
)

echo Installing dependencies...
cd backend
pip install -r requirements.txt

echo 🚀 Starting Flask Backend on http://localhost:5000
python flask_api.py
pause
