@echo off
cd /d "%~dp0"

if not exist ".venv" (
    echo Creating virtual environment...
    python -m venv .venv
)

if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
) else (
    echo Virtual environment activation script not found.
    pause
    exit /b 1
)

echo Installing dependencies...
pip install -r requirements.txt

echo 🚀 Starting Flask Backend on http://localhost:5000
python flask_api.py
pause
