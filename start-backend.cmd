@echo off
cd /d "%~dp0"

set "VENV_DIR=.venv"

if not exist "%VENV_DIR%" (
    echo Creating virtual environment...
    python -m venv "%VENV_DIR%"
    call "%VENV_DIR%\Scripts\activate.bat"
    echo Installing dependencies...
    pip install -r requirements.txt
) else (
    call "%VENV_DIR%\Scripts\activate.bat"
    echo Dependencies are already installed in %VENV_DIR%.
)

echo 🚀 Starting Flask Backend on http://localhost:5000
set "PYTHONPATH=%cd%"
python backend\flask_api.py
pause
