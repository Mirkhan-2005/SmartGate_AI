@echo off
setlocal
cd /d "%~dp0"

echo ========================================
echo SmartGate AI - Windows Setup and Launch
 echo ========================================

if not exist ".venv\Scripts\python.exe" (
    echo [1/3] Creating Python 3.11 environment...
    py -3.11 -m venv .venv
    if errorlevel 1 goto ERROR
)

if not exist ".venv\.setup_complete" (
    echo [2/3] Installing libraries. First run may take several minutes...
    ".venv\Scripts\python.exe" -m pip install --upgrade pip
    if errorlevel 1 goto ERROR
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
    if errorlevel 1 goto ERROR
    ".venv\Scripts\python.exe" -m pip check
    if errorlevel 1 goto ERROR
    type nul > ".venv\.setup_complete"
) else (
    echo [2/3] Dependencies already installed.
)

echo [3/3] Launching SmartGate AI...
".venv\Scripts\python.exe" smartgate_app.py
if errorlevel 1 goto ERROR
exit /b 0

:ERROR
echo.
echo Setup or app startup failed. Read the error above.
pause
exit /b 1
