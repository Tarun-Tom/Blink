@echo off
setlocal EnableDelayedExpansion

:: Change working directory to the project root directory
cd /d "%~dp0"

echo ====================================================
echo Starting Blink...
echo Real-Time Driver Drowsiness Detection
echo ====================================================

:: 1. Validate virtual environment
if not exist ".venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found at .venv\Scripts\activate.bat
    echo Please ensure the .venv folder exists in the project root.
    echo.
    pause
    exit /b 1
)

:: 2. Validate application entry point
if not exist "src\desktop_app.py" (
    echo [ERROR] Application entry point not found at src\desktop_app.py
    echo Please ensure the project files are intact.
    echo.
    pause
    exit /b 1
)

:: 3. Activate the virtual environment
call ".venv\Scripts\activate.bat"
if errorlevel 1 (
    echo [ERROR] Failed to activate virtual environment.
    echo.
    pause
    exit /b 1
)

:: 4. Launch the Blink desktop application
echo Launching Blink desktop interface...
python src\desktop_app.py

:: 5. Keep terminal open if application exits or encounters an error
if errorlevel 1 (
    echo.
    echo [ERROR] Blink exited with code %errorlevel%.
) else (
    echo.
    echo Blink closed normally.
)

pause
