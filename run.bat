@echo off
setlocal
TITLE LAMOS Installer and Runner
ECHO This script sets up LAMOS in a Python virtual environment and starts it.
ECHO Requirements: Git and Python 3.10-3.13 (python.org installer, "Add Python to PATH" checked).
ECHO.

:: Run in place when started from inside the repository; otherwise clone or update ./lamos
IF EXIST lamos_app.py GOTO setup
IF EXIST lamos\.git (
    ECHO [1/4] Updating the existing copy in .\lamos ...
    git -C lamos pull
) ELSE (
    ECHO [1/4] Cloning https://github.com/manjukcus/lamos ...
    git clone https://github.com/manjukcus/lamos.git
    IF ERRORLEVEL 1 (
        ECHO ERROR: Could not clone the repository. Check that Git is installed and you are online.
        PAUSE
        EXIT /B 1
    )
)
cd lamos

:setup
IF NOT EXIST venv (
    ECHO [2/4] Creating the virtual environment ...
    python -m venv venv
    IF ERRORLEVEL 1 (
        ECHO ERROR: Could not create the virtual environment. Check that Python 3 is on PATH.
        PAUSE
        EXIT /B 1
    )
)
CALL venv\Scripts\activate.bat

ECHO [3/4] Installing the required packages ...
python -m pip install --upgrade pip >NUL
python -m pip install -r requirements.txt
IF ERRORLEVEL 1 (
    ECHO ERROR: Package installation failed. Check your internet connection.
    PAUSE
    EXIT /B 1
)

ECHO [4/4] Starting LAMOS ...
python lamos_app.py
CALL venv\Scripts\deactivate.bat
PAUSE
