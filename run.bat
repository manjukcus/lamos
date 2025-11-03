@echo off
TITLE LAMOS Installer & Runner
ECHO This script will clone the LAMOS repository, set up a virtual environment,
ECHO install dependencies, and run the application.
ECHO.
ECHO Make sure you have Git and Python 3 installed and added to your PATH.
ECHO.
PAUSE
ECHO.

:: 1. Clone the repository
ECHO [1/5] Cloning repository from manjukcus/lamos...
git clone https://github.com/manjukcus/lamos.git
IF %ERRORLEVEL% NEQ 0 (
    ECHO ERROR: Failed to clone repository.
    ECHO Make sure Git is installed and you have network access.
    ECHO Or, if the repository already exists, this script will try to continue...
)

:: Check if clone was successful or directory already exists
IF NOT EXIST lamos (
    ECHO ERROR: 'lamos' directory not found. Exiting.
    PAUSE
    EXIT /B 1
)
cd lamos
ECHO.

:: 2. Create a virtual environment
ECHO [2/5] Creating virtual environment (venv)...
python -m venv venv
IF %ERRORLEVEL% NEQ 0 (
    ECHO ERROR: Failed to create virtual environment. 
    ECHO Make sure Python 3 is installed and in your PATH.
    PAUSE
    EXIT /B 1
)
ECHO.

:: 3. Activate the virtual environment
ECHO [3/5] Activating virtual environment...
CALL venv\Scripts\activate.bat
ECHO.

:: 4. Install dependencies
ECHO [4/5] Installing required packages from requirements.txt...
pip install -r requirements.txt
IF %ERRORLEVEL% NEQ 0 (
    ECHO ERROR: Failed to install packages. 
    ECHO Check your internet connection and the requirements.txt file.
    PAUSE
    EXIT /B 1
)
ECHO.

:: 5. Run the application
ECHO [5/5] Running LAMOS application (lamos_app.py)...
python lamos_app.py
ECHO.

:: Deactivate virtual environment
ECHO Application finished.
CALL venv\Scripts\deactivate.bat
PAUSE
