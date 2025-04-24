@echo off
echo ---------------------------------------
echo SAXO ML CLASSIFIER - ENV SETUP STARTED
echo ---------------------------------------

:: Optional - Create a virtual environment
echo Creating virtual environment (venv)...
python -m venv venv

:: Activate virtual environment
echo Activating environment...
call venv\Scripts\activate

:: Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

:: Install required libraries
echo Installing required Python libraries...
pip install librosa numpy soundfile pyodbc

echo ---------------------------------------
echo ENVIRONMENT SETUP COMPLETE
echo Run this before launching extractor.py
echo ---------------------------------------
pause
