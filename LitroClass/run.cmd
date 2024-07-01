@echo off
setlocal

echo Activating virtual environment...

call myenv\Scripts\activate

if %errorlevel% neq 0 (
    echo Failed to activate virtual environment.
    pause
    exit /b %errorlevel%
)

echo Installing libraries and dependencies...
py -m ensurepip
pip3 install -r requirements.txt

echo Running Flask app... Please wait a little!
py app_hf.py

if %errorlevel% neq 0 (
    echo Failed to start Flask app.
    pause
    exit /b %errorlevel%
)

echo Flask app started successfully.
pause
endlocal
