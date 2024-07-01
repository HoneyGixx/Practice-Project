@echo off
setlocal
echo Activating virtual environment...

call myenv\Scripts\activate

echo Installing Python...

.\python-3.11.9-amd64.exe /quiet InstallAllUsers=1 PrependPath=1 Include_test=0
if %errorlevel% neq 0 (
    echo Python failed to install.
    pause
    exit /b %errorlevel%
)
echo Python intalled successfully.
pause
endlocal