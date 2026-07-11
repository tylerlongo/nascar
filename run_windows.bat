@echo off
cd /d "%~dp0"

REM Open the dashboard in the default browser after Flask has a moment to start.
start "" cmd /c "timeout /t 2 >nul && start http://127.0.0.1:5000"

REM Start the NASCAR dashboard server.
python app.py

REM Keep this window open if Python exits or shows an error.
pause
