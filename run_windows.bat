@echo off
cd /d "%~dp0"

echo Starting NASCAR dashboard...
echo.

REM Use the Windows Python launcher if available, otherwise fall back to python
py app.py
if errorlevel 1 (
    echo.
    echo py app.py failed. Trying python app.py...
    python app.py
)

echo.
echo If the dashboard started successfully, open:
echo http://localhost:5000
echo.
pause
