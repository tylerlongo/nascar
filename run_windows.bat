@echo off
setlocal
cd /d "%~dp0"

echo ================================
echo Starting NASCAR dashboard
echo ================================
echo.

if not exist "app.py" (
    echo ERROR: app.py was not found in this folder.
    echo Put this run_windows.bat file in the same folder as app.py, predict.py, getdata.py, and dashboard.html.
    echo.
    pause
    exit /b 1
)

where py >nul 2>nul
if %errorlevel%==0 (
    set "PY=py -3"
) else (
    set "PY=python"
)

if not exist ".venv" (
    echo Creating local Python environment...
    %PY% -m venv .venv
    if errorlevel 1 (
        echo.
        echo ERROR: Could not create the Python environment.
        echo Make sure Python is installed and added to PATH.
        pause
        exit /b 1
    )
)

call ".venv\Scripts\activate.bat"
if errorlevel 1 (
    echo.
    echo ERROR: Could not activate the Python environment.
    pause
    exit /b 1
)

echo Installing/updating needed packages...
python -m pip install --upgrade pip
if errorlevel 1 (
    echo.
    echo ERROR: pip upgrade failed.
    pause
    exit /b 1
)

if exist "requirements.txt" (
    python -m pip install -r requirements.txt
) else (
    python -m pip install flask pandas numpy scikit-learn requests beautifulsoup4 lxml playwright
)

if errorlevel 1 (
    echo.
    echo ERROR: Package installation failed.
    pause
    exit /b 1
)

REM If Playwright is installed in this venv, make sure its Chromium browser is installed too.
python -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('playwright') else 1)" >nul 2>nul
if %errorlevel%==0 (
    echo Installing/checking Playwright Chromium browser...
    python -m playwright install chromium
    if errorlevel 1 (
        echo.
        echo ERROR: Playwright Chromium installation failed.
        pause
        exit /b 1
    )
)

echo.
echo Launching dashboard...
echo A browser tab should open at http://127.0.0.1:5000
echo Keep this black window open while using the dashboard.
echo.

start "" powershell -NoProfile -ExecutionPolicy Bypass -Command "Start-Sleep -Seconds 3; Start-Process 'http://127.0.0.1:5000'"
python app.py

echo.
echo Dashboard stopped.
pause
