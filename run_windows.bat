@echo off
setlocal
cd /d "%~dp0"

echo ================================
echo Starting NASCAR dashboard
echo ================================
echo.

if not exist "app.py" (
    echo ERROR: app.py was not found in this folder.
    echo Put this file in the same folder as app.py, predict.py, getdata.py, and dashboard.html.
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

REM Keep Playwright's browser attached to this project/venv instead of some random global location.
set "PLAYWRIGHT_BROWSERS_PATH=0"

echo Installing/updating needed packages...
python -m pip install --upgrade pip

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

REM If your code uses Playwright, this is the missing piece: install Chromium for THIS .venv.
python -c "import playwright" >nul 2>nul
if %errorlevel%==0 (
    echo.
    echo Installing/checking Playwright Chromium for this project...
    python -m playwright install chromium
    if errorlevel 1 (
        echo.
        echo ERROR: Playwright Chromium install failed.
        pause
        exit /b 1
    )
) else (
    echo.
    echo Playwright is not installed in this environment; skipping Chromium install.
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
