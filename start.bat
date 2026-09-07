@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  python -m venv .venv
  if errorlevel 1 goto fail
)
if not exist ".venv\.fantasy-installed" (
  ".venv\Scripts\python.exe" -m pip install -r requirements.txt
  if errorlevel 1 goto fail
  type nul > ".venv\.fantasy-installed"
)
if not exist "frontend\node_modules" (
  call npm --prefix frontend ci
  if errorlevel 1 goto fail
)
call npm --prefix frontend run build
if errorlevel 1 goto fail
echo Fantasy Manager: http://127.0.0.1:8000
echo Press Ctrl+C to stop. Public data refresh can be started inside the app.
".venv\Scripts\python.exe" -m uvicorn app.api:app --host 127.0.0.1 --port 8000
exit /b %errorlevel%
:fail
echo Startup failed. See the error above. Requires Python 3.12+ and Node 20.19+.
pause
exit /b 1
