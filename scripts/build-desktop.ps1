$ErrorActionPreference = 'Stop'
Set-Location (Join-Path $PSScriptRoot '..')
function Check-Exit { if ($LASTEXITCODE -ne 0) { throw "Build failed ($LASTEXITCODE)" } }
Push-Location frontend
npm ci
Check-Exit
npm run build
Check-Exit
Pop-Location
if (!(Test-Path '.build-venv/Scripts/python.exe')) { python -m venv .build-venv; Check-Exit }
& ./.build-venv/Scripts/python.exe -m pip install -r requirements-desktop.txt
Check-Exit
& ./.build-venv/Scripts/python.exe -m PyInstaller --noconfirm --clean --onedir --name FantasyBackend --paths . --distpath build/backend --workpath build/pyinstaller --specpath build --add-data "${PWD}/frontend/dist;frontend/dist" --collect-submodules uvicorn --collect-submodules sklearn --collect-submodules scipy._external --exclude-module tkinter --exclude-module matplotlib --exclude-module pytest desktop/backend.py
Check-Exit
Push-Location desktop
npm ci
Check-Exit
node node_modules/electron/install.js
Check-Exit
npm run dist
Check-Exit
Pop-Location
Get-FileHash release/Fantasy-Manager-Setup-0.4.0.exe -Algorithm SHA256
