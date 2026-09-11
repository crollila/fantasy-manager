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
if (!(Test-Path 'storage/nfl/models/registry.json')) { throw 'No NFL champion model to bundle: run `python -m app.nfl backtest` and `python -m app.nfl train` first' }
& ./.build-venv/Scripts/python.exe -m PyInstaller --noconfirm --clean --onedir --name FantasyBackend --paths . --distpath build/backend --workpath build/pyinstaller --specpath build --add-data "${PWD}/frontend/dist;frontend/dist" --add-data "${PWD}/storage/nfl/models;nfl_models" --collect-submodules uvicorn --collect-submodules sklearn --collect-submodules scipy._external --collect-data lightgbm --collect-binaries lightgbm --collect-data xgboost --collect-binaries xgboost --collect-data polars --collect-binaries polars --collect-data duckdb --collect-binaries duckdb --hidden-import lightgbm --hidden-import xgboost --hidden-import polars --hidden-import duckdb --exclude-module xgboost.testing --exclude-module polars.testing --exclude-module lightgbm.testing --exclude-module tkinter --exclude-module matplotlib --exclude-module pytest desktop/backend.py
Check-Exit
Push-Location desktop
npm ci
Check-Exit
node node_modules/electron/install.js
Check-Exit
npm run dist
Check-Exit
Pop-Location
Get-FileHash release/Fantasy-Manager-Setup-0.6.1.exe -Algorithm SHA256
