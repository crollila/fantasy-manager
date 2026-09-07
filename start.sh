#!/usr/bin/env sh
set -eu
cd "$(dirname "$0")"
test -d .venv || python3 -m venv .venv
if ! test -f .venv/.fantasy-installed; then
  .venv/bin/python -m pip install -r requirements.txt
  touch .venv/.fantasy-installed
fi
test -d frontend/node_modules || npm --prefix frontend ci
npm --prefix frontend run build
exec .venv/bin/python -m uvicorn app.api:app --host 127.0.0.1 --port 8000
