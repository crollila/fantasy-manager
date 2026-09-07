"""Exercise real API behavior using the frozen executable and isolated fixture data."""
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tests'))
from test_weekly import fixture

with socket.socket() as sock:
    sock.bind(('127.0.0.1', 0))
    port = sock.getsockname()[1]
with tempfile.TemporaryDirectory(prefix='fantasy-frozen-') as data:
    log_path = ROOT / 'storage' / 'frozen-test.log'
    log_path.parent.mkdir(exist_ok=True)
    with log_path.open('w') as log:
        process = subprocess.Popen([str(ROOT / 'build/backend/FantasyBackend/FantasyBackend.exe')], env={**os.environ, 'FANTASY_DISABLE_AUTO_REFRESH':'1', 'FANTASY_DATA_DIR': data, 'FANTASY_PORT': str(port)}, stdout=log, stderr=log, creationflags=subprocess.CREATE_NO_WINDOW)
        token = ''
        def api(route, body=None):
            request = urllib.request.Request(f'http://127.0.0.1:{port}/api{route}', data=json.dumps(body).encode() if body is not None else None, headers={'Content-Type': 'application/json', 'X-Local-Token': token})
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.load(response)
        try:
            for _ in range(120):
                if process.poll() is not None:
                    raise RuntimeError(log_path.read_text())
                try:
                    token = api('/bootstrap')['token']
                    break
                except OSError:
                    time.sleep(.25)
            assert token, 'Backend never became ready'
            imported = api('/espn/import', {'league_id': '123', 'my_team_id': 5, 'season': 2026, 'week': 1, 'snapshot': fixture()})
            assert imported['players_synced'] == 2
            assert len(api('/connections')) == 1
            assert api('/intelligence')['accuracy']['games_graded']==0
            assert api('/intelligence/refresh',{})['status']=='complete'
            advice = api('/leagues/123/lineup', {'week': 1, 'risk': 'balanced'})
            assert advice['starters'][0]['player']['name'] == 'B'
            assert advice['bench'][0]['name'] == 'A'
            assert api('/leagues/demo/state')['players']
            with urllib.request.urlopen(f'http://127.0.0.1:{port}/') as response:
                assert b'<div id="root">' in response.read()
            print('Frozen backend: import, connection persistence, injury exclusion, optimization, demo projections and bundled UI passed.')
        finally:
            process.terminate()
            process.wait(timeout=15)
