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
            assert api('/intelligence')['learning']['phase']=='collecting'
            # Exercise the new tables and serialized review through the frozen API.
            from datetime import datetime,timezone,timedelta
            from app.storage import Store
            from app.tracking import save_game,settle_games
            from app.game_learning import apply_learning
            local=Store(Path(data));issued=datetime.now(timezone.utc)-timedelta(days=2)
            sample={'game_id':'frozen-review-fixture','season':2099,'week':1,'home_team':'DEN','away_team':'KC','home_score':24.,'away_score':20.,'pick':'DEN','home_win_probability':.6,'away_win_probability':.39,'tie_probability':.01,'margin':4.,'total':44.,'market_home_margin':3.,'market_total':45.,'ats_pick':'home','total_pick':'under','model_version':'synthetic-test','kickoff':(issued+timedelta(hours=1)).isoformat(),'state':'pre','reasons':[],'process_expectations':{'home':{'turnovers':{'expected':1.,'low':0.,'high':2.5,'scatter':1.,'label':'Turnovers lost','unit':'turnovers'}}}}
            sample=apply_learning(sample,{'active':None,'pending':None})
            save_game(local,sample,issued)
            settle_games(local,[sample|{'home_score':17.,'away_score':24.,'completed':True}],{('frozen-review-fixture','DEN'):{'metrics':{'turnovers':4.}}})
            intel=api('/intelligence')
            assert intel['accuracy']['games_graded']==1
            assert intel['accuracy']['results'][0]['review']['comparisons'][0]['status']=='surprise'
            audit=api('/intelligence/games/frozen-review-fixture/audit')
            assert audit['forecasts'][0]['home_score']==24 and audit['reviews'][0]['home_score']==17
            assert api('/intelligence/refresh',{})['status']=='complete'
            advice = api('/leagues/123/lineup', {'week': 1, 'risk': 'balanced'})
            assert advice['starters'][0]['player']['name'] == 'B'
            assert advice['bench'][0]['name'] == 'A'
            assert api('/leagues/demo/state')['players']
            with urllib.request.urlopen(f'http://127.0.0.1:{port}/') as response:
                assert b'<div id="root">' in response.read()
            print('Frozen backend: ESPN import, connection persistence, injury exclusion, optimization, review/learning tables, audit export and bundled UI passed.')
        finally:
            process.terminate()
            process.wait(timeout=15)
