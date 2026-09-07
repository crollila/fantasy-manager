from fastapi.testclient import TestClient
from app.storage import Store
from app.domain import League
from app.demo import demo_players
import app.api as api

def test_api_auth_idempotency_and_league_isolation(tmp_path,monkeypatch):
    store=Store(tmp_path)
    monkeypatch.setattr(api,'store',store)
    league=League(id='test',name='Test')
    store.save_players(2026,demo_players());store.save_league(league)
    with TestClient(api.app) as client:
        assert client.get('/api/leagues').status_code==401
        assert client.get('/api/bootstrap',headers={'Origin':'https://evil.example'}).status_code==403
        assert client.get('/api/bootstrap',headers={'Origin':'chrome-extension://abc'}).status_code==403
        headers={'X-Local-Token':store.token}
        assert client.get('/api/leagues',headers=headers).status_code==200
        body={'picks':[{'number':1,'team':0,'player_id':'demo-1'}]}
        r=client.post('/api/leagues/test/draft?revision=0',headers=headers,json=body)
        assert r.status_code==200
        assert client.post('/api/leagues/test/draft?revision=0',headers=headers,json=body).status_code==400
        assert client.post('/api/leagues/test/draft?revision=1',headers=headers,json=body).json()['revision']==1
        assert client.get('/api/leagues/test/state',headers=headers).json()['draft']['picks'][0]['player_id']=='demo-1'
