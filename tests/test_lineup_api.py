from fastapi.testclient import TestClient
import app.api as api
from app.storage import Store
from test_weekly import fixture

def test_sync_and_lineup_api(tmp_path,monkeypatch):
    monkeypatch.setattr(api,'store',Store(tmp_path))
    monkeypatch.setattr('app.injuries.injury_report',lambda **kwargs:{'status':'fresh','season':2026,'players':{},'fetched_at':'2026-09-07T00:00:00Z'})
    with TestClient(api.app) as client:
        headers={'X-Local-Token':api.store.token}
        response=client.post('/api/espn/import',headers=headers,json={'league_id':'123','my_team_id':5,'season':2026,'week':1,'snapshot':fixture()})
        assert response.status_code==200,response.text
        result=client.post('/api/leagues/123/lineup',headers=headers,json={'week':1,'risk':'balanced'})
        assert result.status_code==200,result.text
        assert result.json()['starters'][0]['player']['name']=='B'
        assert result.json()['start']==['B'] and result.json()['sit']==['A']
        stale=client.post('/api/leagues/123/lineup',headers=headers,json={'week':2})
        assert stale.status_code==400

def test_import_id_mismatch_is_rejected(tmp_path,monkeypatch):
    monkeypatch.setattr(api,'store',Store(tmp_path))
    with TestClient(api.app) as client:
        result=client.post('/api/espn/import',headers={'X-Local-Token':api.store.token},json={'league_id':'999','my_team_id':5,'season':2026,'week':1,'snapshot':fixture()})
        assert result.status_code==400
