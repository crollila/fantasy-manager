from fastapi.testclient import TestClient
import app.api as api
from app.storage import Store
from test_weekly import fixture


def test_connections_private_and_validated(tmp_path, monkeypatch):
    monkeypatch.setattr(api, 'store', Store(tmp_path / 'app.sqlite3'))
    with TestClient(api.app) as client:
        assert client.get('/api/connections').status_code == 401
        token = client.get('/api/bootstrap').json()['token']
        headers = {'X-Local-Token': token}
        assert client.post('/api/connections', headers=headers, json={'url': 'https://evil.test/?leagueId=123&teamId=5'}).status_code == 400
        link = 'https://fantasy.espn.com/football/team?leagueId=123&teamId=5&seasonId=2026'
        assert client.post('/api/connections', headers=headers, json={'url': link}).status_code == 200
        assert len(client.get('/api/connections', headers=headers).json()) == 1
        response = client.post('/api/espn/import', headers=headers, json={'league_id': '123', 'my_team_id': 5, 'season': 2026, 'week': 1, 'snapshot': fixture()})
        assert response.status_code == 200
        saved = client.get('/api/connections', headers=headers).json()
        assert len(saved) == 1 and saved[0]['url'] == link


def test_desktop_origin_is_exact(tmp_path, monkeypatch):
    monkeypatch.setattr(api, 'store', Store(tmp_path / 'app.sqlite3'))
    monkeypatch.setenv('FANTASY_PORT', '19876')
    monkeypatch.setenv('FANTASY_INSTANCE', 'test-instance')
    with TestClient(api.app) as client:
        assert client.get('/api/bootstrap', headers={'Origin': 'http://127.0.0.1:19876'}).status_code == 200
        assert client.get('/api/bootstrap', headers={'Origin': 'http://127.0.0.1:19877'}).status_code == 403
        assert client.get('/api/bootstrap', headers={'Origin': 'https://evil.test'}).status_code == 403
        assert client.get('/api/health').json()['instance'] == 'test-instance'
