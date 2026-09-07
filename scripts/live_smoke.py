"""Read-only checks against the running real catalog and application."""
import json
import time
import httpx
with httpx.Client(base_url='http://127.0.0.1:8000',timeout=60) as client:
    assert client.get('/').status_code==200
    bootstrap=client.get('/api/bootstrap').json()
    client.headers['X-Local-Token']=bootstrap['token']
    results=[]
    for league in bootstrap['leagues']:
        started=time.monotonic();r=client.get(f'/api/leagues/{league["id"]}/state');r.raise_for_status()
        data=r.json();elapsed=time.monotonic()-started
        assert data['players']
        assert all(p['p10']<=p['median']<=p['p90'] for p in data['players'])
        assert len(data['recommendations'])>=6
        second=time.monotonic();client.get(f'/api/leagues/{league["id"]}/state').raise_for_status()
        results.append({'league':league['id'],'players':len(data['players']),'cold_seconds':elapsed,'warm_seconds':time.monotonic()-second})
    print(json.dumps({'passed':True,'leagues':results},indent=2))
