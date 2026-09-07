"""Exercise normalized ESPN ingestion pick-by-pick through the real API in isolation."""
import json
import os
import sys
import tempfile
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
task_dir=tempfile.TemporaryDirectory(prefix='fantasy-replay-')
os.environ['FANTASY_DATA_DIR']=task_dir.name
from fastapi.testclient import TestClient
from app.api import app,store
from app.demo import demo_players
from app.domain import League,Pick
from app.projections import project_all
from app.simulation import simulate_drafts

def run():
    league=League(id='replay',name='Synthetic ESPN replay',teams=12,my_team=0)
    players=demo_players(360)
    store.save_players(2026,players);store.save_league(league)
    reports=project_all(players,league,n=128)
    rosters,_=simulate_drafts(players,reports,league,[],n=1)
    seen=[0]*league.teams
    picks=[];latencies=[]
    with TestClient(app) as client:
        headers={'X-Local-Token':store.token}
        assert client.get('/api/leagues').status_code==401
        for number in range(1,league.teams*league.roster_size+1):
            team=league.owner(number)
            p=players[int(rosters[0,team,seen[team]])];seen[team]+=1
            picks.append({'number':number,'team':team,'espn_id':p.ids['espn']})
            started=time.monotonic()
            r=client.post('/api/leagues/replay/espn',headers=headers,json={'picks':picks})
            assert r.status_code==200,r.text
            state=client.get('/api/leagues/replay/state',headers=headers)
            assert state.status_code==200,state.text
            assert len(state.json()['draft']['picks'])==number
            assert p.id not in [r['id'] for r in state.json()['recommendations']]
            latencies.append(time.monotonic()-started)
        retry=client.post('/api/leagues/replay/espn',headers=headers,json={'picks':picks})
        assert retry.json()['revision']==len(picks)
        conflict=client.post('/api/leagues/replay/espn',headers=headers,json={'picks':picks[:-1]})
        assert conflict.status_code==400
    result={'picks_replayed':len(picks),'all_ingestions_verified':True,'p95_ingest_and_recommend_seconds':sorted(latencies)[int(len(latencies)*.95)],'max_seconds':max(latencies),'fixture':'synthetic ESPN IDs; not a real logged-in ESPN draft'}
    out=Path(__file__).resolve().parents[1]/'storage'
    out.mkdir(exist_ok=True)
    (out/'replay-report.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    from app.identity import Identity
    identity=Identity(players)
    canonical=[{'number':p['number'],'team':p['team'],'player_id':identity.resolve('espn',p['espn_id'])} for p in picks]
    (out/'demo-replay.json').write_text(json.dumps({'league':league.model_dump(),'draft':{'picks':canonical}},indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
    return result

if __name__=='__main__':run()
