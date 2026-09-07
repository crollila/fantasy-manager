"""Current season ownership, kept separate from immutable draft history."""
import json
from app.domain import Pick
from app.storage import now

def save_rosters(store,league,rosters):
    valid={p.id for p in store.players(league.season)}
    if set(rosters)!={str(i) for i in range(league.teams)}:
        raise ValueError('Provide every team index, including empty rosters')
    flattened=[p for ids in rosters.values() for p in ids]
    if len(set(flattened))!=len(flattened) or any(p not in valid for p in flattened):
        raise ValueError('Duplicate or unknown player in ownership snapshot')
    if any(len(ids)>league.roster_size+league.ir for ids in rosters.values()):
        raise ValueError('Roster exceeds bench/start/IR capacity')
    body={'rosters':rosters,'updated':now()}
    with store.connect() as c:
        c.execute('INSERT OR REPLACE INTO meta VALUES(?,?)',(f'ownership:{league.id}',json.dumps(body)))
    return body

def current_ownership(store,league,picks):
    with store.connect() as c:
        row=c.execute('SELECT value FROM meta WHERE key=?',(f'ownership:{league.id}',)).fetchone()
    if not row:return picks,None
    body=json.loads(row[0]);result=[]
    for team,ids in body['rosters'].items():
        for pid in ids:result.append(Pick(number=len(result)+1,team=int(team),player_id=pid))
    return result,body['updated']
