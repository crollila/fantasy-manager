"""ESPN league snapshots: explicit IDs, exact provider scoring, no cookies stored."""
import json
from datetime import datetime, timezone
import httpx
from app.domain import League, Player
from app.identity import Identity
from app.storage import now

SLOTS={0:'QB',2:'RB',3:'RB/WR',4:'WR',5:'WR/TE',6:'TE',7:'SUPERFLEX',16:'DST',17:'K',23:'FLEX'}
POS={1:'QB',2:'RB',3:'WR',4:'TE',5:'K',16:'DST'}
STATS={0:'attempts',1:'completions',3:'passing_yards',4:'passing_tds',20:'passing_interceptions',23:'carries',24:'rushing_yards',25:'rushing_tds',41:'receptions',42:'receiving_yards',43:'receiving_tds',53:'receptions',58:'targets',62:'two_point_conversions',72:'fumbles_lost',77:'fg_made_40_49',85:'fg_missed',86:'pat_made',88:'pat_missed',95:'def_interceptions',96:'fumble_recovery_opp',98:'def_safeties',99:'def_sacks'}
TEAMS={1:'ATL',2:'BUF',3:'CHI',4:'CIN',5:'CLE',6:'DAL',7:'DEN',8:'DET',9:'GB',10:'TEN',11:'IND',12:'KC',13:'LV',14:'LAR',15:'MIA',16:'MIN',17:'NE',18:'NO',19:'NYG',20:'NYJ',21:'PHI',22:'ARI',23:'PIT',24:'LAC',25:'SF',26:'SEA',27:'TB',28:'WAS',29:'CAR',30:'JAX',33:'BAL',34:'HOU'}

def stat_key(key):return STATS.get(int(key),f'espn_stat_{key}')

def fetch_public(league_id,season,week):
    if not str(league_id).isdigit():raise ValueError('ESPN league ID must be numeric')
    url=f'https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl/seasons/{season}/segments/0/leagues/{league_id}'
    with httpx.Client(timeout=25,follow_redirects=False) as client:
        response=client.get(url,params=[('view',v) for v in ('mSettings','mTeam','mRoster','mMatchup','mStatus')]+[('scoringPeriodId',week)])
    if response.status_code in (401,403):raise ValueError('Private ESPN league: use Sync league in the paired Chrome extension while logged in. No cookies need to be copied.')
    if response.status_code!=200:raise ValueError(f"ESPN league fetch failed ({response.status_code}); verify league ID, season and access")
    try:return response.json()
    except ValueError:raise ValueError("ESPN returned a non-JSON response")

def normalize(payload,my_team_id,season,week,existing):
    if not isinstance(payload,dict) or not payload.get('teams') or not payload.get('settings'):raise ValueError('Incomplete ESPN snapshot: teams and settings are required')
    if int(payload.get('seasonId',season))!=season:raise ValueError('ESPN season mismatch')
    teams=sorted(payload['teams'],key=lambda t:int(t['id']))
    team_map={str(t['id']):i for i,t in enumerate(teams)}
    if str(my_team_id) not in team_map:raise ValueError('Your ESPN team ID is not in this league')
    settings=payload['settings'];rs=settings.get('rosterSettings',{});counts=rs.get('lineupSlotCounts',{})
    unsupported=[k for k,v in counts.items() if v and int(k) not in SLOTS and int(k) not in (20,21)]
    if unsupported:raise ValueError(f'Unsupported ESPN roster slots: {unsupported}')
    slots={SLOTS[int(k)]:int(v) for k,v in counts.items() if int(k) in SLOTS and v}
    if not slots:raise ValueError('Missing ESPN lineup slots')
    scoring_items=settings.get('scoringSettings',{}).get('scoringItems',[])
    if not scoring_items:raise ValueError('Missing ESPN scoring settings')
    scoring={};special=[]
    for item in scoring_items:
        # Positional scoring overrides cannot be collapsed into one coefficient.
        if item.get('pointsOverrides'):special.append(str(item['statId'])+' position overrides')
        points=float(item.get('points',0));key=stat_key(item['statId'])
        scoring[key]=scoring.get(key,0)+points
        if key.startswith('espn_stat_') and points:special.append(str(item['statId']))
    schedule=settings.get('scheduleSettings',{})
    playoff=min(len(teams),int(schedule.get('playoffTeamCount',min(6,len(teams)))))
    regular=min(14,int(schedule.get('matchupPeriodCount',14)))
    league=League(id=str(payload['id']),name=settings.get('name',f'ESPN {payload["id"]}'),season=season,teams=len(teams),my_team=team_map[str(my_team_id)],team_names=[t.get('name') or (' '.join([t.get('location',''),t.get('nickname','')]).strip()) or f'Team {t["id"]}' for t in teams],slots=slots,bench=int(counts.get('20',0)),ir=int(counts.get('21',0)),scoring=scoring,playoff_teams=max(2,playoff),regular_weeks=regular,mode='SEASON',source='ESPN league snapshot',settings_verified=not special)
    identity=Identity(existing);players={p.id:p.model_copy(deep=True) for p in existing};rosters={str(i):[] for i in range(len(teams))};weekly={};current={};issues=[]
    for team in teams:
        index=team_map[str(team['id'])]
        if 'roster' not in team or 'entries' not in team['roster']:raise ValueError(f'Missing complete roster for ESPN team {team["id"]}')
        for entry in team['roster']['entries']:
            pp=entry.get('playerPoolEntry',{});raw=pp.get('player',{});eid=str(raw.get('id',entry.get('playerId','')))
            if not eid or eid=='None':raise ValueError('Roster player missing ESPN ID')
            try:pid=identity.resolve('espn',eid)
            except ValueError:
                pos=POS.get(int(raw.get('defaultPositionId',0)))
                if not pos:raise ValueError(f'Unsupported position for ESPN player {eid}')
                pid=f'espn:{eid}'
                players[pid]=Player(id=pid,name=raw.get('fullName',eid),position=pos,team=TEAMS.get(raw.get('proTeamId'),'FA'),ids={'espn':eid},source='ESPN roster; independent history unavailable',warnings=['No reconciled nflverse history; using ESPN weekly forecast when available'])
            p=players[pid]
            if raw.get('proTeamId') in TEAMS:p.team=TEAMS[raw['proTeamId']]
            rosters[str(index)].append(pid)
            slot_id=int(entry.get('lineupSlotId',20));slot=SLOTS.get(slot_id,'IR' if slot_id==21 else 'BE')
            projection=None
            for stat in raw.get('stats',[]):
                if int(stat.get('seasonId',season))==season and int(stat.get('scoringPeriodId',-1))==week and int(stat.get('statSourceId',-1))==1:
                    projection=stat;break
            applied=projection.get('appliedTotal') if projection is not None else None
            if applied is None and projection is not None:applied=sum(float(v) for v in projection.get('appliedStats',{}).values()) if projection.get('appliedStats') else None
            status=str(raw.get('injuryStatus','UNKNOWN')).upper()
            weekly[pid]={'espn_projection':float(applied) if applied is not None else None,'projected_stats':{stat_key(k):float(v) for k,v in (projection or {}).get('stats',{}).items()},'injury_status':status,'slot':slot,'locked':bool(entry.get('isLocked',False)),'actual_points':entry.get('appliedStatTotal'),'eligible_slots':[SLOTS[s] for s in raw.get('eligibleSlots',[]) if s in SLOTS]}
            if slot not in ('BE','IR'):current.setdefault(str(index),[]).append({'slot':slot,'player_id':pid})
            if applied is None:issues.append(f'{p.name}: ESPN weekly forecast missing')
    flat=[p for ids in rosters.values() for p in ids]
    if len(set(flat))!=len(flat):raise ValueError('Duplicate player across ESPN rosters')
    opponents={}
    for matchup in payload.get('schedule',[]):
        if int(matchup.get('matchupPeriodId',-1))==int(payload.get('status',{}).get('currentMatchupPeriod',week)):
            home=matchup.get('home',{}).get('teamId');away=matchup.get('away',{}).get('teamId')
            if str(home) in team_map and str(away) in team_map:
                opponents[str(team_map[str(home)])]=team_map[str(away)];opponents[str(team_map[str(away)])]=team_map[str(home)]
    meta={'season':season,'week':week,'updated':now(),'team_map':team_map,'weekly':weekly,'current_lineups':current,'opponents':opponents,'scoring_items':scoring_items,'independent_scoring_incomplete':bool(special),'special_scoring':special,'issues':issues}
    return league,list(players.values()),rosters,meta

def save_snapshot(store,payload,my_team_id,season,week):
    league,players,rosters,meta=normalize(payload,my_team_id,season,week,store.players(season))
    # All writes are one transaction: failed snapshot validation never partially updates a league.
    from app.identity import Identity
    Identity(players)
    if any(len(ids)>league.roster_size+league.ir for ids in rosters.values()):raise ValueError('ESPN roster exceeds configured capacity')
    with store.connect() as c:
        c.execute('BEGIN IMMEDIATE')
        c.execute('INSERT INTO catalogs VALUES(?,?,?,1) ON CONFLICT(season) DO UPDATE SET body=excluded.body,updated=excluded.updated,revision=catalogs.revision+1',(season,json.dumps([p.model_dump() for p in players]),now()))
        c.execute('INSERT OR REPLACE INTO leagues VALUES(?,?)',(league.id,league.model_dump_json()))
        c.execute('INSERT OR REPLACE INTO meta VALUES(?,?)',(f'ownership:{league.id}',json.dumps({'rosters':rosters,'updated':meta['updated']})))
        c.execute('INSERT OR REPLACE INTO meta VALUES(?,?)',(f'espn-season:{league.id}',json.dumps(meta)))
    return {'league':league.model_dump(),'players_synced':sum(map(len,rosters.values())),'metadata':{k:v for k,v in meta.items() if k not in ('weekly','scoring_items')}}

def sync_metadata(store,league_id):
    with store.connect() as c:r=c.execute('SELECT value FROM meta WHERE key=?',(f'espn-season:{league_id}',)).fetchone()
    return json.loads(r[0]) if r else {}
