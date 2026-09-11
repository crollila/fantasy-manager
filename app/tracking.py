"""Append-only pregame forecasts. Never backfill a winning pick after kickoff."""
import hashlib
import json
import math
from datetime import datetime,timezone,timedelta
import numpy as np
from app.game_model import same_team
from app.storage import now


def initialize(store):
    with store.connect() as c:
        c.executescript('''
        CREATE TABLE IF NOT EXISTS game_forecasts(id INTEGER PRIMARY KEY AUTOINCREMENT,game_id TEXT NOT NULL,created_at TEXT NOT NULL,kickoff TEXT NOT NULL,body TEXT NOT NULL,fingerprint TEXT NOT NULL);
        CREATE INDEX IF NOT EXISTS game_forecasts_game ON game_forecasts(game_id,id);
        CREATE TABLE IF NOT EXISTS game_results(game_id TEXT PRIMARY KEY,body TEXT NOT NULL,updated_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS player_forecasts(id INTEGER PRIMARY KEY AUTOINCREMENT,league_id TEXT NOT NULL,player_id TEXT NOT NULL,game_id TEXT NOT NULL,created_at TEXT NOT NULL,kickoff TEXT NOT NULL,body TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS player_results(forecast_id INTEGER PRIMARY KEY,body TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS game_observations(game_id TEXT NOT NULL,team TEXT NOT NULL,body TEXT NOT NULL,PRIMARY KEY(game_id,team));
        CREATE TABLE IF NOT EXISTS game_result_versions(id INTEGER PRIMARY KEY AUTOINCREMENT,game_id TEXT NOT NULL,forecast_id INTEGER NOT NULL,created_at TEXT NOT NULL,body TEXT NOT NULL,fingerprint TEXT NOT NULL,UNIQUE(game_id,forecast_id,fingerprint));
        ''')


def date(value):return datetime.fromisoformat(value.replace('Z','+00:00'))


def save_game(store,game,as_of=None):
    initialize(store);stamp=as_of or datetime.now(timezone.utc)
    if stamp>=date(game['kickoff']) or game.get('state','pre')!='pre':return False
    body=json.dumps(game,sort_keys=True,allow_nan=False);fingerprint=hashlib.sha256(body.encode()).hexdigest()
    with store.connect() as c:
        previous=c.execute('SELECT fingerprint FROM game_forecasts WHERE game_id=? ORDER BY id DESC LIMIT 1',(game['game_id'],)).fetchone()
        if previous and previous[0]==fingerprint:return False
        c.execute('INSERT INTO game_forecasts(game_id,created_at,kickoff,body,fingerprint) VALUES(?,?,?,?,?)',(game['game_id'],stamp.isoformat(),game['kickoff'],body,fingerprint))
    return True


def saved_games(store):
    initialize(store)
    with store.connect() as c:rows=c.execute('SELECT * FROM game_forecasts WHERE id IN (SELECT MAX(id) FROM game_forecasts GROUP BY game_id)').fetchall()
    return {r['game_id']:json.loads(r['body'])|{'forecast_id':r['id'],'saved_at':r['created_at']} for r in rows if date(r['created_at'])<date(r['kickoff'])}


def grade(sign):return 'W' if sign>0 else 'L' if sign<0 else 'P'


def settle_games(store,games,team_actuals=None):
    initialize(store);forecasts=saved_games(store);team_actuals=team_actuals or {}
    for game in games:
        if not game.get('completed') or game['game_id'] not in forecasts:continue
        forecast=forecasts[game['game_id']];h=game['home_score'];a=game['away_score'];margin=h-a
        winner=game['home_team'] if h>a else game['away_team'] if a>h else 'TIE'
        result={'forecast_id':forecast['forecast_id'],'game_id':game['game_id'],'season':game['season'],'week':game['week'],'home_team':game['home_team'],'away_team':game['away_team'],'home_score':h,'away_score':a,'pick':forecast['pick'],'winner':winner,'winner_result':'T' if winner=='TIE' else 'W' if same_team(winner,forecast['pick']) else 'L','predicted_home':forecast['home_score'],'predicted_away':forecast['away_score'],'score_mae':(abs(h-forecast['home_score'])+abs(a-forecast['away_score']))/2,'margin_error':margin-forecast['margin'],'total_error':h+a-forecast['total'],'saved_at':forecast['saved_at'],'model_version':forecast['model_version']}
        probabilities=[forecast['home_win_probability'],forecast['away_win_probability'],forecast.get('tie_probability',0)]
        actual=[int(h>a),int(a>h),int(h==a)];result['brier']=sum((p-y)**2 for p,y in zip(probabilities,actual))
        line=forecast.get('market_home_margin');total=forecast.get('market_total')
        result['ats_result']=None if line is None or forecast.get('ats_pick') not in ('home','away') else grade((margin-line)*(1 if forecast['ats_pick']=='home' else -1))
        result['total_result']=None if total is None or forecast.get('total_pick') not in ('over','under') else grade((h+a-total)*(1 if forecast['total_pick']=='over' else -1))
        result['market_favorite_result']=None if line is None or line==0 or h==a else grade(margin*(1 if line>0 else -1))
        result['market_home_margin']=line;result['market_total']=total
        result['paper_profit']={}
        for category,selection,field in [('ats',forecast.get('ats_pick'),'ats_result'),('total',forecast.get('total_pick'),'total_result')]:
            odds_key=(selection+'_spread_odds') if category=='ats' and selection in ('home','away') else (selection+'_odds') if category=='total' and selection in ('over','under') else None
            odds=forecast.get(odds_key) if odds_key else None
            if odds and result.get(field) in ('W','L','P'):
                result['paper_profit'][category]=0. if result[field]=='P' else -1. if result[field]=='L' else 100/abs(odds) if odds<0 else odds/100

        from app.game_evidence import review_game,normalize
        merged={}
        with store.connect() as c:
            for side in ('home','away'):
                team=normalize(game[side+'_team']);key=(game['game_id'],team)
                old=c.execute('SELECT body FROM game_observations WHERE game_id=? AND team=?',key).fetchone()
                item=json.loads(old[0]) if old else {}
                for field in ('metrics','sources','players','weather'):
                    item[field]=item.get(field,{})|team_actuals.get(key,{}).get(field,{})
                merged[key]=item
                c.execute('INSERT OR REPLACE INTO game_observations VALUES(?,?,?)',(*key,json.dumps(item,allow_nan=False)))
        result['review']=review_game(forecast,game,merged)
        result['base_artifact']=forecast.get('base_artifact')
        result['learning_artifact']=forecast.get('learning',{}).get('active_artifact')
        result['learning_eligible_after']=(date(forecast['kickoff'])+timedelta(days=4)).isoformat()
        body=json.dumps(result,sort_keys=True,allow_nan=False);fingerprint=hashlib.sha256(body.encode()).hexdigest()
        with store.connect() as c:
            c.execute('INSERT OR IGNORE INTO game_result_versions(game_id,forecast_id,created_at,body,fingerprint) VALUES(?,?,?,?,?)',(game['game_id'],forecast['forecast_id'],now(),body,fingerprint))
            result['review_revisions']=c.execute('SELECT COUNT(*) FROM game_result_versions WHERE game_id=?',(game['game_id'],)).fetchone()[0]
            c.execute('INSERT OR REPLACE INTO game_results VALUES(?,?,?)',(game['game_id'],json.dumps(result,allow_nan=False),now()))


def rates(rows,field):
    values=[r[field] for r in rows if r.get(field)]
    wins=values.count('W');losses=values.count('L');pushes=values.count('P')+values.count('T');n=wins+losses
    interval=None
    if n:
        z=1.96;p=wins/n;denom=1+z*z/n;center=(p+z*z/(2*n))/denom;radius=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/denom
        interval=[max(0,center-radius),min(1,center+radius)]
    return {'wins':wins,'losses':losses,'pushes_or_ties':pushes,'win_rate':wins/n if n else None,'decisions':n,'ci95':interval}


def dashboard(store):
    initialize(store)
    with store.connect() as c:
        rows=[json.loads(r[0]) for r in c.execute('SELECT body FROM game_results ORDER BY game_id DESC')]
        player_rows=[json.loads(r[0]) for r in c.execute('SELECT body FROM player_results')]
        count=c.execute('SELECT COUNT(*) FROM game_forecasts').fetchone()[0]
    common=[r for r in rows if r.get('market_favorite_result')]
    common_players=[r for r in player_rows if r.get('espn') is not None]
    paper={k:{'picks':len(v),'units':sum(v),'roi':sum(v)/len(v) if v else None} for k in ('ats','total') for v in [[r.get('paper_profit',{})[k] for r in rows if k in r.get('paper_profit',{})]]}
    return {'paper_results':paper,'straight_up':rates(rows,'winner_result'),'against_spread':rates(rows,'ats_result'),'totals':rates(rows,'total_result'),'market_common_games':{'model':rates(common,'winner_result'),'market_favorite':rates(common,'market_favorite_result')},'score_mae':float(np.mean([r['score_mae'] for r in rows])) if rows else None,'brier_score':float(np.mean([r['brier'] for r in rows])) if rows else None,'brier_definition':'Three-outcome home/away/tie Brier score; lower is better','games_graded':len(rows),'forecast_versions_saved':count,'results':rows[:300],'by_model_version':{v:rates([r for r in rows if r['model_version']==v],'winner_result') for v in {r['model_version'] for r in rows}},'players':{**{event+'_brier':float(np.mean([r[event+'_brier'] for r in player_rows if event+'_brier' in r])) if any(event+'_brier' in r for r in player_rows) else None for event in ('boom','bust')},'graded':len(player_rows),'common_espn_games':len(common_players),'common_model_mae':float(np.mean([abs(r['actual']-r['forecast']) for r in common_players])) if common_players else None,'mae':float(np.mean([abs(r['actual']-r['forecast']) for r in player_rows])) if player_rows else None,'espn_mae':float(np.mean([abs(r['actual']-r['espn']) for r in player_rows if r.get('espn') is not None])) if any(r.get('espn') is not None for r in player_rows) else None,'range_coverage':float(np.mean([r['p10']<=r['actual']<=r['p90'] for r in player_rows])) if player_rows else None},'note':'Only saved pre-kickoff forecasts count. W/L excludes ties and pushes. No bets are placed; accuracy is not evidence of profitability.'}


def save_players(store,league,advice):
    initialize(store);stamp=datetime.now(timezone.utc)
    rows=[s['player'] for s in advice['starters'] if s['player']]+advice['bench']
    with store.connect() as c:
        for p in rows:
            game=p.get('game',{});when=game.get('kickoff');gid=game.get('game_id')
            if not when or not gid or stamp>=date(when):continue
            data={k:p.get(k) for k in ('name','position','mean','points_if_active','espn_projection','independent_projection','p10','p90','boom','bust','boom_threshold','bust_threshold','play_probability','injury_status','context','weights','scoring')}
            data['scoring']=p.get('scoring',league.scoring);data['bonuses']=[b.model_dump() for b in league.bonuses];data['season']=league.season;data['week']=advice['week'];data['model_version']='0.4.0-context-1'
            previous=c.execute('SELECT body FROM player_forecasts WHERE league_id=? AND player_id=? AND game_id=? ORDER BY id DESC LIMIT 1',(league.id,p['id'],gid)).fetchone()
            if previous:
                old=json.loads(previous[0]);p['projection_change']={'previous':old['mean'],'current':p['mean'],'delta':p['mean']-old['mean'],'injury_changed':old.get('injury_status')!=p.get('injury_status'),'context_changed':old.get('context')!=p.get('context')}
            c.execute('INSERT INTO player_forecasts(league_id,player_id,game_id,created_at,kickoff,body) VALUES(?,?,?,?,?,?)',(league.id,p['id'],gid,stamp.isoformat(),when,json.dumps(data,allow_nan=False)))


def settle_players(store,stats,players,completed_games):
    initialize(store)
    if stats.empty:return
    from app.scoring import score_week
    from app.domain import League,Bonus
    mapping={p.id:p.ids.get('gsis',p.id) for p in players}
    lookup={(r['game_id'],str(r['player_id'])):r for r in stats.to_dict('records')}
    with store.connect() as c:
        rows=c.execute('SELECT * FROM player_forecasts WHERE id IN (SELECT MAX(id) FROM player_forecasts GROUP BY league_id,player_id,game_id)').fetchall()
        # A newer pregame version supersedes earlier versions; retain forecast history, grade once.
        valid={r['id'] for r in rows}
        for old in c.execute('SELECT forecast_id FROM player_results').fetchall():
            if old[0] not in valid:c.execute('DELETE FROM player_results WHERE forecast_id=?',(old[0],))
        for r in rows:
            if r['game_id'] not in completed_games or date(r['created_at'])>=date(r['kickoff']):continue
            actual=lookup.get((r['game_id'],mapping.get(r['player_id'],r['player_id'])))
            if actual is None:continue # Missing statistics do not prove a DNP or a zero score.
            body=json.loads(r['body']);scoring=body['scoring']
            known=set(actual)|{'fumbles_lost','two_point_conversions'}
            if any(v and k not in known for k,v in scoring.items()) or any(b['stat'] not in known for b in body.get('bonuses',[])):continue
            actual={k:float(v) for k,v in actual.items() if isinstance(v,(int,float)) and math.isfinite(float(v))}
            actual['fumbles_lost']=sum(actual.get(k,0) for k in ('rushing_fumbles_lost','receiving_fumbles_lost','sack_fumbles_lost'))
            actual['two_point_conversions']=sum(actual.get(k,0) for k in ('passing_2pt_conversions','rushing_2pt_conversions','receiving_2pt_conversions'))
            league=League(id='evaluation',name='Evaluation',scoring=scoring,bonuses=[Bonus(**b) for b in body.get('bonuses',[])])
            result={'forecast_id':r['id'],'player_id':r['player_id'],'position':body['position'],'actual':score_week(actual,league),'forecast':body['mean'],'espn':body.get('espn_projection'),'independent':body.get('independent_projection'),'p10':body['p10'],'p90':body['p90'],'saved_at':r['created_at']}
            for event,comparison in [('boom',lambda a,t:a>t),('bust',lambda a,t:a<t)]:
                if body.get(event) is not None and body.get(event+'_threshold') is not None:
                    result[event+'_brier']=(body[event]-int(comparison(result['actual'],body[event+'_threshold'])))**2
            c.execute('INSERT OR REPLACE INTO player_results VALUES(?,?)',(r['id'],json.dumps(result)))


def blend_weight(store,position):
    initialize(store)
    with store.connect() as c:rows=[json.loads(r[0]) for r in c.execute('SELECT body FROM player_results')]
    rows=[r for r in rows if r.get('position')==position and r.get('espn') is not None and r.get('independent') is not None]
    if len(rows)<50:return .65,{'method':'provisional','observations':len(rows)}
    # Only completed, previously archived predictions can change the blend.
    grid=np.linspace(0,1,21);errors=[np.mean([abs(w*r['independent']+(1-w)*r['espn']-r['actual']) for r in rows]) for w in grid]
    fitted=float(grid[int(np.argmin(errors))]);shrink=len(rows)/(len(rows)+100)
    return .65*(1-shrink)+fitted*shrink,{'method':'past completed forecast MAE, shrunk toward initial weight','observations':len(rows),'fitted_weight':fitted}
