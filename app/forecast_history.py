"""Week-by-week view of the forecast ledger: the final pregame forecast for every game next to
its actual score (and, when collected, the prediction-market price at kickoff).

Reads only what was saved before each kickoff, so a past week shows exactly what the app
published at the time, never a forecast regenerated afterwards.
"""
import json
import math
from app.tracking import initialize,date
from app.game_model import same_team


def _finals(season):
    """Final scores and the full schedule from the engine's normalized games (best effort)."""
    try:
        import polars as pl
        from app.nfl.config import paths
        from app.nfl.reference.teams import DISPLAY
        frame=pl.read_parquet(paths().normalized/'games.parquet').filter(pl.col('season')==season)
    except Exception:return {}
    out={}
    for r in frame.select(['game_id','season','week','game_type','kickoff_utc','home_team','away_team','home_score','away_score']).to_dicts():
        out[r['game_id']]=r|{'home_team':DISPLAY.get(r['home_team'],r['home_team']),'away_team':DISPLAY.get(r['away_team'],r['away_team']),'kickoff':r['kickoff_utc'].isoformat() if r['kickoff_utc'] else None}
    return out


def _ledger(store):
    """For each game: its final pregame version id, first version id and version count."""
    initialize(store);best={}
    with store.connect() as c:rows=c.execute('SELECT id,game_id,created_at,kickoff FROM game_forecasts ORDER BY id').fetchall()
    for r in rows:
        try:
            if date(r['created_at'])>=date(r['kickoff']):continue
        except ValueError:continue
        item=best.setdefault(r['game_id'],{'first':r['id'],'last':r['id'],'versions':0})
        item['last']=r['id'];item['versions']+=1
    return best


def _bodies(store,ids):
    if not ids:return {}
    with store.connect() as c:
        rows=c.execute(f'SELECT id,created_at,body FROM game_forecasts WHERE id IN ({",".join("?"*len(ids))})',list(ids)).fetchall()
    return {r['id']:json.loads(r['body'])|{'saved_at':r['created_at']} for r in rows}


def weeks(store):
    """Every (season, week) with at least one saved pregame forecast, newest first."""
    ledger=_ledger(store)
    if not ledger:return []
    with store.connect() as c:
        rows=c.execute(f'SELECT game_id,json_extract(body,"$.season") AS season,json_extract(body,"$.week") AS week FROM game_forecasts WHERE id IN ({",".join("?"*len(ledger))})',[v['last'] for v in ledger.values()]).fetchall()
        graded={r[0]:json.loads(r[1]) for r in c.execute('SELECT game_id,body FROM game_results')}
    out={}
    for r in rows:
        if r['season'] is None or r['week'] is None:continue
        key=(int(r['season']),int(r['week']));item=out.setdefault(key,{'season':key[0],'week':key[1],'games':0,'graded':0,'correct':0})
        item['games']+=1;result=graded.get(r['game_id'])
        if result and result.get('winner_result') in ('W','L'):item['graded']+=1;item['correct']+=result['winner_result']=='W'
    return sorted(out.values(),key=lambda w:(w['season'],w['week']),reverse=True)


def _score(pairs):
    pairs=[(min(max(p,1e-4),1-1e-4),y) for p,y in pairs if p is not None]
    if not pairs:return None
    return {'games':len(pairs),'accuracy':sum((p>.5)==(y==1.) for p,y in pairs)/len(pairs),'brier':sum((p-y)**2 for p,y in pairs)/len(pairs),
            'log_loss':-sum(y*math.log(p)+(1-y)*math.log(1-p) for p,y in pairs)/len(pairs)}


def _probability_metrics(rows):
    """Accuracy, two-outcome Brier and log loss of the home-win probability on decided games;
    each market is compared with the model on exactly the games that market priced."""
    decided=[r for r in rows if r.get('final') and r.get('forecast') and r['final']['home_score']!=r['final']['away_score']]
    home_won=lambda r:1. if r['final']['home_score']>r['final']['away_score'] else 0.
    out={'model':_score([(r['forecast']['home_win_probability_norm'],home_won(r)) for r in decided])}
    for source in ('kalshi','polymarket'):
        priced=[r for r in decided if (r.get('markets') or {}).get(source)]
        if priced:out[source]={'market':_score([(r['markets'][source]['home_prob_norm'],home_won(r)) for r in priced]),'model':_score([(r['forecast']['home_win_probability_norm'],home_won(r)) for r in priced])}
    return out


def week(store,season,week_number,markets=None):
    """Final pregame forecast, actual result and kickoff market prices for every game of a week."""
    ledger=_ledger(store);finals=_finals(season);markets=markets or {}
    with store.connect() as c:
        mine=c.execute(f'SELECT id FROM game_forecasts WHERE id IN ({",".join("?"*len(ledger))}) AND json_extract(body,"$.season")=? AND json_extract(body,"$.week")=?',[v['last'] for v in ledger.values()]+[season,week_number]).fetchall() if ledger else []
        graded={r[0]:json.loads(r[1]) for r in c.execute('SELECT game_id,body FROM game_results')}
    last_ids={r[0] for r in mine};game_ids={g for g,v in ledger.items() if v['last'] in last_ids}
    bodies=_bodies(store,[ledger[g][k] for g in game_ids for k in ('first','last')])
    schedule={g:f for g,f in finals.items() if f['week']==week_number and f.get('game_type') in ('REG',None)}
    rows=[]
    for gid in sorted(game_ids|set(schedule)):
        sched=schedule.get(gid,{});result=graded.get(gid);forecast=None
        if gid in game_ids:
            last=bodies[ledger[gid]['last']];first=bodies[ledger[gid]['first']]
            hp,ap=last['home_win_probability'],last['away_win_probability']
            forecast={'home_score':last['home_score'],'away_score':last['away_score'],'pick':last['pick'],'home_win_probability':hp,'away_win_probability':ap,
                      'home_win_probability_norm':hp/(hp+ap) if hp+ap>0 else .5,'pick_win_probability':last.get('pick_win_probability',max(hp,ap)),
                      'opening_home_win_probability':first['home_win_probability'],'margin':last.get('margin'),'total':last.get('total'),
                      'market_home_margin':last.get('market_home_margin'),'model_version':last.get('model_version'),'saved_at':last['saved_at'],'first_saved_at':first['saved_at'],'versions':ledger[gid]['versions']}
            teams={'home_team':last['home_team'],'away_team':last['away_team'],'kickoff':last['kickoff']}
        else:teams={k:sched.get(k) for k in ('home_team','away_team','kickoff')}
        final=None
        if result:final={'home_score':int(result['home_score']),'away_score':int(result['away_score'])}
        elif sched.get('home_score') is not None and sched.get('away_score') is not None:final={'home_score':int(sched['home_score']),'away_score':int(sched['away_score'])}
        row={'game_id':gid,**teams,'forecast':forecast,'final':final,'markets':markets.get(gid) or None}
        if final and forecast:
            h,a=final['home_score'],final['away_score'];winner=teams['home_team'] if h>a else teams['away_team'] if a>h else 'TIE'
            row['winner']=winner;row['pick_result']='T' if winner=='TIE' else 'W' if same_team(winner,forecast['pick']) else 'L'
            row['score_error']=(abs(h-forecast['home_score'])+abs(a-forecast['away_score']))/2
        rows.append(row)
    rows.sort(key=lambda r:(r.get('kickoff') or '',r['game_id']))
    decided=[r for r in rows if r.get('pick_result') in ('W','L')]
    errors=[r['score_error'] for r in rows if r.get('score_error') is not None]
    summary={'games':len(rows),'forecasts':sum(r['forecast'] is not None for r in rows),'finished':sum(r['final'] is not None for r in rows),
             'wins':sum(r['pick_result']=='W' for r in decided),'losses':sum(r['pick_result']=='L' for r in decided),'ties':sum(r.get('pick_result')=='T' for r in rows),
             'score_mae':sum(errors)/len(errors) if errors else None,'probability':_probability_metrics(rows),
             'models':sorted({r['forecast']['model_version'] for r in rows if r['forecast']})}
    return {'season':season,'week':week_number,'games':rows,'summary':summary,
            'note':'Each forecast is the last version saved before that game kicked off. Games without a saved pregame forecast show the final score only.'}
