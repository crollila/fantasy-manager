"""Refresh public inputs, generate pregame forecasts and settle prior picks on app open."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone,timedelta
import json
import threading
from pathlib import Path
import pandas as pd
from app.storage import DATA,now
from app.research_sources import refresh_inputs,read_frame,scoreboard,weather_for,depth_charts
from app.game_model import train_games,game_prediction,kickoff,number,label_pick
from app.context_models import completed_stats,fit_matchups,play_calling,enrich_weather
from app.roster_context import replacement_study,current_rosters
from app.injuries import injury_report
from app.tracking import initialize,saved_games,save_game,settle_games,settle_players,dashboard
from app.game_evidence import observations,fit_expectations,expectations
from app.game_learning import update_learning,apply_learning,archive_artifact,artifact,diagnostics,dashboard as learning_dashboard
from app.player_learning import fit_correction as fit_player_correction, summary as player_correction_summary

import logging
log=logging.getLogger(__name__)
_lock=threading.Lock()


def get_meta(store,key,default=None):
    with store.connect() as c:r=c.execute('SELECT value FROM meta WHERE key=?',(key,)).fetchone()
    return json.loads(r[0]) if r else default


def set_meta(store,key,value):
    with store.connect() as c:c.execute('INSERT OR REPLACE INTO meta VALUES(?,?)',(key,json.dumps(value,allow_nan=False)))


def current_week(schedule,season,as_of):
    relevant=[r for r in schedule.to_dict('records') if r.get('season')==season and r.get('game_type')=='REG' and kickoff(r)]
    upcoming=[r for r in relevant if kickoff(r)>as_of-timedelta(hours=6)]
    return int(min(upcoming,key=kickoff)['week']) if upcoming else 18


def state(store):
    report=get_meta(store,'intelligence-report',{})
    accuracy=dashboard(store)
    # Diagnostics use the full ledger, not just the 300 recent rows shown in the UI.
    with store.connect() as c:results=[json.loads(r[0]) for r in c.execute('SELECT body FROM game_results')]
    from app.game_benchmarks import comparison
    forecasts=saved_games(store)
    return report|{'refresh':get_meta(store,'intelligence-status',{'state':'not_started'}),'accuracy':accuracy,'learning':learning_dashboard(store),'player_learning':player_learning_summary(store),'diagnostics':diagnostics(forecasts,results),'external_comparisons':comparison(forecasts,results)}


def refresh_intelligence(store,season=None,cache_path=None):
    cache_path=Path(cache_path or DATA/'cache');as_of=datetime.now(timezone.utc)
    season=season or (as_of.year if as_of.month>=3 else as_of.year-1)
    if not _lock.acquire(blocking=False):return {'status':'already running'}
    def progress(stage):set_meta(store,'intelligence-status',{'state':'running','stage':stage,'started_at':as_of.isoformat()})
    try:
        progress('Refreshing schedules, statistics, snaps and play-by-play')
        sources=refresh_inputs(season,cache_path)
        schedule=enrich_weather(read_frame(cache_path/'schedules.parquet'),cache_path,season)
        if schedule.empty:raise ValueError('No usable NFL schedule; saved forecasts remain available')
        week=current_week(schedule,season,as_of)
        progress('Checking official game status, injury reports and depth charts')
        from app.data import Cache
        with ThreadPoolExecutor(max_workers=3) as pool:
            depth_job=pool.submit(depth_charts,season,cache_path);health_job=pool.submit(injury_report,Cache(cache_path),True);board_job=pool.submit(scoreboard,season,week,cache_path)
            depth=depth_job.result();health=health_job.result();board=board_job.result()
        matching={str(r.get('espn')).removesuffix('.0'):r for r in schedule.to_dict('records') if r.get('season')==season and r.get('game_type')=='REG'}
        for g in board:
            match=matching.get(g['espn_id'])
            if match:g['game_id']=match['game_id']
        board=[g for g in board if g.get('game_id')]
        from app.market_data import fetch_markets
        markets,market_status=fetch_markets(board);sources.append(market_status)
        # Point-in-time forecasting engine (app.nfl): champion ensemble forecasts for upcoming games.
        from app.nfl.integration import engine_forecasts,refresh_engine_data
        progress('Refreshing the forecasting engine (nflverse raw store, features, champion forecasts)')
        # Build current features first so an automatic refit and the forecasts that follow
        # both see every completed game; the forecast call then reuses that same data.
        engine_data_ok=True
        try:refresh_engine_data(season)
        except Exception as exc:engine_data_ok=False;log.warning('engine data refresh failed: %s',exc)
        from app.auto_refit import maybe_refit
        refit=maybe_refit(store,schedule,season,as_of,progress) if engine_data_ok else {'action':'skipped; engine data unavailable'}
        engine,engine_status=engine_forecasts([g['game_id'] for g in board if g.get('state')=='pre'],season=season,refresh=False);sources.append(engine_status)
        engine_status['automatic_refit']=refit
        progress('Fitting opponent strength, weather and replacement-player models')
        stats,historical_games=completed_stats(cache_path,season,as_of)
        model=train_games(schedule,as_of)
        matchups=fit_matchups(stats,historical_games,as_of)
        league_models={}
        for league in store.leagues():
            if league.season==season and league.mode=='SEASON':
                from app.domain import Player
                from app.scoring_support import player_scoring
                from app.league_sync import sync_metadata
                metadata=sync_metadata(store,league.id)
                scoring_by_position={pos:player_scoring(league,Player(id='scoring',name='Scoring',position=pos),metadata,{})[0].scoring for pos in ('QB','RB','WR','TE','K')}
                league_models[league.id]=fit_matchups(stats,historical_games,as_of,league.scoring,scoring_by_position)
        calling=play_calling(cache_path,season,as_of,historical_games)
        study=replacement_study(cache_path,season,historical_games,stats)
        rosters=current_rosters(depth,health,study,stats,cache_path,as_of,season)
        progress('Reviewing completed picks against team, player and Next Gen Stats evidence')
        # Revisit the weeks in which actual archived forecasts exist, including past seasons.
        previous=saved_games(store);periods={(f['season'],f['week']) for f in previous.values() if datetime.fromisoformat(f['kickoff'].replace('Z','+00:00'))<=datetime.now(timezone.utc)}
        settle_board=list(board)
        for year,past_week in sorted(periods):
            if (year,past_week)==(season,week):continue
            try:
                for g in scoreboard(year,past_week,cache_path):
                    known=next((f for f in previous.values() if f.get('espn_id')==g['espn_id']),None)
                    if known:settle_board.append(g|{'game_id':known['game_id']})
            except Exception:pass
        actuals=observations(cache_path,schedule,season)
        settle_games(store,settle_board,actuals)
        settle_players(store,stats,store.players(season),{g['game_id'] for g in settle_board if g.get('completed')})
        progress('Testing learned corrections and fitting statistical expectations')
        learning=update_learning(store)
        player_correction=fit_player_correction(store,as_of)
        # Refit the promoted point model on every completed week, so the next projections
        # use it without any command, release or restart.
        progress('Refitting the player point projection model on completed weeks')
        try:
            from app.player_model import cached_fit as fit_point_model, current_features
            point_model=fit_point_model(cache_path=cache_path,as_of=as_of)
            from app.player_model import upcoming_players
            point_features=current_features(season,week,cache_path,upcoming_players(store,season,board)) if point_model else None
        except Exception as exc:
            point_model=None;point_features=None;log.warning('player point model unavailable: %s',exc)
        process=fit_expectations(actuals,schedule,as_of)
        # Artifacts preserve coefficients and source hashes without copying personal league data.
        base_artifact=archive_artifact(store,'foundation',{'model':model,'process':process,'sources':sources},as_of)
        active=artifact(store,learning.get('active'));pending=artifact(store,learning.get('pending'))
        progress('Fetching kickoff weather and saving pregame picks')
        with ThreadPoolExecutor(max_workers=3) as pool:weather=dict(zip([g['game_id'] for g in board],pool.map(lambda g:weather_for(g,cache_path),board)))
        forecasts=saved_games(store);games=[]
        for game in board:
            row=matching[game['espn_id']]
            forecast=forecasts.get(game['game_id'])
            if game['state']=='pre' and datetime.fromisoformat(game['kickoff'].replace('Z','+00:00'))>datetime.now(timezone.utc):
                market=markets.get(game['game_id'],{})
                forecast=game_prediction(model,row|market.get('fields',{}),weather.get(game['game_id']),rosters)
                if game['game_id'] in engine:
                    # The engine forecast becomes the published pick; the legacy ridge output is kept for comparison.
                    legacy={k:forecast.get(k) for k in ('home_score','away_score','home_win_probability','away_win_probability','margin','total','model_version','components')}
                    forecast=engine[game['game_id']]|{'legacy_forecast':legacy,'weather':forecast.get('weather')}
                    if market.get('fields'):
                        fields=market['fields'];line=fields.get('spread_line');market_total=fields.get('total_line')
                        if line is not None:forecast.update(market_home_margin=line,ats_pick='home' if forecast['margin']>line else 'away' if forecast['margin']<line else 'pass')
                        if market_total is not None:forecast.update(market_total=market_total,total_pick='over' if forecast['total']>market_total else 'under' if forecast['total']<market_total else 'pass')
                        forecast.update({k:fields.get(k) for k in ('home_spread_odds','away_spread_odds','over_odds','under_odds','home_moneyline','away_moneyline') if k in fields})
                if market:forecast.update(market_source=market['source'],market_quotes=market['quotes'],market_fetched_at=market['fetched_at'])
                forecast.update({k:game[k] for k in ('game_id','espn_id','season','week','kickoff','home_team','away_team','state','source_url')})
                forecast.update(input_as_of=as_of.isoformat(),injury_source_status=health['status'],base_artifact=base_artifact,margin_sd=model['margin_sd'])
                forecast['process_expectations']=expectations(process,row,weather.get(game['game_id']),rosters)
                forecast['personnel_snapshot']={side:rosters.get(game[side+'_team'],{}).get('availability',[]) for side in ('home','away')}
                from app.game_benchmarks import for_game
                forecast['external_benchmarks']=for_game(store,game)
                forecast=apply_learning(forecast,learning,active,pending)
                # The pick and its stated probability must match the team labels published above.
                forecast=label_pick(forecast)
                save_game(store,forecast)
            games.append(game|{'forecast':forecast,'weather':(forecast or {}).get('weather',weather.get(game['game_id'],{}))})
        report={'season':season,'week':week,'updated_at':now(),'games':games,'sources':sources,'model':model,'matchups':matchups,'league_matchups':league_models,'play_calling':calling,'replacement_study':study,'rosters':rosters,'injury_status':health['status'],'process_model':{'metrics':list(process['models']),'method':process['method'],'range_note':process['range_note']},'coverage_notes':['Participation probabilities are separate from points-if-active.','Weather uses city-level forecast coordinates; missing weather receives neutral inputs.','Historical roster effects are observational and shrink toward zero.','Statistical expectations use joint offense/opponent strength; Next Gen Stats cover qualifying players, not every snap.','Current-season play-by-play and player/team stats usually arrive after game days; later corrections update reviews without rewriting picks.','Live route participation and licensed player-prop data are not bundled; unavailable inputs are explicitly missing.','Market schedule lines are a benchmark; no claim of beating closing lines.','Learning tests frozen corrections on future games; insufficient evidence leaves the current scoring model in place.']}
        progress('Projecting and archiving every player with an upcoming game')
        from app.player_archive import archive_projections
        projection_context=report|{'player_correction':player_correction,'player_point_model':point_model,'player_model_features':point_features}
        archive=archive_projections(store,season,week,board,health,projection_context,cache_path,datetime.now(timezone.utc))
        report['player_archive']=archive
        from app.player_model import summary as point_model_summary
        report['player_point_model']=point_model_summary(point_model)
        report['automatic_refit']=refit
        set_meta(store,'intelligence-report',report)
        set_meta(store,'intelligence-status',{'state':'complete','updated_at':now(),'stage':'Forecasts and results updated'})
        return {'season':season,'week':week,'games':len(games),'updated_at':report['updated_at']}
    except Exception as exc:
        set_meta(store,'intelligence-status',{'state':'error','error':str(exc),'updated_at':now()})
        raise
    finally:_lock.release()


def weekly_context(store,league,week):
    report=get_meta(store,'intelligence-report',{})
    if report.get('season')!=league.season or report.get('week')!=week:return {}
    if league.id not in report.get('league_matchups',{}):
        # Newly connected leagues must not silently borrow default PPR matchup effects.
        from app.domain import Player
        from app.scoring_support import player_scoring
        from app.league_sync import sync_metadata
        stats,games=completed_stats(DATA/'cache',league.season,datetime.now(timezone.utc))
        metadata=sync_metadata(store,league.id)
        by_position={pos:player_scoring(league,Player(id='scoring',name='Scoring',position=pos),metadata,{})[0].scoring for pos in ('QB','RB','WR','TE','K')}
        report.setdefault('league_matchups',{})[league.id]=fit_matchups(stats,games,datetime.now(timezone.utc),league.scoring,by_position)
    return report


def player_learning_summary(store):
    """Current state of the weekly player projection correction (fit fresh, never cached)."""
    return player_correction_summary(fit_player_correction(store))
