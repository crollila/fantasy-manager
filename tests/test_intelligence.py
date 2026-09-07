from datetime import datetime,timezone,timedelta
import numpy as np
import pandas as pd
from app.availability import participation
from app.domain import Player,League
from app.weekly import project_week
from app.game_model import game_prediction,weather_features,ridge_fit
from app.context_models import fit_matchups
from app.storage import Store
from app.tracking import save_game,settle_games,dashboard,saved_games

NOW=datetime(2026,9,7,tzinfo=timezone.utc)


def test_questionable_does_not_discount_active_points(tmp_path):
    player=Player(id='a',name='A',position='RB',stats={'rushing_yards':1000},games=10)
    league=League(id='l',name='L')
    active,_=project_week(player,league,1,{'injury_status':'ACTIVE'},{},'fresh',tmp_path)
    questionable,_=project_week(player,league,1,{'injury_status':'QUESTIONABLE'},{},'fresh',tmp_path)
    assert active['mean']==questionable['mean']
    assert questionable['play_probability']==.7
    assert questionable['points_if_active']>0
    doubtful,_=project_week(player,league,1,{'injury_status':'DOUBTFUL'},{},'fresh',tmp_path)
    assert doubtful['mean']==0 and doubtful['points_if_active']>0


def test_report_negation_and_official_absence():
    report={'reported_at':NOW.isoformat(),'description':'Not expected to play on Sunday.'}
    assert participation('QUESTIONABLE',report,NOW)['probability']==.05
    assert participation('OUT',{'reported_at':NOW.isoformat(),'description':'Expected to play.'},NOW)['probability']==0
    assert participation('QUESTIONABLE',{'reported_at':(NOW-timedelta(days=8)).isoformat(),'description':'Ruled out'},NOW)['probability']==.7


def test_defense_adjusts_for_opponent_strength():
    stats=[];schedule={};index=0
    # BADLOOK only faces elite offenses; WEAK actually adds 25% to every offense.
    for repeat in range(8):
        for offense in range(12):
            for defense in ['AVERAGE','WEAK','BADLOOK']:
                if defense=='BADLOOK' and offense>=4:continue
                index+=1;gid=str(index);team='O'+str(offense)
                points=(30 if offense<4 else 8)*(1.25 if defense=='WEAK' else 1)
                stats.append({'game_id':gid,'team':team,'opponent_team':defense,'position':'RB','rushing_yards':points*10})
                schedule[gid]={'game_id':gid,'gameday':'2026-08-01','gametime':'13:00','home_team':team,'away_team':defense,'roof':'dome'}
    model=fit_matchups(pd.DataFrame(stats),schedule,NOW)['RB']['defenses']
    assert model['BADLOOK']['raw_points_allowed']>model['WEAK']['raw_points_allowed']
    assert model['WEAK']['adjusted_multiplier']>model['BADLOOK']['adjusted_multiplier']
    assert model['BADLOOK']['adjusted_multiplier']<1.15


def forecast(gid='g',when=None):
    return {'game_id':gid,'season':2026,'week':1,'home_team':'H','away_team':'A','home_score':24.,'away_score':20.,'pick':'H','home_win_probability':.6,'away_win_probability':.39,'tie_probability':.01,'margin':4.,'total':44.,'market_home_margin':7.,'market_total':45.,'ats_pick':'away','total_pick':'under','model_version':'test','kickoff':(when or NOW+timedelta(days=1)).isoformat(),'state':'pre','reasons':[]}


def final(gid='g',home=24,away=20):
    return {'game_id':gid,'season':2026,'week':1,'home_team':'H','away_team':'A','home_score':home,'away_score':away,'completed':True}


def test_cannot_backfill_after_kickoff(tmp_path):
    store=Store(tmp_path)
    assert not save_game(store,forecast(when=NOW),NOW)
    settle_games(store,[final()])
    assert dashboard(store)['games_graded']==0


def test_latest_pregame_pick_and_idempotent_grading(tmp_path):
    store=Store(tmp_path);first=forecast()
    assert save_game(store,first,NOW)
    assert not save_game(store,first,NOW+timedelta(minutes=1))
    second=first|{'pick':'A','home_score':18.,'away_score':20.,'margin':-2.,'total':38.}
    assert save_game(store,second,NOW+timedelta(hours=1))
    settle_games(store,[final()]);settle_games(store,[final()])
    report=dashboard(store)
    assert report['games_graded']==1 and report['straight_up']['losses']==1
    assert report['against_spread']['wins']==1 # Away +7 covers a four-point loss.
    assert report['totals']['wins']==1
    assert report['forecast_versions_saved']==2


def test_ties_pushes_and_missing_lines_are_separate(tmp_path):
    store=Store(tmp_path)
    save_game(store,forecast()|{'market_home_margin':0,'market_total':40},NOW)
    settle_games(store,[final(home=20,away=20)])
    report=dashboard(store)
    assert report['straight_up']['decisions']==0
    assert report['straight_up']['pushes_or_ties']==1
    assert report['against_spread']['pushes_or_ties']==1
    assert report['totals']['pushes_or_ties']==1
    assert report['straight_up']['win_rate'] is None


def test_market_line_does_not_change_independent_score():
    model={'intercept':22.,'coefficients':{'home':2.},'margin_sd':14.,'total_sd':15.}
    row={'home_team':'H','away_team':'A','spread_line':3.,'total_line':45.}
    a=game_prediction(model,row);b=game_prediction(model,row|{'spread_line':14.,'total_line':65.})
    assert a['home_score']==b['home_score'] and a['away_score']==b['away_score']
    assert abs(a['home_win_probability']+a['away_win_probability']+a['tie_probability']-1)<1e-10
    assert a['ats_pick']!=b['ats_pick'] or a['market_home_margin']!=b['market_home_margin']


def test_home_and_weather_factors_are_explicit():
    model={'intercept':22.,'coefficients':{'home':2.,'wind':-3.,'team_wind:H':-1.},'margin_sd':14.,'total_sd':15.}
    row={'home_team':'H','away_team':'A'}
    calm=game_prediction(model,row,{'wind_mph':0,'temperature_f':70})
    windy=game_prediction(model,row,{'wind_mph':20,'temperature_f':70})
    assert calm['home_score']>calm['away_score']
    assert windy['home_score']<calm['home_score']
