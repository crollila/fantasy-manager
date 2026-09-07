from datetime import datetime, timezone, timedelta
import json
import numpy as np
import pandas as pd
import pytest
from app.domain import League, Player
from app.scoring_support import player_scoring
from app.availability import participation
from app.tracking import initialize, settle_players, dashboard, blend_weight, save_game, settle_games
from app.storage import Store
from app.weekly import project_week, choose_lineup, matchup_optimize
from test_intelligence import forecast, final, NOW


def test_espn_units_and_position_overrides_ignore_defense_only_rules():
    league=League(id='l',name='L')
    meta={'scoring_items':[{'statId':8,'points':1},{'statId':28,'points':1},{'statId':48,'points':1},{'statId':53,'points':1,'pointsOverrides':{'4':1.5}},{'statId':132,'points':10}]}
    for position,reception in [('RB',1),('TE',1.5)]:
        player=Player(id='p',name='P',position=position,stats={'rushing_yards':100,'receptions':5})
        rules,incomplete=player_scoring(league,player,meta,{})
        assert not incomplete
        assert rules.scoring=={'passing_yards':.04,'rushing_yards':.1,'receiving_yards':.1,'receptions':reception}


def test_unresolved_injury_is_not_ruled_out():
    value=participation('QUESTIONABLE',{'description':'He has not been ruled out.','reported_at':NOW.isoformat()},NOW)
    assert value['probability']==.7 and value['likely']


def test_live_clock_adds_only_remaining_production(tmp_path):
    player=Player(id='p',name='P',position='RB',team='H',stats={'rushing_yards':1000},games=10)
    league=League(id='l',name='L')
    schedule=pd.DataFrame([{'game_id':'g','season':2026,'week':1,'game_type':'REG','home_team':'H','away_team':'A','gameday':'2026-09-01','gametime':'13:00'}])
    schedule.to_parquet(tmp_path/'schedules.parquet')
    row,_=project_week(player,league,1,{'injury_status':'ACTIVE','actual_points':7},{},'fresh',tmp_path,context={'games':[{'game_id':'g','state':'in','remaining_fraction':.5}]})
    assert row['mean']==pytest.approx(7+row['points_if_active']*.5)
    row,_=project_week(player,league,1,{'injury_status':'ACTIVE','actual_points':7},{},'fresh',tmp_path,context={'games':[{'game_id':'g','completed':True}]})
    assert row['mean']==7 and row['p90']==7


def test_win_objective_can_choose_lower_mean_without_breaking_slots():
    league=League(id='l',name='L',slots={'RB':1})
    rows=[{'id':pid,'name':pid,'position':'RB','mean':mean,'p25':0,'p75':20,'eligible_slots':[],'current_slot':'BE','likely_to_play':True,'play_probability':1,'locked':False} for pid,mean in [('safe',12),('swing',10)]]
    samples=np.array([[12]*10,[20]*5+[0]*5])
    advice,_=choose_lineup(rows,samples,league,'balanced',[])
    assert advice['starters'][0]['player']['id']=='safe'
    result=matchup_optimize(rows,samples,samples,league,[],advice,np.array([15]*10),True)
    assert result['starters'][0]['player']['id']=='swing'
    assert result['estimated_win_probability']==.5


def test_actual_player_grading_uses_saved_scoring_and_is_idempotent(tmp_path):
    store=Store(tmp_path);initialize(store)
    body={'position':'RB','mean':12,'espn_projection':11,'independent_projection':13,'p10':3,'p90':22,'scoring':{'rushing_yards':.1,'receptions':.5},'bonuses':[]}
    with store.connect() as c:
        c.execute('INSERT INTO player_forecasts(league_id,player_id,game_id,created_at,kickoff,body) VALUES(?,?,?,?,?,?)',('l','p','g',NOW.isoformat(),(NOW+timedelta(days=1)).isoformat(),json.dumps(body)))
    stats=pd.DataFrame([{'game_id':'g','player_id':'gsis','rushing_yards':100,'receptions':4}])
    players=[Player(id='p',name='P',position='RB',ids={'gsis':'gsis'})]
    settle_players(store,stats,players,set());assert dashboard(store)['players']['graded']==0
    settle_players(store,stats,players,{'g'});settle_players(store,stats,players,{'g'})
    report=dashboard(store)['players']
    assert report['graded']==1 and report['mae']==0 and report['espn_mae']==1


def test_learned_blend_requires_completed_archive(tmp_path):
    store=Store(tmp_path);initialize(store)
    assert blend_weight(store,'RB')[0]==.65
    with store.connect() as c:
        for i in range(60):c.execute('INSERT INTO player_results VALUES(?,?)',(i,json.dumps({'position':'RB','actual':10,'espn':10,'independent':20})))
    weight,evidence=blend_weight(store,'RB')
    assert 0<weight<.65 and evidence['observations']==60
    assert blend_weight(store,'WR')[0]==.65


def test_paper_record_uses_saved_odds_not_assumed_even_money(tmp_path):
    store=Store(tmp_path)
    save_game(store,forecast()|{'away_spread_odds':-110,'under_odds':105},NOW)
    settle_games(store,[final()])
    report=dashboard(store)['paper_results']
    assert report['ats']['units']==pytest.approx(100/110)
    assert report['total']['units']==1.05
