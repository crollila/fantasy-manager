import numpy as np
import pytest
from app.domain import League,Player
from app.weekly import project_week,choose_lineup,lineup_advice
from app.injuries import normalize_injuries,availability
from app.league_sync import normalize,save_snapshot,sync_metadata
from app.storage import Store

def fixture():
    return {'id':123,'seasonId':2026,'settings':{'name':'Fixture league','rosterSettings':{'lineupSlotCounts':{'4':1,'20':1}},'scoringSettings':{'scoringItems':[{'statId':42,'points':.1},{'statId':53,'points':1}]}},'teams':[{'id':5,'name':'Mine','roster':{'entries':[{'lineupSlotId':4,'playerPoolEntry':{'player':{'id':100,'fullName':'A','defaultPositionId':3,'injuryStatus':'OUT','stats':[{'seasonId':2026,'scoringPeriodId':1,'statSourceId':1,'appliedTotal':15}]}}},{'lineupSlotId':20,'playerPoolEntry':{'player':{'id':101,'fullName':'B','defaultPositionId':3,'injuryStatus':'ACTIVE','stats':[{'seasonId':2026,'scoringPeriodId':1,'statSourceId':1,'appliedTotal':12}]}}}]}},{'id':9,'name':'Other','roster':{'entries':[]}}]}

def test_espn_sync_uses_ids_exact_teams_and_week():
    l,ps,rs,meta=normalize(fixture(),5,2026,1,[])
    assert l.my_team==0 and meta['team_map']=={'5':0,'9':1}
    assert rs['0']==['espn:100','espn:101']
    assert meta['weekly']['espn:101']['espn_projection']==12
    assert l.scoring=={'receiving_yards':.1,'receptions':1}

def test_sync_atomic_and_rejects_incomplete(tmp_path):
    store=Store(tmp_path);save_snapshot(store,fixture(),5,2026,1)
    original=store.catalog_meta(2026)['revision']
    bad=fixture();del bad['teams'][1]['roster']
    with pytest.raises(ValueError):save_snapshot(store,bad,5,2026,1)
    assert store.catalog_meta(2026)['revision']==original
    assert sync_metadata(store,'123')['week']==1

def test_out_benched_and_replacement_selected(tmp_path):
    l,ps,rs,meta=normalize(fixture(),5,2026,1,[])
    result=lineup_advice(ps,l,rs['0'],meta,{'status':'fresh','players':{},'season':2026},1,cache_path=tmp_path)
    assert result['starters'][0]['player']['id']=='espn:101'
    assert result['bench'][0]['mean']==0
    assert result['bench'][0]['bust']==1

def test_wrong_week_refuses_stale_projections(tmp_path):
    l,ps,rs,meta=normalize(fixture(),5,2026,1,[])
    with pytest.raises(ValueError):lineup_advice(ps,l,rs['0'],meta,{'status':'fresh'},2,cache_path=tmp_path)

def test_injuries_match_ids_never_names():
    data={'status':'success','injuries':[{'injuries':[{'status':'Out','date':'2026-09-07','athlete':{'displayName':'A','links':[{'href':'https://www.espn.com/nfl/player/_/id/100/a'}]}}]}]}
    assert normalize_injuries(data)['100']['status']=='OUT'
    assert availability('INJURED_RESERVE')==0
    assert availability('UNKNOWN')<1

def test_bye_zero_and_reproducible(tmp_path):
    l=League(id='a',name='A',slots={'WR':1});p=Player(id='x',name='X',position='WR',stats={'receiving_yards':1000},bye=2)
    a,x=project_week(p,l,2,{}, {},'fresh',tmp_path)
    b,y=project_week(p,l,2,{}, {},'fresh',tmp_path)
    assert a['mean']==0 and a['bye'];np.testing.assert_array_equal(x,y)

def test_locked_starter_cannot_be_swapped(tmp_path):
    raw=fixture();raw['teams'][0]['roster']['entries'][0]['isLocked']=True
    raw['teams'][0]['roster']['entries'][0]['appliedStatTotal']=2
    l,ps,rs,meta=normalize(raw,5,2026,1,[])
    result=lineup_advice(ps,l,rs['0'],meta,{'status':'fresh','players':{}},1,cache_path=tmp_path)
    assert result['starters'][0]['player']['id']=='espn:100'

def test_unknown_scoring_uses_exact_espn_forecast(tmp_path):
    raw=fixture();raw['settings']['scoringSettings']['scoringItems'].append({'statId':999,'points':4})
    l,ps,rs,meta=normalize(raw,5,2026,1,[])
    assert not l.settings_verified and meta['independent_scoring_incomplete']
    result=lineup_advice(ps,l,rs['0'],meta,{'status':'fresh','players':{}},1,cache_path=tmp_path)
    assert result['starters'][0]['player']['weights']=={'independent':0,'espn':1}
