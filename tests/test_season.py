import pytest
from app.domain import League,Player
from app.projections import project_all
from app.season import start_sit
from app.ownership import save_rosters,current_ownership
from app.storage import Store

def test_bye_and_weekly_override():
    league=League(id='x',name='X',slots={'WR':1},bench=1)
    a=Player(id='a',name='A',position='WR',stats={'receiving_yards':1000},bye=5)
    b=Player(id='b',name='B',position='WR',stats={'receiving_yards':500})
    reports=project_all([a,b],league,n=128)
    assert start_sit([a,b],reports,league,5)['lineup'][0]['player_id']=='b'
    assert start_sit([a,b],reports,league,1,overrides={'b':100})['lineup'][0]['player_id']=='b'

def test_current_ownership_does_not_rewrite_draft(tmp_path):
    store=Store(tmp_path);league=League(id='x',name='X',teams=2,playoff_teams=2)
    ps=[Player(id='a',name='A',position='WR')];store.save_players(2026,ps)
    save_rosters(store,league,{'0':[],'1':['a']})
    picks,updated=current_ownership(store,league,[])
    assert picks[0].team==1 and updated
    assert store.draft('x')['picks']==[]
    with pytest.raises(ValueError):save_rosters(store,league,{'0':['a'],'1':['a']})
