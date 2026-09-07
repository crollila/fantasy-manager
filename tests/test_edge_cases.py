import numpy as np
import pytest
from app.domain import League,Player,Pick,validate_picks
from app.demo import demo_players
from app.projections import project_all
from app.scoring import feasible_add,lineup
from app.simulation import simulate_drafts,evaluate_candidates
from app.season import trade

def test_pick_gaps_wrong_owner_and_duplicate():
    l=League(id='x',name='X');ps=demo_players(120)
    for picks in ([Pick(number=2,team=1,player_id=ps[0].id)], [Pick(number=1,team=1,player_id=ps[0].id)], [Pick(number=1,team=0,player_id=ps[0].id),Pick(number=2,team=1,player_id=ps[0].id)]):
        with pytest.raises(ValueError):validate_picks(l,picks,ps)

def test_last_pick_must_fill_remaining_slot():
    l=League(id='x',name='X',slots={'QB':1,'RB':1},bench=0)
    rb=Player(id='r',name='R',position='RB');another=Player(id='b',name='B',position='RB');qb=Player(id='q',name='Q',position='QB')
    assert not feasible_add([rb],another,l)
    assert feasible_add([rb],qb,l)

def test_superflex_and_overlapping_slots_simulation():
    l=League(id='x',name='X',teams=4,playoff_teams=4,slots={'QB':1,'RB/WR':1,'WR/TE':1,'SUPERFLEX':1},bench=1)
    ps=demo_players(120);reports=project_all(ps,l,n=64)
    rosters,_=simulate_drafts(ps,reports,l,[],n=16)
    for trial in rosters:
        for roster in trial:
            assert lineup([ps[i] for i in roster],[1]*len(roster),l)[1]==4

def test_candidate_equal_to_baseline_has_zero_effect():
    l=League(id='x',name='X',teams=4,playoff_teams=4,slots={'RB':1,'WR':1},bench=1)
    ps=demo_players(96);reports=project_all(ps,l,n=64)
    candidate=min(ps,key=lambda p:p.adp).id
    result=evaluate_candidates(ps,reports,l,[],[candidate],n=64)
    assert result['results'][0]['championship_delta']==0
    assert result['results'][0]['delta_ci95']==[0,0]

def test_auction_fails_explicitly():
    l=League(id='x',name='X',draft_type='auction')
    with pytest.raises(ValueError):l.owner(1)
    with pytest.raises(ValueError):simulate_drafts([],[],l,[],n=1)

def test_too_small_pool_fails():
    l=League(id='x',name='X')
    with pytest.raises(ValueError):simulate_drafts(demo_players(20),[],l,[],n=1)

def test_trade_rejects_unowned_players():
    l=League(id='x',name='X');ps=demo_players(30)
    with pytest.raises(ValueError):trade(ps,[],l,[],['demo-1'],['demo-2'])
