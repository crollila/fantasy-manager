import numpy as np
import pytest
from app.domain import League,Player,Pick,Bonus,validate_picks
from app.identity import Identity
from app.scoring import score_week,lineup,replacement_levels,feasible_add
from app.projections import distribution,project_all,ensemble_weight
from app.draft import survival,quick_recommendations
from app.demo import demo_players,demo_league
from app.simulation import simulate_drafts,season_outcomes,valid_positions
from app.storage import Store
from app.espn import normalize_snapshot

@pytest.fixture
def small():
    league=League(id='test',name='Test',teams=4,my_team=0,slots={'QB':1,'RB':1,'WR':1,'FLEX':1},bench=2,playoff_teams=4)
    players=demo_players(120)
    return league,players,project_all(players,league,n=128)

def test_scoring_ppr_bonuses():
    l=League(id='x',name='X',bonuses=[Bonus(stat='passing_yards',threshold=300,points=3)])
    assert score_week({'passing_yards':300,'passing_tds':2,'passing_interceptions':1,'receptions':5},l)==26
    l.scoring['receptions']=.5
    assert score_week({'receptions':5},l)==2.5

def test_snake_and_next():
    l=League(id='x',name='X',teams=10,my_team=7)
    assert [l.owner(p) for p in (1,10,11,20,21)]==[0,9,9,0,0]
    assert l.next_pick(8)==13
    assert l.next_pick(13)==28

def test_invalid_rules():
    with pytest.raises(ValueError):League(id='x',name='X',teams=4)
    with pytest.raises(ValueError):League(id='x',name='X',slots={'IDP':1})

def test_identity_no_name_fallback():
    ps=[Player(id='a',name='Same Name',position='WR',ids={'espn':'1'}),Player(id='b',name='Same Name',position='RB',ids={'espn':'2'})]
    i=Identity(ps)
    assert i.resolve('espn','2')=='b'
    assert len(i.suggestions('Same Name'))==2
    with pytest.raises(ValueError):i.resolve('espn','3')
    with pytest.raises(ValueError):Identity(ps+[Player(id='c',name='Other',position='TE',ids={'espn':'1'})])

def test_flex_assignment_not_double_count():
    l=League(id='x',name='X',slots={'WR':1,'FLEX':1,'SUPERFLEX':1},bench=0)
    ps=[Player(id='q',name='Q',position='QB'),Player(id='w',name='W',position='WR'),Player(id='r',name='R',position='RB')]
    chosen,total=lineup(ps,[30,20,15],l)
    assert total==65
    assert len({r['player_id'] for r in chosen})==3
    assert next(r for r in chosen if r['slot']=='WR')['player_id']=='w'

def test_replacement_flex(small):
    l,ps,reports=small
    result=replacement_levels(ps,[r['mean'] for r in sorted(reports,key=lambda r:next(i for i,p in enumerate(ps) if p.id==r['id']))],l)
    assert result['RB']['starter']>=result['RB']['bench']

def test_distribution_and_scoring(small):
    l,ps,_=small
    a,r=distribution(ps[0],l,n=1000)
    assert r['p10']<=r['p25']<=r['median']<=r['p75']<=r['p90']
    assert all(0<=r[k]<=1 for k in ('bust','breakout'))
    b,_=distribution(ps[0],l,n=1000)
    np.testing.assert_array_equal(a,b)
    zero=ps[0].model_copy(update={'games':0})
    assert distribution(zero,l,n=100)[1]['mean']==0

def test_survival_conditional():
    p=Player(id='p',name='P',position='WR',adp=40,adp_sd=10)
    assert survival(p,20,25,[],[p])>survival(p,20,40,[],[p])>survival(p,20,60,[],[p])
    assert survival(p,20,40,[],[p])>survival(p,0,40,[],[p])
    assert survival(p.model_copy(update={'adp':None}),0,40,[],[p]) is None

def test_ensemble_fits_errors():
    assert ensemble_weight([10]*40,[-2]*40)==pytest.approx(.8)
    assert ensemble_weight([1]*40,[10]*40)==0
    with pytest.raises(ValueError):ensemble_weight([1],[2])

def test_transaction_and_multileague(tmp_path,small):
    l,ps,_=small
    store=Store(tmp_path);store.save_players(l.season,ps);store.save_league(l)
    other=l.model_copy(update={'id':'other'});store.save_league(other)
    p=Pick(number=1,team=0,player_id=ps[0].id)
    assert store.save_draft(l,[p],expected_revision=0)==1
    assert store.save_draft(l,[p])==1
    with pytest.raises(ValueError):store.save_draft(l,[],expected_revision=1)
    assert store.draft('other')['picks']==[]
    with pytest.raises(ValueError):store.save_draft(l,[p],expected_revision=0)

def test_espn_snapshot(small):
    l,ps,_=small
    result=normalize_snapshot({'picks':[{'number':1,'team':0,'espn_id':ps[0].ids['espn']}]},l,ps)
    assert result[0].player_id==ps[0].id
    with pytest.raises(ValueError):normalize_snapshot({'picks':[{'number':2,'team':0,'espn_id':ps[0].ids['espn']}]},l,ps)

def test_simulator_legality_reproducibility(small):
    l,ps,reports=small
    a,paths=simulate_drafts(ps,reports,l,[],n=32)
    b,_=simulate_drafts(ps,reports,l,[],n=32)
    np.testing.assert_array_equal(a,b)
    for trial in a:
        assert len(set(trial.flatten()))==trial.size
        for roster in trial:
            chosen,total=lineup([ps[i] for i in roster],[1]*len(roster),l)
            assert total==4
    outcome,champions=season_outcomes(a,ps,reports,l)
    assert 0<=outcome['championship_probability']<=outcome['playoff_probability']<=1
    assert len(champions)==32

def test_hall_matches_exact_lineup():
    l=League(id='x',name='X',slots={'RB/WR':2,'WR/TE':2,'SUPERFLEX':1},bench=0)
    rng=np.random.default_rng(12)
    for _ in range(60):
        counts=rng.integers(0,3,6)
        mask=valid_positions(counts[None,:],0,l)[0]
        for pos in range(6):
            updated=counts.copy();updated[pos]+=1
            ps=[Player(id=f'{p}-{j}',name='P',position=('QB','RB','WR','TE','K','DST')[p]) for p,n in enumerate(updated) for j in range(n)]
            _,filled=lineup(ps,[1]*len(ps),l)
            assert bool(mask[pos])==(filled==5)

def test_quick_roster_constraints(small):
    l,ps,reports=small
    recs=quick_recommendations(ps,reports,l,[])
    assert len(recs)>=6
    assert all(r['championship_probability'] is None for r in recs)
    picks=[Pick(number=1,team=0,player_id=recs[0]['id'])]
    assert all(r['id']!=recs[0]['id'] for r in quick_recommendations(ps,reports,l,picks))
