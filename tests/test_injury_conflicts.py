from app.domain import League,Player
from app.weekly import project_week

def test_synced_out_is_not_overridden_by_generic_active_report(tmp_path):
    league=League(id='test',name='Test')
    player=Player(id='p',name='P',position='WR',stats={'receiving_yards':1000})
    report,_=project_week(player,league,1,{'injury_status':'OUT'},{'status':'ACTIVE'},'fresh',tmp_path)
    assert report['injury_status']=='OUT'
    assert report['mean']==0
