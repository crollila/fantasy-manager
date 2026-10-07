from datetime import datetime,timezone,timedelta
import pytest
from app.storage import Store
from app.tracking import save_game,settle_games
from app.forecast_history import weeks,week
from app.market_backtest import simulate,head_to_head,buy_price,fee

NOW=datetime(2026,9,7,tzinfo=timezone.utc)


def forecast(gid,home_p,when,week_number=1):
    return {'game_id':gid,'season':2026,'week':week_number,'home_team':'H'+gid,'away_team':'A'+gid,'home_score':24.,'away_score':20.,'pick':'H'+gid if home_p>=.5 else 'A'+gid,
            'home_win_probability':home_p,'away_win_probability':1-home_p-.01,'tie_probability':.01,'margin':4.,'total':44.,'market_home_margin':None,'market_total':None,
            'ats_pick':None,'total_pick':None,'model_version':'test','kickoff':when.isoformat(),'state':'pre','reasons':[]}


def test_week_history_uses_last_pregame_version_and_grades(tmp_path):
    store=Store(tmp_path);kick=NOW+timedelta(days=1)
    save_game(store,forecast('1',.55,kick),as_of=NOW)
    save_game(store,forecast('1',.70,kick),as_of=NOW+timedelta(hours=2))
    save_game(store,forecast('2',.40,kick),as_of=NOW)
    settle_games(store,[{'game_id':'1','season':2026,'week':1,'home_team':'H1','away_team':'A1','home_score':10,'away_score':17,'completed':True},
                        {'game_id':'2','season':2026,'week':1,'home_team':'H2','away_team':'A2','home_score':13,'away_score':20,'completed':True}])
    assert weeks(store)==[{'season':2026,'week':1,'games':2,'graded':2,'correct':1}]
    data=week(store,2026,1,{'1':{'kalshi':{'home_prob_norm':.6}}})
    rows={r['game_id']:r for r in data['games']}
    assert rows['1']['forecast']['home_win_probability']==.70 and rows['1']['forecast']['opening_home_win_probability']==.55
    assert rows['1']['forecast']['versions']==2 and rows['1']['pick_result']=='L' and rows['2']['pick_result']=='W'
    assert data['summary']['wins']==1 and data['summary']['losses']==1
    assert data['summary']['probability']['kalshi']['market']['games']==1


def games(*rows):
    return [{'game_id':str(i),'season':2025,'week':1+i//2,'kickoff':f'2025-09-{10+i:02d}T17:00:00+00:00','home_team':'H','away_team':'A','model_home':p,'home_score':h,'away_score':a,'origin':'walk-forward'} for i,(p,h,a) in enumerate(rows)]


def test_backtest_buys_only_with_edge_and_pays_one_dollar():
    g=games((.70,24,10),(.52,10,24),(.30,17,20))
    prices={'0':{'polymarket':{'home_ask':.60,'away_ask':.42}},'1':{'polymarket':{'home_ask':.50,'away_ask':.52}},'2':{'polymarket':{'home_ask':.45,'away_ask':.57}}}
    out=simulate(g,prices,'polymarket',.05,'flat')
    # Game 0: home edge .10 -> bet home at .60, wins. Game 1: edge .02 -> no bet. Game 2: away edge .70-.57=.13 -> wins.
    assert out['bets']==2 and out['wins']==2
    assert out['profit']==pytest.approx(100/.60-100+100/.57-100,abs=.02)


def test_kalshi_fee_reduces_edge_and_ties_pay_half():
    assert fee('kalshi',.5)==pytest.approx(.0175) and fee('polymarket',.5)==0
    g=games((.56,20,20))
    prices={'0':{'kalshi':{'home_ask':.50,'away_ask':.51}}}
    assert simulate(g,prices,'kalshi',.05,'flat')['bets']==0  # .56-.50-.0175 < .05
    out=simulate(g,prices,'kalshi',.04,'flat')
    assert out['bets']==1 and out['profit']==pytest.approx(100/.5175*.5-100,abs=.02)


def test_mid_price_used_when_no_book_and_head_to_head():
    assert buy_price({'home_prob_norm':.6},'away')==(pytest.approx(.4),'mid')
    g=games((.8,30,10),(.6,10,30))
    prices={'0':{'kalshi':{'home_prob_norm':.7}},'1':{'kalshi':{'home_prob_norm':.4}}}
    h=head_to_head(g,prices,'kalshi')
    assert h['weeks']==1 and h['model']['games']==2 and h['market']['brier']<h['model']['brier'] and h['weeks_model_won']==0
