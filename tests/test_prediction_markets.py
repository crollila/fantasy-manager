"""Pure parsing/matching helpers of app.prediction_markets. No network."""
from datetime import date,datetime,timezone
from app import prediction_markets as pm


def market(ticker,**extra):return {'ticker':ticker,'event_ticker':ticker.rsplit('-',1)[0],**extra}


def test_team_codes_map_to_nflverse():
    assert [pm.kalshi_team(c) for c in ('LAR','LA','JAC','WSH','WAS','KC','XXX')]==['LA','LA','JAX','WAS','WAS','KC',None]
    assert [pm.poly_team(c) for c in ('lar','la','las','wsh','jax','sf','zzz')]==['LA','LA','LV','WAS','JAX','SF',None]


def test_kalshi_event_date_and_grouping():
    assert pm.kalshi_event_date('KXNFLGAME-26OCT05ATLNO')==date(2026,10,5)
    assert pm.kalshi_event_date('KXSB-26') is None
    idx=pm.kalshi_events([market('KXNFLGAME-26OCT05ATLNO-ATL'),market('KXNFLGAME-26OCT05ATLNO-NO'),
                          market('KXNFLGAME-25SEP28LARIND-LAR'),market('KXNFLGAME-25SEP28LARIND-IND'),market('KXNFLGAME-25OCT01JACKC-JAC')])
    assert set(idx)=={(frozenset({'ATL','NO'}),date(2026,10,5)),(frozenset({'LA','IND'}),date(2025,9,28))}  # one-sided event dropped
    assert idx[(frozenset({'LA','IND'}),date(2025,9,28))]['markets']['LA']['ticker']=='KXNFLGAME-25SEP28LARIND-LAR'


def test_match_uses_eastern_date_and_prefers_exact_day():
    idx={(frozenset({'ATL','NO'}),date(2026,10,5)):'exact',(frozenset({'ATL','NO'}),date(2026,10,6)):'next'}
    # Monday night 00:15 UTC on Oct 6 is Oct 5 in US Eastern
    assert pm.et_date('2026-10-06T00:15:00Z')==date(2026,10,5)
    assert pm.match_events(idx,'NO','ATL','2026-10-06T00:15:00Z')==['exact','next']
    assert pm.match_event(idx,'ATL','NO',datetime(2026,10,6,0,15,tzinfo=timezone.utc))=='exact'
    assert pm.match_event(idx,'ATL','NO','2026-10-09T00:15:00Z') is None


def test_kalshi_candle_both_schemas():
    live={'end_period_ts':100,'price':{'close_dollars':'0.5200'},'yes_bid':{'close_dollars':'0.5100'},'yes_ask':{'close_dollars':'0.5300'},'volume_fp':'12.00'}
    hist={'end_period_ts':160,'price':{'close':None},'yes_bid':{'close':'0.0000'},'yes_ask':{'close':'0.5400'},'volume':'0'}
    assert pm.kalshi_candle(live)=={'ts':100,'bid':0.51,'ask':0.53,'last':0.52,'volume':12.0}
    assert pm.kalshi_candle(hist)=={'ts':160,'bid':0.0,'ask':0.54,'last':None,'volume':0.0}


def test_quote_at_ignores_post_kickoff_and_empty_sides():
    c=[{'ts':60,'bid':0.40,'ask':0.42,'last':0.41,'volume':5},{'ts':120,'bid':0.0,'ask':1.0,'last':None,'volume':0},
       {'ts':180,'bid':0.90,'ask':0.95,'last':0.93,'volume':9}]
    quote,trade=pm.quote_at(c,150)
    assert quote=={'ts':60,'bid':0.40,'ask':0.42} and trade=={'ts':60,'last':0.41}
    assert pm.side_price(quote,trade)==(0.41,0.40,0.42,60,'mid')


def test_side_price_falls_back_to_last_trade_on_wide_spread():
    prob,bid,ask,ts,how=pm.side_price({'ts':100,'bid':0.30,'ask':0.60},{'ts':90,'last':0.44})
    assert (prob,bid,ask,ts,how)==(0.44,0.30,0.60,90,'last')
    assert pm.side_price(None,None)==(None,None,None,None,None)
    assert pm.side_price({'ts':5,'bid':0.2,'ask':0.5},None)[4]=='wide_mid'


def test_last_point_and_norm():
    assert pm.last_point([{'t':10,'p':0.5},{'t':20,'p':0.6},{'t':31,'p':0.9}],30)=={'t':20,'p':0.6}
    assert pm.last_point([],30) is None
    row=pm.finish_row({'home_prob':0.52,'away_prob':0.50})
    assert abs(row['home_prob_norm']-0.52/1.02)<1e-12 and row['home_prob']+row['away_prob']>1  # raw values kept


def test_poly_events_unordered_pair_and_moneyline_outcomes():
    ev={'slug':'nfl-kc-bal-2024-09-05','markets':[
        {'slug':'nfl-kc-bal-2024-09-05-total-46pt5','sportsMarketType':'totals','outcomes':'["Over","Under"]','clobTokenIds':'["1","2"]'},
        {'slug':'nfl-kc-bal-2024-09-05','sportsMarketType':None,'outcomes':'["Chiefs", "Ravens"]','clobTokenIds':'["111","222"]','conditionId':'0xabc'}]}
    other={'slug':'nfl-ari-lar-2024-09-15','markets':[]}
    idx=pm.poly_events([ev,other,{'slug':'nfl-will-the-49ers-win'}])
    assert set(idx)=={(frozenset({'KC','BAL'}),date(2024,9,5)),(frozenset({'ARI','LA'}),date(2024,9,15))}
    # Ravens at Chiefs: home/away come from outcome names, not slug order
    assert pm.match_event(idx,'KC','BAL','2024-09-06T00:20:00Z') is ev
    picked=pm.poly_moneyline(ev)
    assert picked['tokens']=={'KC':'111','BAL':'222'} and picked['market']['conditionId']=='0xabc'
    assert pm.poly_moneyline({'slug':'y','markets':[{'slug':'y','outcomes':'["LAR", "Panthers"]','clobTokenIds':'["7","8"]'}]})['tokens']=={'LA':'7','CAR':'8'}
    assert pm.poly_moneyline({'slug':'x','markets':[{'slug':'x','outcomes':'["Yes","No"]','clobTokenIds':'["1","2"]'}]}) is None


def test_futures_event_for_conference_and_super_bowl():
    fut={'KXNFLNFCCHAMP-25':{'LA':{'ticker':'KXNFLNFCCHAMP-25-LA'},'SEA':{'ticker':'KXNFLNFCCHAMP-25-SEA'},'SF':{}},
         'KXSB-26':{'SEA':{'ticker':'KXSB-26-SEA'},'NE':{'ticker':'KXSB-26-NE'}}}
    s,e=pm.futures_event(fut,{'game_type':'CON','season':2025,'home_team':'SEA','away_team':'LA'})
    assert s=='KXNFLNFCCHAMP' and set(e['markets'])=={'SEA','LA'}
    s,e=pm.futures_event(fut,{'game_type':'SB','season':2025,'home_team':'NE','away_team':'SEA'})
    assert e['event_ticker']=='KXSB-26'
    assert pm.futures_event(fut,{'game_type':'REG','season':2025,'home_team':'SEA','away_team':'LA'}) is None


def test_http_cache_serves_offline(tmp_path):
    http=pm.Http(tmp_path,offline=True)
    assert http.get('https://example.test/x',{'a':1}) is None and http.calls==0
