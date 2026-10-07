"""Kickoff prediction-market prices (Kalshi, Polymarket) and the betting backtest against them.

The backtest takes the model at its word: when our win probability for a team exceeds the
price of that team's "wins" contract by at least the edge threshold, it buys that contract
at kickoff. A contract pays $1 if the team wins. It never bets both sides of a game.

Two model sources, kept separate in every result:
- live: the last forecast the app saved before kickoff (2026 onward), the real record;
- walk-forward: the engine's historical backtest, where each season was predicted by a
  model trained only on earlier seasons (bundled for 2024-2025, before the app was live).
"""
import json
import math
import re
from datetime import datetime,timezone
from app.storage import DATA,ROOT

MARKET_DIR=DATA/'cache'/'markets'
HISTORY_FILE=ROOT/'app'/'nfl'/'reference'/'data'/'market_history.json'
SOURCES=('kalshi','polymarket')
THRESHOLDS=(0.0,0.025,0.05,0.075,0.10)
BANKROLL=1000.0
FLAT_STAKE=100.0
KELLY_FRACTION=0.25


def kickoff_lookup(path=None):
    """{game_id: {source: price row}} from the cached kickoff prices; empty when none collected."""
    path=path or MARKET_DIR/'kickoff_prices.json'
    try:rows=json.loads(path.read_text())
    except (OSError,ValueError):return {}
    out={}
    for r in rows:out.setdefault(r['game_id'],{})[r['source']]=r
    return out


def fee(source,price,row=None):
    """Per-contract taker fee in dollars. Kalshi: 0.07 x P x (1-P) (rounded up to the cent per
    order; rounding ignored). Polymarket: rate x P x (1-P), with the rate recorded per market
    (no fees on 2024-2025 game markets; 0.03-0.05 on some 2026 markets)."""
    if source=='kalshi':return 0.07*price*(1-price)
    found=re.search(r'rate=([0-9.]+)',str((row or {}).get('fee_note') or ''))
    return float(found.group(1))*price*(1-price) if found and 'feesEnabled=True' in str(row.get('fee_note')) else 0.0


def buy_price(row,side):
    """What a contract on `side` cost at kickoff: the ask when the book was recorded, otherwise
    the mid/last price (flagged, since a real order would usually pay a little more)."""
    ask=row.get(f'{side}_ask')
    if ask is not None and 0<ask<1:return float(ask),'ask'
    prob=row.get(f'{side}_prob')
    if prob is None:
        norm=row.get('home_prob_norm')
        if norm is None:return None,None
        prob=norm if side=='home' else 1-norm
    return float(prob),'mid'


def live_games(store):
    """Final pregame forecast of every finished game in the app's own ledger."""
    from app.forecast_history import _ledger,_bodies
    ledger=_ledger(store)
    with store.connect() as c:graded={r[0]:json.loads(r[1]) for r in c.execute('SELECT game_id,body FROM game_results')}
    bodies=_bodies(store,[v['last'] for g,v in ledger.items() if g in graded])
    out=[]
    for gid,v in ledger.items():
        if gid not in graded:continue
        f=bodies[v['last']];r=graded[gid];hp,ap=f['home_win_probability'],f['away_win_probability']
        out.append({'game_id':gid,'season':r['season'],'week':r['week'],'kickoff':f['kickoff'],'home_team':f['home_team'],'away_team':f['away_team'],
                    'model_home':hp/(hp+ap),'home_score':int(r['home_score']),'away_score':int(r['away_score']),'origin':'live'})
    return out


def history_games(path=None):
    """Bundled walk-forward predictions with their kickoff market prices (2024-2025)."""
    try:data=json.loads((path or HISTORY_FILE).read_text())
    except (OSError,ValueError):return [],{}
    games=[g|{'origin':'walk-forward'} for g in data.get('games',[])]
    prices={}
    for r in data.get('prices',[]):prices.setdefault(r['game_id'],{})[r['source']]=r
    return games,prices


def simulate(games,prices,source,threshold,staking):
    """Chronological bets on one market. Returns bets, summary and the equity curve."""
    bankroll=BANKROLL;bets=[];peak=bankroll;drawdown=0.
    for g in sorted(games,key=lambda g:(g['kickoff'] or '',g['game_id'])):
        row=(prices.get(g['game_id']) or {}).get(source)
        if not row:continue
        best=None
        for side in ('home','away'):
            price,basis=buy_price(row,side)
            if price is None or not 0.01<=price<=0.99:continue
            p=g['model_home'] if side=='home' else 1-g['model_home']
            cost=price+fee(source,price,row);edge=p-cost
            if edge>=threshold and (best is None or edge>best['edge']):best={'side':side,'price':price,'cost':cost,'edge':edge,'p':p,'basis':basis}
        if not best:continue
        if staking=='kelly':
            b=(1-best['cost'])/best['cost'];kelly=max(0.,(best['p']*b-(1-best['p']))/b)
            stake=min(bankroll*kelly*KELLY_FRACTION,bankroll*0.05)
        else:stake=FLAT_STAKE
        if stake<=0:continue
        contracts=stake/best['cost']
        if g['home_score']==g['away_score']:payout=contracts*0.5
        else:payout=contracts*(1. if (g['home_score']>g['away_score'])==(best['side']=='home') else 0.)
        profit=payout-stake;bankroll+=profit
        peak=max(peak,bankroll);drawdown=max(drawdown,peak-bankroll)
        team=g['home_team'] if best['side']=='home' else g['away_team']
        bets.append({'game_id':g['game_id'],'season':g['season'],'week':g['week'],'origin':g['origin'],'matchup':f"{g['away_team']} at {g['home_team']}",'team':team,
                     'model_probability':round(best['p'],4),'price':round(best['price'],4),'price_basis':best['basis'],'edge':round(best['edge'],4),'stake':round(stake,2),
                     'profit':round(profit,2),'won':payout>stake,'final':f"{g['away_score']}-{g['home_score']}",'bankroll':round(bankroll,2)})
    staked=sum(b['stake'] for b in bets);profit=sum(b['profit'] for b in bets)
    return {'bets':len(bets),'wins':sum(b['won'] for b in bets),'staked':round(staked,2),'profit':round(profit,2),'roi':profit/staked if staked else None,
            'ending_bankroll':round(bankroll,2) if staking=='kelly' else None,'max_drawdown':round(drawdown,2),'average_edge':sum(b['edge'] for b in bets)/len(bets) if bets else None,
            'mid_priced':sum(b['price_basis']=='mid' for b in bets),'history':bets}


def _score(pairs):
    pairs=[(min(max(p,1e-4),1-1e-4),y) for p,y in pairs]
    if not pairs:return None
    return {'games':len(pairs),'accuracy':sum((p>.5)==(y==1) for p,y in pairs)/len(pairs),'brier':sum((p-y)**2 for p,y in pairs)/len(pairs),
            'log_loss':-sum(y*math.log(p)+(1-y)*math.log(1-p) for p,y in pairs)/len(pairs)}


def head_to_head(games,prices,source):
    """Model versus market probabilities on the same decided games, overall and per week."""
    def pairs(rows,key):return [(r[key],1 if r['home_score']>r['away_score'] else 0) for r in rows]
    rows=[]
    for g in games:
        row=(prices.get(g['game_id']) or {}).get(source)
        if row and row.get('home_prob_norm') is not None and g['home_score']!=g['away_score']:rows.append(g|{'market_home':float(row['home_prob_norm'])})
    weeks={}
    for r in rows:weeks.setdefault((r['season'],r['week']),[]).append(r)
    by_week=[]
    for (season,week),part in sorted(weeks.items()):
        m,k=_score(pairs(part,'model_home')),_score(pairs(part,'market_home'))
        by_week.append({'season':season,'week':week,'origin':part[0]['origin'],'games':len(part),'model':m,'market':k,'model_won':m['brier']<k['brier']})
    return {'model':_score(pairs(rows,'model_home')),'market':_score(pairs(rows,'market_home')),'by_week':by_week,
            'weeks_model_won':sum(w['model_won'] for w in by_week),'weeks':len(by_week)}


def backtest(store,threshold=0.05,staking='flat',origin='all'):
    live=live_games(store);history,history_prices=history_games()
    prices=history_prices|kickoff_lookup()
    games={'live':live,'walk-forward':history,'all':history+live}[origin]
    out={'generated_at':datetime.now(timezone.utc).isoformat(),'threshold':threshold,'staking':staking,'origin':origin,'thresholds':list(THRESHOLDS),
         'settings':{'flat_stake':FLAT_STAKE,'bankroll':BANKROLL,'kelly_fraction':KELLY_FRACTION,'kelly_cap':0.05},
         'coverage':{'live_games':len(live),'walk_forward_games':len(history)},'sources':{}}
    for source in SOURCES:
        sim=simulate(games,prices,source,threshold,staking)
        seasons={}
        for b in sim['history']:
            s=seasons.setdefault(f"{b['season']} {b['origin']}",{'season':b['season'],'origin':b['origin'],'bets':0,'wins':0,'staked':0.,'profit':0.})
            s['bets']+=1;s['wins']+=b['won'];s['staked']+=b['stake'];s['profit']+=b['profit']
        for s in seasons.values():s['roi']=s['profit']/s['staked'] if s['staked'] else None
        sweep=[{'threshold':t}|{k:v for k,v in simulate(games,prices,source,t,staking).items() if k!='history'} for t in THRESHOLDS]
        out['sources'][source]=sim|{'by_season':sorted(seasons.values(),key=lambda s:(s['season'],s['origin'])),'threshold_sweep':sweep,
                                    'priced_games':sum(1 for g in games if (prices.get(g['game_id']) or {}).get(source)),'head_to_head':head_to_head(games,prices,source)}
    out['notes']=['A contract on a team costs its price in dollars and pays $1 if that team wins (ties pay $0.50).',
                  'Bets are placed at the kickoff price: the ask when the order book was recorded, otherwise the mid/last trade price (counted as "mid priced").',
                  'Taker fees are deducted: Kalshi 0.07 x P x (1-P) per contract; Polymarket rate x P x (1-P) using the rate recorded for each market (zero for 2024-2025 game markets, 0.03-0.05 on some 2026 markets).',
                  'Walk-forward rows come from the engine backtest: each season predicted by a model trained only on earlier seasons. Live rows are the forecasts the app actually saved before kickoff.',
                  'Hypothetical: no orders are placed, liquidity and slippage beyond the quoted price are ignored, and past results do not guarantee future returns.']
    return out


def refresh_live_prices(store,cache_dir=None):
    """Collect kickoff prices for finished live games that have none yet. Never raises."""
    cache_dir=cache_dir or MARKET_DIR
    try:
        from app.prediction_markets import kickoff_prices
        existing=kickoff_lookup(cache_dir/'kickoff_prices.json')
        live=live_games(store)
        missing=[g for g in live if set(SOURCES)-set(existing.get(g['game_id'],{}))]
        if not missing:return {'status':'up to date','games':len(existing)}
        from app.nfl.reference.teams import DISPLAY
        reverse={v:k for k,v in DISPLAY.items()}
        rows=kickoff_prices([{'game_id':g['game_id'],'season':g['season'],'week':g['week'],'home_team':reverse.get(g['home_team'],g['home_team']),'away_team':reverse.get(g['away_team'],g['away_team']),'kickoff_utc':g['kickoff']} for g in missing],cache_dir)
        merged=[r for rows_ in existing.values() for r in rows_.values()]
        known={(r['game_id'],r['source']) for r in merged}
        merged+=[r for r in rows if (r['game_id'],r['source']) not in known]
        cache_dir.mkdir(parents=True,exist_ok=True)
        (cache_dir/'kickoff_prices.json').write_text(json.dumps(merged,default=str))
        return {'status':'ok','added':len(merged)-sum(len(v) for v in existing.values()),'games':len({r['game_id'] for r in merged})}
    except Exception as exc:  # noqa: BLE001 - market data must never block forecasting
        return {'status':'unavailable','error':str(exc)}
