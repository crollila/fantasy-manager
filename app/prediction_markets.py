"""Historical NFL game-winner prediction-market prices at kickoff (Kalshi, Polymarket) for betting backtests.

Price = last observation at or before scheduled kickoff; staleness is kept in minutes_before_kickoff.
Prices are never invented: Kalshi gives minute candles with yes bid/ask; Polymarket's public history is a
displayed price (~midpoint) only, so its bid/ask stay None. Every HTTP response is cached as JSON under cache_dir.
"""
import hashlib
import json
import re
import time
from datetime import datetime,timezone,timedelta
from pathlib import Path
from zoneinfo import ZoneInfo
import httpx

KALSHI='https://api.elections.kalshi.com/trade-api/v2'
GAMMA='https://gamma-api.polymarket.com'
CLOB='https://clob.polymarket.com'
DATA='https://data-api.polymarket.com'
ET=ZoneInfo('America/New_York')
TEAMS='ARI ATL BAL BUF CAR CHI CIN CLE DAL DEN DET GB HOU IND JAX KC LA LAC LV MIA MIN NE NO NYG NYJ PHI PIT SEA SF TB TEN WAS'.split()
# Venue codes that differ from nflverse. Kalshi used LA (2025) then LAR (2026) for the Rams and JAC throughout;
# Polymarket used lar/las/wsh early in 2024 and nflverse codes afterwards.
KALSHI_CODES={'LAR':'LA','JAC':'JAX','WSH':'WAS','LVR':'LV','KAN':'KC','GNB':'GB','NWE':'NE','NOR':'NO','SFO':'SF','TAM':'TB'}
POLY_CODES={'lar':'LA','las':'LV','wsh':'WAS','jac':'JAX','lvr':'LV'}
NICKNAMES={'Cardinals':'ARI','Falcons':'ATL','Ravens':'BAL','Bills':'BUF','Panthers':'CAR','Bears':'CHI','Bengals':'CIN','Browns':'CLE','Cowboys':'DAL','Broncos':'DEN','Lions':'DET','Packers':'GB','Texans':'HOU','Colts':'IND','Jaguars':'JAX','Chiefs':'KC','Rams':'LA','Chargers':'LAC','Raiders':'LV','Dolphins':'MIA','Vikings':'MIN','Patriots':'NE','Saints':'NO','Giants':'NYG','Jets':'NYJ','Eagles':'PHI','Steelers':'PIT','Seahawks':'SEA','49ers':'SF','Niners':'SF','Buccaneers':'TB','Titans':'TEN','Commanders':'WAS'}
POLY_SERIES=('1','10187','12185')  # gamma series: 'nfl' (holds 2024 season), 'nfl-2025', 'nfl-2026'; /sports adds newer ones
KALSHI_FEE='Kalshi taker fee ceil(0.07*C*P*(1-P)) $, maker ceil(0.0175*C*P*(1-P)) (series fee_type quadratic_with_maker_fees, multiplier 1)'
WINDOW_MIN=180;PREFERRED_MIN=30


# ---------- pure helpers (unit tested) ----------
def utc(value):
    if isinstance(value,datetime):return (value if value.tzinfo else value.replace(tzinfo=timezone.utc)).astimezone(timezone.utc)
    if hasattr(value,'to_pydatetime'):return utc(value.to_pydatetime())
    return utc(datetime.fromisoformat(str(value).replace('Z','+00:00')))


def et_date(kickoff):return utc(kickoff).astimezone(ET).date()


def num(value):
    try:return None if value is None or value=='' else float(value)
    except (TypeError,ValueError):return None


def kalshi_team(code):
    code=(code or '').upper();code=KALSHI_CODES.get(code,code)
    return code if code in TEAMS else None


def poly_team(code):
    code=(code or '').lower();code=POLY_CODES.get(code,code).upper()
    return code if code in TEAMS else None


MONTHS={m:i+1 for i,m in enumerate('JAN FEB MAR APR MAY JUN JUL AUG SEP OCT NOV DEC'.split())}


def kalshi_event_date(event_ticker):
    """KXNFLGAME-26OCT05ATLNO -> date(2026,10,5) (US Eastern game date)."""
    m=re.match(r'^[A-Z0-9]+-(\d\d)([A-Z]{3})(\d\d)',event_ticker or '')
    if not m or m.group(2) not in MONTHS:return None
    from datetime import date
    return date(2000+int(m.group(1)),MONTHS[m.group(2)],int(m.group(3)))


def kalshi_events(markets):
    """Group KXNFLGAME markets into {(frozenset(teams),date):event}; each event maps nflverse team -> market."""
    groups={}
    for m in markets:groups.setdefault(m['event_ticker'],[]).append(m)
    out={}
    for ticker,ms in groups.items():
        day=kalshi_event_date(ticker);teams={}
        for m in ms:
            team=kalshi_team(m['ticker'].rsplit('-',1)[-1])
            if team:teams[team]=m
        if day is None or len(teams)!=2:continue
        out[(frozenset(teams),day)]={'event_ticker':ticker,'date':day,'markets':teams}
    return out


def match_events(index,home,away,kickoff,tolerance_days=1):
    """Events for an unordered team pair whose listed date is within tolerance of the US Eastern kickoff date,
    nearest date first (Kalshi sometimes lists a duplicate, untraded event on an adjacent placeholder date)."""
    day=et_date(kickoff);pair=frozenset((home,away))
    return [index[(pair,day+timedelta(days=d))] for d in sorted(range(-tolerance_days,tolerance_days+1),key=abs) if (pair,day+timedelta(days=d)) in index]


def match_event(index,home,away,kickoff,tolerance_days=1):
    hits=match_events(index,home,away,kickoff,tolerance_days);return hits[0] if hits else None


def futures_event(futures,game):
    """Championship futures that equal the game-winner market at kickoff: conference champion for CON games
    (event suffix = season), Super Bowl winner for the SB (suffix = season+1). Both teams must be listed."""
    kind=game.get('game_type');season=int(game['season'])
    series={'CON':('KXNFLAFCCHAMP','KXNFLNFCCHAMP'),'SB':('KXSB',)}.get(kind,())
    yy=f'{(season+(kind=="SB"))%100:02d}'
    for s in series:
        teams=futures.get(f'{s}-{yy}',{})
        if game['home_team'] in teams and game['away_team'] in teams:
            return s,{'event_ticker':f'{s}-{yy}','markets':{t:teams[t] for t in (game['home_team'],game['away_team'])}}
    return None


def kalshi_candle(candle):
    """Normalise a candle from either the live (_dollars/_fp strings) or historical (plain strings) schema."""
    def group(name):
        g=candle.get(name) or {};return {k.removesuffix('_dollars'):num(v) for k,v in g.items()}
    price,bid,ask=group('price'),group('yes_bid'),group('yes_ask')
    return {'ts':int(candle['end_period_ts']),'bid':bid.get('close'),'ask':ask.get('close'),'last':price.get('close'),
            'volume':num(candle.get('volume_fp',candle.get('volume')))}


def quote_at(candles,kickoff_ts):
    """Latest two-sided quote and latest trade at or before kickoff. A 0 bid / 1 ask means an empty side."""
    rows=sorted((c for c in candles if c['ts']<=kickoff_ts),key=lambda c:c['ts']);quote=trade=None
    for c in rows:
        bid=c['bid'] if c['bid'] is not None and c['bid']>0 else None;ask=c['ask'] if c['ask'] is not None and c['ask']<1 else None
        if bid is not None or ask is not None:quote={'ts':c['ts'],'bid':bid,'ask':ask}
        if c['last'] is not None and (c['volume'] is None or c['volume']>0):trade={'ts':c['ts'],'last':c['last']}
    return quote,trade


def side_price(quote,trade,max_spread=0.10):
    """(prob, bid, ask, ts, how). Mid of a tight two-sided quote, else last trade, else a wide mid."""
    bid=quote and quote['bid'];ask=quote and quote['ask']
    if bid is not None and ask is not None and ask-bid<=max_spread+1e-9:return round((bid+ask)/2,6),bid,ask,quote['ts'],'mid'
    if trade:return trade['last'],bid,ask,trade['ts'],'last'
    if bid is not None and ask is not None:return round((bid+ask)/2,6),bid,ask,quote['ts'],'wide_mid'
    return None,bid,ask,None,None


def last_point(history,kickoff_ts):
    """Polymarket prices-history [{t,p}] -> latest point at or before kickoff."""
    pts=[h for h in history or [] if h.get('t') is not None and h['t']<=kickoff_ts and h.get('p') is not None]
    return max(pts,key=lambda h:h['t']) if pts else None


POLY_SLUG=re.compile(r'^nfl-([a-z]+)-([a-z]+)-(\d{4})-(\d\d)-(\d\d)$')


def poly_events(events):
    """Index gamma game events {(frozenset(teams),date):event}. Slug team order flips between seasons (home-away in
    2024, away-home later) so only the unordered pair and date are trusted; sides come from outcome names."""
    from datetime import date
    out={}
    for e in events:
        m=POLY_SLUG.match(e.get('slug') or '')
        if not m:continue
        a,b=poly_team(m.group(1)),poly_team(m.group(2))
        if not a or not b or a==b:continue
        out[(frozenset((a,b)),date(int(m.group(3)),int(m.group(4)),int(m.group(5))))]=e
    return out


def poly_moneyline(event):
    """Pick the game-winner market and map its outcome tokens to nflverse teams: {'market':m,'tokens':{team:token}}."""
    markets=event.get('markets') or []
    ranked=sorted(markets,key=lambda m:(m.get('slug')!=event.get('slug'),m.get('sportsMarketType') not in (None,'moneyline')))
    for m in ranked:
        if m.get('sportsMarketType') not in (None,'moneyline'):continue
        outcomes=_jsonish(m.get('outcomes'));tokens=_jsonish(m.get('clobTokenIds'))
        if len(outcomes)!=2 or len(tokens)!=2:continue
        teams=[NICKNAMES.get(str(o).strip().split()[-1]) or poly_team(str(o).strip()) for o in outcomes]  # 'Rams' or occasionally 'LAR'
        if None in teams or teams[0]==teams[1]:continue
        return {'market':m,'tokens':dict(zip(teams,tokens))}
    return None


def _jsonish(value):
    if isinstance(value,list):return value
    try:return json.loads(value or '[]')
    except (TypeError,ValueError):return []


def finish_row(row):
    h,a=row.get('home_prob'),row.get('away_prob')
    row['home_prob_norm']=h/(h+a) if h is not None and a is not None and h+a>0 else None
    return row


# ---------- HTTP with permanent JSON cache ----------
class Http:
    def __init__(self,cache_dir,pause=0.12,offline=False):
        self.root=Path(cache_dir);self.pause=pause;self.offline=offline;self.session=httpx.Client(follow_redirects=True);self.calls=0

    def get(self,url,params=None,ttl=None):
        """GET JSON. ttl=None caches forever (completed games never change); ttl seconds for listings that grow.
        404 is cached as None. Retries 429/5xx with backoff."""
        params=dict(sorted((params or {}).items()));key=hashlib.sha1((url+'?'+json.dumps(params)).encode()).hexdigest()
        host=re.sub(r'\W+','_',url.split('/')[2]);path=self.root/'http'/host/f'{key}.json'
        if path.exists():
            blob=json.loads(path.read_text(encoding='utf-8'))
            if ttl is None or self.offline or time.time()-blob['fetched']<ttl:return blob['body']
        if self.offline:return None
        for attempt in range(6):
            time.sleep(self.pause);self.calls+=1
            try:r=self.session.get(url,params=params,timeout=40)
            except httpx.HTTPError:
                time.sleep(2**attempt);continue
            if r.status_code==429 or r.status_code>=500:
                time.sleep(float(r.headers.get('Retry-After') or 0) or min(60,2**attempt));continue
            body=r.json() if r.status_code==200 else None
            if r.status_code not in (200,404):raise RuntimeError(f'{url} HTTP {r.status_code}: {r.text[:200]}')
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(json.dumps({'url':url,'params':params,'status':r.status_code,'fetched':time.time(),'body':body}),encoding='utf-8')
            return body
        raise RuntimeError(f'{url} failed after retries')


# ---------- Kalshi ----------
def _kalshi_list(http,endpoint,series,ttl):
    out=[];cursor=None
    while True:
        params={'series_ticker':series,'limit':1000}|({'cursor':cursor} if cursor else {})
        body=http.get(KALSHI+endpoint,params,ttl=ttl) or {};ms=body.get('markets') or []
        out+=[m|{'_endpoint':endpoint} for m in ms];cursor=body.get('cursor')
        if not cursor or not ms:return out


FUTURES=('KXSB','KXNFLAFCCHAMP','KXNFLNFCCHAMP')


def kalshi_index(http,ttl=6*3600):
    """Game-winner events from the live and historical (settled before /historical/cutoff) market listings, plus
    conference/Super Bowl championship futures, used for CON/SB games when no traded KXNFLGAME event exists."""
    games=_kalshi_list(http,'/historical/markets','KXNFLGAME',ttl)+_kalshi_list(http,'/markets','KXNFLGAME',ttl);futures={}
    for series in FUTURES:
        for m in _kalshi_list(http,'/historical/markets',series,ttl)+_kalshi_list(http,'/markets',series,ttl):
            team=kalshi_team(m['ticker'].rsplit('-',1)[-1])
            if team:futures.setdefault(m['event_ticker'],{})[team]=m
    return kalshi_events(games),futures


def _kalshi_candles(http,market,series,start,end,interval):
    live=f'{KALSHI}/series/{series}/markets/{market["ticker"]}/candlesticks';hist=f'{KALSHI}/historical/markets/{market["ticker"]}/candlesticks'
    params={'start_ts':start,'end_ts':end,'period_interval':interval}
    for url in ([hist,live] if market.get('_endpoint')=='/historical/markets' else [live,hist]):
        body=http.get(url,params)
        if body is not None:return [kalshi_candle(c) for c in body.get('candlesticks') or []]
    return []


def kalshi_price(http,game,event,series='KXNFLGAME',note=''):
    k=int(utc(game['kickoff_utc']).timestamp());home,away=game['home_team'],game['away_team'];sides={}
    for team in (home,away):
        market=event['markets'][team];quote=trade=None
        for interval,span in ((1,WINDOW_MIN*60),(60,7*86400)):
            quote,trade=quote_at(_kalshi_candles(http,market,series,k-span,k,interval),k)
            if quote or trade:break
        sides[team]=side_price(quote,trade)
    if sides[home][0] is None or sides[away][0] is None:return None
    times=[sides[t][3] for t in (home,away)];ts=min(times);hows={sides[t][4] for t in (home,away)}
    volume=sum(num(event['markets'][t].get('volume_fp',event['markets'][t].get('volume'))) or 0 for t in (home,away))
    return finish_row({'game_id':game['game_id'],'source':'kalshi','market_id':event['event_ticker'],
        'home_contract':event['markets'][home]['ticker'],'away_contract':event['markets'][away]['ticker'],
        'home_prob':sides[home][0],'away_prob':sides[away][0],'home_bid':sides[home][1],'home_ask':sides[home][2],'away_bid':sides[away][1],'away_ask':sides[away][2],
        'price_time':datetime.fromtimestamp(ts,timezone.utc).isoformat(),'minutes_before_kickoff':round((k-ts)/60,1),
        'volume':volume,'volume_unit':'contracts (lifetime, both sides)','url':f'https://kalshi.com/markets/{series.lower()}/{event["event_ticker"].lower()}',
        'method':f'kalshi candlesticks: yes bid/ask close of last candle ending <= kickoff; prob={"/".join(sorted(hows))} (mid if spread<=0.10 else last trade){note}',
        'fee_note':KALSHI_FEE})


def kalshi_game(http,game,index):
    events,futures=index
    for event in match_events(events,game['home_team'],game['away_team'],game['kickoff_utc']):
        row=kalshi_price(http,game,event)
        if row:return row
    hit=futures_event(futures,game)
    return hit and kalshi_price(http,game,hit[1],hit[0],f'; {hit[0]} championship futures (= game winner once only these two teams remain)')


# ---------- Polymarket ----------
def poly_index(http,ttl=6*3600):
    series=list(POLY_SERIES)
    for s in http.get(GAMMA+'/sports',ttl=ttl) or []:
        if s.get('sport')=='nfl' and s.get('series') and str(s['series']) not in series:series.append(str(s['series']))
    events=[]
    for sid in series:
        offset=0
        while True:
            page=http.get(GAMMA+'/events',{'series_id':sid,'limit':100,'offset':offset},ttl=ttl) or []
            events+=page;offset+=len(page)
            if len(page)<100:break
    return poly_events(events)


def _poly_lookup(http,game):
    """Direct slug fallback when an event is missing from the series listings."""
    day=et_date(game['kickoff_utc'])
    for d in (day,day-timedelta(days=1),day+timedelta(days=1)):
        for a,b in ((game['away_team'],game['home_team']),(game['home_team'],game['away_team'])):
            hit=http.get(GAMMA+'/events',{'slug':f'nfl-{a.lower()}-{b.lower()}-{d.isoformat()}'})
            if hit:return hit[0]
    return None


def _poly_trades(http,condition,tokens,k,max_pages=20):
    """Fallback: last trade per token at/before kickoff from data-api (newest first)."""
    last={}
    for page in range(max_pages):
        rows=http.get(DATA+'/trades',{'market':condition,'limit':500,'offset':page*500}) or []
        for t in rows:
            if t.get('timestamp',k+1)<=k and t.get('asset') in tokens and t['asset'] not in last:last[t['asset']]={'t':t['timestamp'],'p':num(t.get('price'))}
        if len(last)==len(tokens) or len(rows)<500 or (rows and rows[-1].get('timestamp',0)<k-86400*14):break
    return last


def poly_game(http,game,index):
    for event in match_events(index,game['home_team'],game['away_team'],game['kickoff_utc'])+[None]:
        event=event or _poly_lookup(http,game);picked=event and poly_moneyline(event)
        if picked and set(picked['tokens'])=={game['home_team'],game['away_team']}:break
    else:return None
    m=picked['market'];k=int(utc(game['kickoff_utc']).timestamp());pts={};how='clob prices-history fidelity=1'
    for team,token in picked['tokens'].items():
        for start,fidelity in ((k-WINDOW_MIN*60,1),(k-7*86400,60)):
            pt=last_point((http.get(CLOB+'/prices-history',{'market':token,'startTs':start,'endTs':k,'fidelity':fidelity}) or {}).get('history'),k)
            if pt:pts[team]=pt;break
    if len(pts)<2:
        trades=_poly_trades(http,m.get('conditionId'),set(picked['tokens'].values()),k)
        pts={team:trades[tok] for team,tok in picked['tokens'].items() if tok in trades and trades[tok]['p'] is not None}
        how='data-api last trade (prices-history empty)'
        if len(pts)<2:return None
    home,away=game['home_team'],game['away_team'];ts=min(pts[home]['t'],pts[away]['t'])
    fee=m.get('feeSchedule') or {}
    fee_note=f"feesEnabled={m.get('feesEnabled')} feeType={m.get('feeType')} rate={fee.get('rate')} takerOnly={fee.get('takerOnly')}; Polymarket taker fee = C*rate*p*(1-p)"
    return finish_row({'game_id':game['game_id'],'source':'polymarket','market_id':m.get('conditionId'),
        'home_contract':picked['tokens'][home],'away_contract':picked['tokens'][away],
        'home_prob':float(pts[home]['p']),'away_prob':float(pts[away]['p']),'home_bid':None,'home_ask':None,'away_bid':None,'away_ask':None,
        'price_time':datetime.fromtimestamp(ts,timezone.utc).isoformat(),'minutes_before_kickoff':round((k-ts)/60,1),
        'volume':num(m.get('volumeNum',m.get('volume'))),'volume_unit':'USDC (lifetime)','url':f'https://polymarket.com/event/{event.get("slug")}',
        'method':f'polymarket {how}: displayed price (book midpoint, or last trade when spread>0.10); no historical bid/ask',
        'fee_note':fee_note})


# ---------- public API ----------
SOURCES={'kalshi':(kalshi_index,kalshi_game),'polymarket':(poly_index,poly_game)}


def kickoff_prices(games,cache_dir,sources=('kalshi','polymarket'),offline=False,log=None):
    """One row per (game, source) with a market; games without a market are skipped. Results for games that
    kicked off more than 2 days ago are cached per game (misses too) so re-runs are offline-fast."""
    http=Http(cache_dir,offline=offline);rows=[];now=datetime.now(timezone.utc)
    for source in sources:
        build,price=SOURCES[source];index=None
        for i,game in enumerate(games):
            game=dict(game);kickoff=utc(game['kickoff_utc']);game['kickoff_utc']=kickoff
            final=kickoff<now-timedelta(days=2);path=Path(cache_dir)/'games'/source/f'{game["game_id"]}.json'
            if final and path.exists():
                row=json.loads(path.read_text(encoding='utf-8'))['row']
            else:
                if kickoff>now:continue
                if index is None:index=build(http)
                row=price(http,game,index)
                if final:
                    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps({'row':row}),encoding='utf-8')
            if row:rows.append(row)
            if log and i%50==0:log(f'{source} {i}/{len(games)} rows={len(rows)} http_calls={http.calls}')
    return rows


def load_games(games_parquet,seasons):
    import pandas as pd
    g=pd.read_parquet(games_parquet)
    g=g[g.season.isin(list(seasons))&g.home_score.notna()&g.kickoff_utc.notna()]
    return [{'game_id':r.game_id,'season':int(r.season),'week':int(r.week),'game_type':r.game_type,'home_team':r.home_team,'away_team':r.away_team,
             'kickoff_utc':r.kickoff_utc.isoformat()} for r in g.itertuples()]


COLUMNS=['game_id','season','week','game_type','home_team','away_team','kickoff_utc','source','market_id','home_contract','away_contract',
         'home_prob','away_prob','home_prob_norm','home_bid','home_ask','away_bid','away_ask','price_time','minutes_before_kickoff',
         'volume','volume_unit','url','method','fee_note']


def collect(seasons,games_parquet,cache_dir,out_path,sources=('kalshi','polymarket'),log=print):
    """Collect completed games of `seasons` and write out_path (.parquet) plus a sibling .csv. Returns the DataFrame."""
    import pandas as pd
    games=load_games(games_parquet,seasons);meta={g['game_id']:g for g in games}
    rows=[{**{k:meta[r['game_id']][k] for k in ('season','week','game_type','home_team','away_team','kickoff_utc')},**r} for r in kickoff_prices(games,cache_dir,sources,log=log)]
    df=pd.DataFrame(rows,columns=COLUMNS);out=Path(out_path);out.parent.mkdir(parents=True,exist_ok=True)
    df.to_parquet(out.with_suffix('.parquet'),index=False);df.to_csv(out.with_suffix('.csv'),index=False)
    return df


if __name__=='__main__':
    import argparse,os
    p=argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--seasons',type=int,nargs='+',default=[2024,2025,2026])
    p.add_argument('--games',default=os.path.join(os.environ.get('LOCALAPPDATA',''),'Fantasy Manager','data','nfl','normalized','games.parquet'))
    p.add_argument('--cache',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();collect(a.seasons,a.games,a.cache,a.out)
