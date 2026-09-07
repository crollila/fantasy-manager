"""Optional licensed market feed. Credentials never enter saved URLs or diagnostics."""
import os
from datetime import datetime,timezone
from statistics import median
import httpx

URL='https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds'


def fetch_markets(games):
    key=os.environ.get('THE_ODDS_API_KEY')
    if not key:return {},{'name':'Live multi-book odds','status':'not configured','url':'https://the-odds-api.com','note':'Set THE_ODDS_API_KEY locally to enable. Public schedule lines remain available.'}
    try:
        response=httpx.get(URL,params={'apiKey':key,'regions':'us','markets':'h2h,spreads,totals','oddsFormat':'american'},timeout=25)
        if response.status_code!=200:return {},{'name':'Live multi-book odds','status':'unavailable','error':f'Provider returned HTTP {response.status_code}','url':'https://the-odds-api.com'}
        raw=response.json();now=datetime.now(timezone.utc);result={}
        for event in raw:
            candidates=[g for g in games if g.get('home_name')==event.get('home_team') and g.get('away_name')==event.get('away_team') and abs((datetime.fromisoformat(g['kickoff'].replace('Z','+00:00'))-datetime.fromisoformat(event['commence_time'].replace('Z','+00:00'))).total_seconds())<3600]
            if len(candidates)!=1:continue
            spreads=[];totals=[];money=[]
            for book in event.get('bookmakers',[]):
                for market in book.get('markets',[]):
                    stamp=datetime.fromisoformat((market.get('last_update') or book.get('last_update')).replace('Z','+00:00'))
                    if not 0<=(now-stamp).total_seconds()<6*3600:continue
                    outcomes={o['name']:o for o in market.get('outcomes',[])}
                    home=outcomes.get(event['home_team']);away=outcomes.get(event['away_team'])
                    metadata={'book':book['title'],'quoted_at':stamp.isoformat()}
                    if market['key']=='spreads' and home and away:spreads.append({'spread_line':-float(home['point']),'home_spread_odds':home['price'],'away_spread_odds':away['price'],**metadata})
                    if market['key']=='h2h' and home and away:money.append({'home_moneyline':home['price'],'away_moneyline':away['price'],**metadata})
                    if market['key']=='totals' and 'Over' in outcomes and 'Under' in outcomes:totals.append({'total_line':float(outcomes['Over']['point']),'over_odds':outcomes['Over']['price'],'under_odds':outcomes['Under']['price'],**metadata})
            combined={};quotes={}
            for category,rows,field in [('spread',spreads,'spread_line'),('total',totals,'total_line'),('moneyline',money,'home_moneyline')]:
                if not rows:continue
                middle=median(r[field] for r in rows);selected=min(rows,key=lambda r:abs(r[field]-middle))
                combined.update({k:v for k,v in selected.items() if k not in ('book','quoted_at')});quotes[category]=selected|{'books_compared':len(rows)}
            if combined:result[candidates[0]['game_id']]={'fields':combined,'quotes':quotes,'source':'The Odds API; quote nearest multi-book median','fetched_at':now.isoformat()}
        return result,{'name':'Live multi-book odds','status':'ok','fetched_at':now.isoformat(),'url':'https://the-odds-api.com','games_matched':len(result)}
    except Exception:
        # HTTP client exception strings may contain the API key in the request URL.
        return {},{'name':'Live multi-book odds','status':'unavailable','error':'Market request or schema validation failed; credentials redacted','url':'https://the-odds-api.com'}
