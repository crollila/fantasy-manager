import argparse
import json
from app.storage import Store,DATA

def main():
    parser=argparse.ArgumentParser(description='Fantasy Manager local commands')
    parser.add_argument('command',choices=['refresh','demo','backtest','calibration','status'])
    parser.add_argument('--season',type=int,default=2026)
    args=parser.parse_args()
    store=Store()
    if args.command=='refresh':
        from app.api import refresh
        print(json.dumps(refresh(args.season),indent=2))
    elif args.command=='demo':
        from app.demo import demo_league,demo_players
        league=demo_league();league.season=2099
        store.save_players(2099,demo_players());store.save_league(league)
        print('Synthetic demo initialized in isolated season 2099')
    elif args.command=='backtest':
        from app.backtest import backtest
        from app.data import historical
        from app.domain import League
        result=backtest(historical(),League(id='validation',name='Validation',season=args.season))
        (DATA/'backtest.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
    elif args.command=='calibration':
        from app.calibration import audit
        data=json.loads((DATA/f'model_{args.season}.json').read_text())
        result=audit(data['residuals']);(DATA/'calibration.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
    else:
        from app.data import Cache
        print(json.dumps({'leagues':[l.model_dump() for l in store.leagues()],'catalog':store.catalog_meta(args.season),'sources':Cache().status()},indent=2))

if __name__=='__main__':main()
