"""Reproducible model, simulator and baseline benchmark report."""
import json
import sys
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app.demo import demo_players,demo_league
from app.projections import project_all
from app.simulation import simulate_drafts,season_outcomes,evaluate_candidates,mock_report
from app.draft import quick_recommendations
from app.domain import Pick
from app.backtest import backtest
from app.data import historical
from app.storage import DATA

def run(n=1000):
    league=demo_league();league.my_team=0
    players=demo_players();reports=project_all(players,league)
    started=time.monotonic()
    quick=quick_recommendations(players,reports,league,[])
    quick_seconds=time.monotonic()-started
    policies={}
    for policy in ('adp','espn','ecr','vorp','bpa','adaptive'):
        started=time.monotonic()
        rosters,_=simulate_drafts(players,reports,league,[],n=n,policy=policy)
        outcome,_=season_outcomes(rosters,players,reports,league)
        policies[policy]=outcome|{'seconds':time.monotonic()-started}
        print(policy,policies[policy],flush=True)
    result={'data':'SYNTHETIC DEMO, not evidence of real-world strategy advantage','simulations_per_policy':n,'quick_recommendation_seconds':quick_seconds,'policies':policies}
    started=time.monotonic()
    result['candidates']=evaluate_candidates(players,reports,league,[],[r['id'] for r in quick[:6]],n=n)
    result['candidate_seconds']=time.monotonic()-started
    result['historical_projection_backtest']=backtest(historical(),league)
    DATA.mkdir(exist_ok=True)
    (DATA/'validation.json').write_text(json.dumps(result,indent=2))
    print('Saved storage/validation.json',flush=True)
    return result
if __name__=='__main__':run(int(sys.argv[1]) if len(sys.argv)>1 else 1000)
