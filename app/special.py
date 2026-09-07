"""Conservative K/DST component projections and team opportunity reconciliation."""
from pathlib import Path
import numpy as np
import pandas as pd
from app.domain import Player

K_STATS=['fg_made_0_19','fg_made_20_29','fg_made_30_39','fg_made_40_49','fg_made_50_59','fg_made_60_','fg_missed','pat_made','pat_missed']
D_STATS=['def_sacks','def_interceptions','fumble_recovery_opp','def_tds','def_safeties','def_punt_blocks','def_pat_blocks','def_fg_blocks','special_teams_tds']

def special_players(season,cache_path):
    cache_path=Path(cache_path)
    roster_path=cache_path/f'roster_{season}.parquet'
    roster=pd.read_parquet(roster_path).fillna('') if roster_path.exists() else pd.DataFrame()
    history=[]
    for y in range(season-3,season):
        path=cache_path/f'stats_{y}.parquet'
        if path.exists():
            df=pd.read_parquet(path)
            if all(k in df for k in K_STATS):
                df=df[(df.position=='K')&(df.season_type=='REG')]
                history.append(df)
    result=[]
    if history:
        df=pd.concat(history)
        for pid,rows in df.groupby('player_id'):
            current=roster[(roster.gsis_id==pid)&(roster.position=='K')] if not roster.empty else pd.DataFrame()
            if current.empty:continue
            row=current.iloc[-1]
            weights=np.power(.55,season-1-rows.season.to_numpy())
            stats={k:float(np.dot(rows[k].fillna(0),weights)/weights.sum()*16) for k in K_STATS}
            ids={'gsis':str(pid)}
            if row.get('espn_id'):ids['espn']=str(row.espn_id).removesuffix('.0')
            result.append(Player(id=str(pid),name=str(row.full_name),position='K',team=str(row.team),ids=ids,stats=stats,games=16,workload_cv=.25,efficiency_cv=.12,warnings=['K model uses historical per-game component rates; verify active kicker','ADP unavailable']))
    team_path=cache_path/f'team_stats_{season-1}.parquet'
    if team_path.exists():
        df=pd.read_parquet(team_path)
        df=df[df.season_type=='REG']
        prior=df[D_STATS].mean()
        for team,rows in df.groupby('team'):
            stats={k:float((rows[k].fillna(0).mean()*.6+prior[k]*.4)*17) for k in D_STATS}
            result.append(Player(id=f'dst:{team}',name=f'{team} Defense / ST',position='DST',team=str(team),stats=stats,games=17,workload_cv=.12,efficiency_cv=.35,warnings=['DST uses regressed prior-year components; points/yards-allowed tiers require explicit supported component import','ESPN DST ID requires explicit mapping','ADP unavailable']))
    return result


def reconcile_opportunity(players,season,cache_path):
    path=Path(cache_path)/f'team_stats_{season-1}.parquet'
    budgets={}
    if path.exists():
        df=pd.read_parquet(path)
        df=df[df.season_type=='REG']
        budgets=df.groupby('team')[['attempts','carries']].sum().to_dict('index')
    for team in {p.team for p in players}:
        pool=[p for p in players if p.team==team and p.position in ('QB','RB','WR','TE')]
        budget=budgets.get(team,{'attempts':560,'carries':440})
        for stat,related,cap in [('attempts',['passing_yards','passing_tds','passing_interceptions'],budget['attempts']),('targets',['receptions','receiving_yards','receiving_tds'],budget['attempts']*.96),('carries',['rushing_yards','rushing_tds'],budget['carries'])]:
            total=sum(p.stats.get(stat,0) for p in pool)
            factor=min(1,cap/max(1,total))
            if factor<.999:
                for p in pool:
                    for k in [stat]+related:
                        if k in p.stats:p.stats[k]*=factor
                    p.warnings.append(f'Team {stat} budget reconciliation: {factor:.2f} multiplier')
    return players
