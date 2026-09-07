"""Expanding-window model selection and empirical residual calibration.

The learned candidate predicts workload and availability first. Historical,
regressed efficiency then converts workload into stat components. Persistence is
a legitimate competing baseline and may win. No consensus labels are involved.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from app.data import STAT_COLUMNS
from app.domain import DEFAULT_SCORING
from app.projections import project_history

OUTPUTS=['attempts','carries','targets','games']
GROUPS={'attempts':['passing_yards','passing_tds','passing_interceptions'], 'carries':['rushing_yards','rushing_tds'], 'targets':['receptions','receiving_yards','receiving_tds']}

def feature(p,previous):
    return [p.stats.get(k,0) for k in STAT_COLUMNS]+[float(previous.get(k,0)) for k in STAT_COLUMNS]+[p.games,p.workload_cv]+[int(p.position==q) for q in ('QB','RB','WR','TE')]

def points(stats):
    return sum(float(stats.get(k,0))*v for k,v in DEFAULT_SCORING.items())

def workload_stats(p,prediction):
    stats=p.stats.copy()
    for key,value in zip(OUTPUTS,prediction):
        if key=='games':continue
        old=max(.001,stats.get(key,0))
        stats[key]=max(0,float(value))
        for related in GROUPS[key]:stats[related]=stats.get(related,0)*stats[key]/old
    return stats

def fit_selected(df,season,players,output_path=None):
    years=sorted(int(y) for y in df.season.unique() if y<season)
    training_x=[];training_y=[];folds=[];residuals=[]
    errors={name:[] for name in ('opportunity','persistence','workload_tree')}
    for heldout in years[1:]:
        projected=project_history(df,heldout)
        previous=df[df.season==heldout-1].set_index('player_id')
        actual=df[df.season==heldout].set_index('player_id')
        xs=[feature(p,previous.loc[p.id]) for p in projected]
        model=None
        if len(training_x)>100:
            model=ExtraTreesRegressor(n_estimators=80,max_depth=8,min_samples_leaf=12,max_features=.8,random_state=42,n_jobs=1)
            model.fit(training_x,training_y)
            predictions=model.predict(xs)
        fold={name:[] for name in errors}
        previous_mae={name:float(np.mean(values)) for name,values in errors.items() if values and (name!='workload_tree' or model is not None)}
        selected=min(previous_mae,key=previous_mae.get) if previous_mae else 'persistence'
        for i,p in enumerate(projected):
            target=actual.loc[p.id].to_dict() if p.id in actual.index else {k:0. for k in STAT_COLUMNS+['games']}
            candidates={'opportunity':p.stats,'persistence':{k:float(previous.loc[p.id].get(k,0)) for k in STAT_COLUMNS}}
            if model is not None:candidates['workload_tree']=workload_stats(p,predictions[i])
            truth=points(target)
            for name,stats in candidates.items():fold[name].append(abs(points(stats)-truth))
            chosen=candidates[selected]
            # Honest residual: model choice uses only earlier completed folds.
            residuals.append({'season':heldout,'position':p.position,'predicted':points(chosen),'actual':truth,'predicted_stats':chosen,'stats_error':{k:float(target.get(k,0))-chosen.get(k,0) for k in STAT_COLUMNS},'selected':selected})
            training_x.append(xs[i]);training_y.append([float(target.get(k,0)) for k in OUTPUTS])
        for name,values in fold.items():errors[name].extend(values)
        folds.append({'season':heldout,'n':len(projected),'mae':{name:float(np.mean(v)) for name,v in fold.items() if v},'selected_from_earlier_folds':selected})
    # Compare models on common folds, not disjoint windows.
    common=[f for f in folds if all(name in f['mae'] for name in errors)]
    maes={name:sum(f['mae'][name]*f['n'] for f in common)/sum(f['n'] for f in common) for name in errors} if common else {'persistence':0}
    selected=min(maes,key=maes.get)
    production=project_history(df,season)
    previous=df[df.season==season-1].set_index('player_id')
    learned=None
    if selected=='workload_tree':
        model=ExtraTreesRegressor(n_estimators=80,max_depth=8,min_samples_leaf=12,max_features=.8,random_state=42,n_jobs=1)
        model.fit(training_x,training_y)
        learned=model.predict([feature(p,previous.loc[p.id]) for p in production])
    predictions={p.id:(p,i) for i,p in enumerate(production)}
    for p in players:
        if p.id not in predictions:continue
        base,i=predictions[p.id]
        if selected=='persistence':
            p.stats={k:float(previous.loc[p.id].get(k,0)) for k in STAT_COLUMNS}
            p.games=min(17,max(0,float(previous.loc[p.id]['games'])))
        elif selected=='workload_tree':
            p.stats=workload_stats(base,learned[i]);p.games=min(17,max(0,float(learned[i,3])))
        p.source=f'independent:{selected}'
        p.warnings.append(f'Production family selected by expanding-window common-fold MAE: {selected}')
    report={'selected':selected,'common_fold_mae':maes,'folds':folds,'residuals':residuals,'calibration':'Rolling-origin empirical residuals, grouped by position and forecast level. Not a coverage guarantee for 2026.','limitations':['Revised historical data, not archived vintages','Rookies absent from backtest','Current roster and team-budget corrections not represented in historical folds','No evidence yet of championship advantage']}
    if output_path:
        Path(output_path).write_text(json.dumps(report,indent=2))
    return players,report

def calibrate_samples(samples,player,league,records,seed=117):
    compatible=[r for r in records if r['position']==player.position]
    if len(compatible)<30:return samples,None
    forecast=sum(player.stats.get(k,0)*v for k,v in league.scoring.items())
    # Local neighborhood prevents bench residuals from dominating high-volume starters.
    compatible.sort(key=lambda r:abs(sum(r.get('predicted_stats',{}).get(k,0)*v for k,v in league.scoring.items())-forecast))
    nearest=compatible[:max(40,min(150,len(compatible)//3))]
    errors=np.array([sum(r['stats_error'].get(k,0)*v for k,v in league.scoring.items()) for r in nearest])
    # Empirical predictive draws expose nonparticipation as a zero floor.
    import hashlib
    rng=np.random.default_rng(seed+int.from_bytes(hashlib.sha256(player.id.encode()).digest()[:4],"little"))
    forecast=float(np.mean(samples))
    weight=player.market_weight if player.market_stats is not None else 0
    empirical=np.maximum(0,forecast+(1-weight)*rng.choice(errors,len(samples),replace=True))
    # Keep modeled workload/efficiency dispersion when larger; never narrow empirical tails.
    center=np.median(samples)
    empirical_spread=np.quantile(empirical,.9)-np.quantile(empirical,.1)
    model_spread=np.quantile(samples,.9)-np.quantile(samples,.1)
    if model_spread>empirical_spread:
        empirical=np.maximum(0,(empirical-np.median(empirical))*model_spread/max(.01,empirical_spread)+np.median(empirical))
    return empirical,{'observations':len(nearest),'method':'rolling-origin local residual bootstrap','bias_correction':float(np.mean(errors))}
