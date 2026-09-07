"""Rolling-origin evaluation. A held-out season never supplies training features."""
from __future__ import annotations
import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from app.domain import League
from app.projections import project_history, distribution
from app.scoring import score
from app.data import STAT_COLUMNS


def metrics(pred, actual):
    a, b = np.asarray(pred), np.asarray(actual)
    return {"n":len(a), "mae":float(np.mean(np.abs(a-b))), "rmse":float(np.sqrt(np.mean((a-b)**2))), "spearman":float(spearmanr(a,b).statistic) if len(a)>2 and np.std(a)>0 and np.std(b)>0 else None}


def backtest(df, league: League):
    years = sorted(int(s) for s in df.season.unique())
    outputs = []
    training_x, training_y = [], []
    for season in years[1:]:
        projected = project_history(df,season)
        actual = df[df.season == season].set_index("player_id")
        previous = df[df.season == season-1].set_index("player_id")
        x, y, baseline, predictions, coverage = [], [], [], [], []
        for p in projected:
            # Missing following-year output is zero, not silently excluded survivor bias.
            target = score(actual.loc[p.id].to_dict(),league) if p.id in actual.index else 0.
            last = score(previous.loc[p.id].to_dict(),league)
            features = [p.stats.get(s,0) for s in STAT_COLUMNS] + [p.games, p.workload_cv] + [int(p.position==s) for s in ("QB","RB","WR","TE")]
            x.append(features)
            y.append(target)
            baseline.append(last)
            predictions.append(score(p.stats,league))
            _, dist = distribution(p,league,n=512)
            coverage.append(dist["p10"] <= target <= dist["p90"])
        if not y:
            continue
        row = {"season":season, "independent":metrics(predictions,y), "last_season":metrics(baseline,y), "p10_p90_coverage":float(np.mean(coverage)), "nominal_coverage":.8, "eligible_population":"Players with prior-year NFL stats, including next-year nonparticipants at zero; rookies excluded"}
        if len(training_x) >= 100:
            tree = HistGradientBoostingRegressor(max_iter=100,max_leaf_nodes=10,l2_regularization=15,random_state=42)
            tree.fit(training_x,training_y)
            row["tree"] = metrics(np.maximum(0,tree.predict(x)),y)
        # Only after scoring the holdout may its labels enter training for the next fold.
        training_x.extend(x)
        training_y.extend(y)
        outputs.append(row)
    averages = {}
    for model in ("independent","last_season","tree"):
        folds = [r[model] for r in outputs if model in r]
        if folds:
            averages[model] = sum(r["mae"]*r["n"] for r in folds)/sum(r["n"] for r in folds)
    comparable = [r for r in outputs if all(m in r for m in ("independent","last_season","tree"))]
    choice = min(("independent","last_season","tree"),key=lambda m:np.mean([r[m]["mae"] for r in comparable])) if comparable else None
    return {"folds":outputs,"weighted_mae":averages,"best_on_common_folds":choice,"production_model":"opportunity-efficiency", "caveats":["Selection is diagnostic; production does not silently switch models.","Uncertainty calibration is measured, not assumed.","Historical data revisions are not point-in-time archived; year cutoffs prevent target-season leakage, but this is not a fully vintage-correct backtest.","No historical ADP/ECR archives imported: cannot establish optimizer alpha against market draft strategies."]}
