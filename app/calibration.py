"""Prospective interval audit: every calibration residual predates its test year."""
import numpy as np

def audit(records):
    folds=[]
    for season in sorted({r['season'] for r in records})[1:]:
        previous=[r for r in records if r['season']<season]
        test=[r for r in records if r['season']==season]
        covered=[];widths=[]
        for row in test:
            candidates=[r for r in previous if r['position']==row['position']]
            if len(candidates)<30:continue
            candidates.sort(key=lambda r:abs(r['predicted']-row['predicted']))
            nearest=candidates[:max(40,min(150,len(candidates)//3))]
            errors=[r['actual']-r['predicted'] for r in nearest]
            lo,hi=np.maximum(0,np.quantile(errors,[.1,.9])+row['predicted'])
            covered.append(lo<=row['actual']<=hi);widths.append(hi-lo)
        if covered:folds.append({'season':season,'n':len(covered),'p10_p90_coverage':float(np.mean(covered)),'mean_interval_width':float(np.mean(widths))})
    return {'nominal_coverage':.8,'folds':folds,'scope':'Prospective local residual intervals on reference PPR scoring; excludes 2026 role adjustments, market ensembles, bonuses and rookies.'}
