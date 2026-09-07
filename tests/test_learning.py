import numpy as np
import pandas as pd
from app.data import STAT_COLUMNS
from app.learning import fit_selected
from app.projections import project_history
from app.calibration import audit

def test_model_does_not_train_on_target_season():
    rows=[]
    for season in range(2021,2026):
        for i in range(40):
            rows.append({k:0. for k in STAT_COLUMNS}|{'player_id':str(i),'season':season,'games':16,'position':'WR','player_display_name':str(i),'team':'A','targets':40+i,'receptions':25+i*.6,'receiving_yards':300+i*6,'receiving_tds':3})
    df=pd.DataFrame(rows)
    first,_=fit_selected(df,2025,project_history(df,2025))
    df.loc[df.season==2025,'targets']=1000000
    second,_=fit_selected(df,2025,project_history(df,2025))
    assert [p.stats for p in first]==[p.stats for p in second]

def test_calibration_does_not_use_current_outcomes():
    rows=[{'season':2022,'position':'WR','predicted':100,'actual':90+i%20} for i in range(100)]
    rows += [{'season':2023,'position':'WR','predicted':100,'actual':100} for _ in range(40)]
    a=audit(rows)
    assert a['folds'][0]['p10_p90_coverage']==1
    for row in rows[100:]:row['actual']=1000
    b=audit(rows)
    assert b['folds'][0]['mean_interval_width']==a['folds'][0]['mean_interval_width']
    assert b['folds'][0]['p10_p90_coverage']==0
