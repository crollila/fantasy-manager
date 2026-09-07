import pandas as pd
import numpy as np
from app.data import STAT_COLUMNS
from app.projections import project_history,apply_events
from app.domain import Player,Event

def test_no_future_information():
    rows=[]
    for season in (2022,2023,2024):
        row={s:0. for s in STAT_COLUMNS}|{'player_id':'x','season':season,'games':16,'position':'WR','player_display_name':'X','team':'A','targets':100,'receptions':65,'receiving_yards':900,'receiving_tds':6}
        rows.append(row)
    df=pd.DataFrame(rows)
    a=project_history(df,2024)[0]
    df.loc[df.season==2024,'targets']=100000
    b=project_history(df,2024)[0]
    assert a.stats==b.stats

def test_news_respects_known_at():
    p=Player(id='x',name='X',position='RB',stats={'carries':100},games=15)
    e=Event(id='e',player_id='x',kind='injured',occurred_at='2026-08-01T00:00:00Z',known_at='2026-08-02T00:00:00Z',source_url='https://example.org/report',note='test',games_delta=-5,confirmed=True)
    assert apply_events([p],[e],'2026-08-01T12:00:00Z')[0].games==15
    assert apply_events([p],[e],'2026-08-03T00:00:00Z')[0].games==10
    assert p.games==15
