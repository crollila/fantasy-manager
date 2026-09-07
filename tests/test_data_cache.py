from pathlib import Path
import httpx
import pandas as pd
from app.data import Cache

def test_optional_source_failure_preserves_cache(tmp_path,monkeypatch):
    path=tmp_path/'valid.parquet';pd.DataFrame({'id':[1]}).to_parquet(path)
    def offline(*args,**kwargs):raise httpx.ConnectError('offline')
    monkeypatch.setattr(httpx.Client,'get',offline)
    monkeypatch.setattr('app.data.time.sleep',lambda _:None)
    cache=Cache(tmp_path)
    assert cache.fetch('valid.parquet','https://example.invalid/source',force=True)==path
    assert pd.read_parquet(path).id.tolist()==[1]
    assert cache.status()[0]['status']=='stale'
