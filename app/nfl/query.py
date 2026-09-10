"""DuckDB views over the engine's Parquet files for ad-hoc analysis.

    from app.nfl.query import connect
    con = connect()
    con.sql("select season, avg(abs(actual_home_points-actual_away_points-pred_margin)) from backtest_ensemble_market_free group by 1 order by 1").show()
"""
from __future__ import annotations
from pathlib import Path
import duckdb
from app.nfl.config import paths, FEATURE_VERSION


def connect(root: Path | None = None) -> duckdb.DuckDBPyConnection:
    p = paths(root)
    con = duckdb.connect()
    views = {
        "games": p.normalized / "games.parquet", "team_games": p.normalized / "team_games.parquet", "qb_games": p.normalized / "qb_games.parquet", "kicker_games": p.normalized / "kicker_games.parquet",
        "injuries": p.normalized / "injuries.parquet", "snaps": p.normalized / "snaps.parquet", "depth_charts": p.normalized / "depth_charts.parquet", "rosters": p.normalized / "rosters.parquet", "players": p.normalized / "players.parquet",
        "features_pregame": p.features / FEATURE_VERSION / "games_pregame.parquet", "features_early": p.features / FEATURE_VERSION / "games_early.parquet",
    }
    for path in (p.predictions / "historical").glob("*.parquet"):
        views[path.stem] = path
    for path in (p.predictions / "current").glob("*.parquet"):
        views[f"current_{path.stem}"] = path
    plays = p.normalized / "plays"
    if plays.exists():
        views["plays"] = plays / "*.parquet"
    for name, path in views.items():
        if str(path).endswith("*.parquet") or Path(path).exists():
            con.execute(f"CREATE OR REPLACE VIEW {name} AS SELECT * FROM read_parquet('{str(path).replace(chr(92), '/')}')")
    return con
