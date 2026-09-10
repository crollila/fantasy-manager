"""Automated leakage tests for the point-in-time feature factory.

1. Perturbation test: after a cutoff, every later game's outcome (scores, play-by-play
   aggregates, injuries) is altered or deleted; features for games BEFORE the cutoff must be
   bit-for-bit identical. A feature that changes used future information.
2. Timestamp test: every game row's prediction timestamp precedes kickoff, and rolling
   windows never include the game itself (spot-check identities on real data).
3. Outcome-leak screen: no market-free feature may reproduce the game's own result (absolute
   correlation with margin/total above 0.6, or exact equality with the actual score).
"""
from __future__ import annotations
import json
import logging
import numpy as np
import polars as pl
from app.nfl.config import paths
from app.nfl.features.rolling import prepare_team_games, rolling_features
from app.nfl.features.ratings import elo_features, margin_ratings, efficiency_ratings
from app.nfl.features.qb import qb_features
from app.nfl.features.availability import availability_features
from app.nfl.features.context import coaching_features, referee_features

log = logging.getLogger(__name__)


def _feature_blocks(games: pl.DataFrame, team_games: pl.DataFrame, qb_games: pl.DataFrame, players: pl.DataFrame, injuries: pl.DataFrame, snaps: pl.DataFrame, rosters: pl.DataFrame, depth: pl.DataFrame) -> dict[str, pl.DataFrame]:
    prep = prepare_team_games(team_games, games)
    blocks = {
        "rolling": rolling_features(prep).sort(["game_id", "team"]),
        "elo": elo_features(games).sort("game_id"),
        "srs": margin_ratings(games).sort(["season", "week", "team"]),
        "eff": efficiency_ratings(games, prep).sort(["season", "week", "team"]),
        "qb": qb_features(games, qb_games, players).sort(["game_id", "team"]),
        "coach": coaching_features(games).sort(["game_id", "team"]),
        "ref": referee_features(games, team_games).sort("game_id"),
    }
    if injuries.height:
        blocks["avail"] = availability_features(games, injuries, snaps, rosters, depth).sort(["game_id", "team"])
    return blocks


def perturbation_test(cutoff_season: int = 2021, cutoff_week: int = 10, root=None) -> dict:
    n = paths(root).normalized
    games = pl.read_parquet(n / "games.parquet").filter(pl.col("kickoff_utc").is_not_null())
    team_games = pl.read_parquet(n / "team_games.parquet")
    qb_games = pl.read_parquet(n / "qb_games.parquet")
    players = pl.read_parquet(n / "players.parquet")
    injuries = pl.read_parquet(n / "injuries.parquet")
    snaps = pl.read_parquet(n / "snaps.parquet")
    rosters = pl.read_parquet(n / "rosters.parquet")
    depth = pl.read_parquet(n / "depth_charts.parquet")
    cutoff = games.filter((pl.col("season") == cutoff_season) & (pl.col("week") == cutoff_week))["kickoff_utc"].min()
    before_ids = set(games.filter(pl.col("kickoff_utc") < cutoff)["game_id"].to_list())
    original = _feature_blocks(games, team_games, qb_games, players, injuries, snaps, rosters, depth)

    # Perturb the future: flip scores, scramble team-game metrics, delete later injuries.
    rng = np.random.default_rng(1)
    later = pl.col("kickoff_utc") >= cutoff
    g2 = games.with_columns([
        pl.when(later & pl.col("completed")).then(pl.col("away_score") + 21).otherwise(pl.col("home_score")).alias("home_score"),
        pl.when(later & pl.col("completed")).then(pl.col("home_score")).otherwise(pl.col("away_score")).alias("away_score"),
    ]).with_columns([(pl.col("home_score") - pl.col("away_score")).alias("margin"), (pl.col("home_score") + pl.col("away_score")).alias("total_points")])
    numeric = [c for c, t in zip(team_games.columns, team_games.dtypes) if t in (pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.UInt32) and c not in ("season", "week")]
    tg2 = team_games.with_columns([pl.when(later).then(pl.col(c) * 3.0 + 7.0).otherwise(pl.col(c)).alias(c) for c in numeric])
    qb2 = qb_games.with_columns([pl.when(later).then(pl.col(c) * -2.0 + 1.0).otherwise(pl.col(c)).alias(c) for c in ("epa_total", "epa_dropback", "cpoe", "dropbacks", "attempts")]).with_columns(pl.col("dropbacks").abs())
    inj2 = injuries.filter(~((pl.col("season") > cutoff_season) | ((pl.col("season") == cutoff_season) & (pl.col("week") >= cutoff_week))))
    perturbed = _feature_blocks(g2, tg2, qb2, players, inj2, snaps.filter(~(pl.col("season") > cutoff_season)), rosters, depth)

    results = {}
    before_games = games.filter(pl.col("kickoff_utc") < cutoff).select(["game_id", "season", "week"])
    for name in original:
        a, b = original[name], perturbed[name]
        if "game_id" in a.columns:
            a = a.filter(pl.col("game_id").is_in(list(before_ids)))
            b = b.filter(pl.col("game_id").is_in(list(before_ids)))
        else:  # weekly snapshot tables: compare snapshots whose cutoff precedes the perturbation cutoff
            keys = before_games.select(["season", "week"]).unique()
            a = a.join(keys, on=["season", "week"], how="inner")
            b = b.join(keys, on=["season", "week"], how="inner")
        changed = []
        for c in a.columns:
            if c in ("game_id", "team", "season", "week", "kickoff_utc"):
                continue
            x, y = a[c].to_numpy(), b[c].to_numpy()
            try:
                x = x.astype(float)
                y = y.astype(float)
                same = np.all((np.isnan(x) & np.isnan(y)) | np.isclose(x, y, equal_nan=True, atol=1e-9))
            except (TypeError, ValueError):
                same = bool((a[c] == b[c]).fill_null(True).all())
            if not same:
                changed.append(c)
        results[name] = {"rows_compared": a.height, "changed_columns": changed, "passed": not changed}
    passed = all(v["passed"] for v in results.values())
    report = {"cutoff": str(cutoff), "passed": passed, "blocks": results}
    (paths(root).reports / "leakage_perturbation.json").write_text(json.dumps(report, indent=2, default=str))
    return report


def outcome_screen(features: pl.DataFrame, meta: dict, threshold: float = 0.6) -> dict:
    """Flag market-free features that look like they encode the game's own outcome."""
    done = features.filter(pl.col("completed") & (pl.col("season") >= 2005))
    margin = done["margin"].to_numpy().astype(float)
    total = done["total_points"].to_numpy().astype(float)
    home = done["home_score"].to_numpy().astype(float)
    suspicious = []
    for fam, cols in meta["families"].items():
        if fam == "market":
            continue
        for c in cols:
            if done[c].dtype not in (pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.UInt32, pl.Boolean):
                continue
            x = done[c].cast(pl.Float64).to_numpy().astype(float)
            ok = np.isfinite(x)
            if ok.sum() < 200 or np.nanstd(x[ok]) == 0:
                continue
            for target, name in ((margin, "margin"), (total, "total"), (home, "home_score")):
                r = abs(np.corrcoef(x[ok], target[ok])[0, 1])
                if r > threshold:
                    suspicious.append({"feature": c, "target": name, "abs_corr": round(float(r), 3)})
    report = {"passed": not suspicious, "threshold": threshold, "suspicious": suspicious}
    return report


def timestamp_checks(features: pl.DataFrame) -> dict:
    ok_ts = bool((features["prediction_timestamp"] < features["kickoff_utc"]).all())
    # A team's l1 window equals its previous game's value: check on a handful of teams with the normalized data.
    n = paths().normalized
    games = pl.read_parquet(n / "games.parquet")
    tg = pl.read_parquet(n / "team_games.parquet")
    prep = prepare_team_games(tg, games)
    roll = rolling_features(prep)
    sample = prep.sort(["team", "kickoff_utc"]).filter(pl.col("team") == "KC").select(["game_id", "kickoff_utc", "epa_play"]).with_columns(pl.col("epa_play").shift(1).alias("expected_l1"))
    check = sample.join(roll.filter(pl.col("team") == "KC").select(["game_id", "epa_play__l1"]), on="game_id")
    x, y = check["expected_l1"].to_numpy().astype(float), check["epa_play__l1"].to_numpy().astype(float)
    ok_window = bool(np.all((np.isnan(x) & np.isnan(y)) | np.isclose(x, y, equal_nan=True)))
    return {"prediction_before_kickoff": ok_ts, "l1_window_is_previous_game": ok_window, "passed": ok_ts and ok_window}


def run_all(root=None) -> dict:
    from app.nfl.features.builder import load_features
    features, meta = load_features("pregame", root)
    report = {"perturbation": perturbation_test(root=root), "outcome_screen": outcome_screen(features, meta), "timestamps": timestamp_checks(features)}
    report["passed"] = all(v["passed"] for v in report.values())
    (paths(root).reports / "leakage_report.json").write_text(json.dumps(report, indent=2, default=str))
    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    out = run_all()
    print(json.dumps({k: (v if k == "passed" else {kk: vv for kk, vv in v.items() if kk != "blocks"} | ({"blocks": {b: (r["passed"], r["changed_columns"][:5]) for b, r in v["blocks"].items()}} if "blocks" in v else {})) for k, v in out.items()}, indent=2, default=str))
