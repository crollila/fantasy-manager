"""Assemble the game-level feature matrix for a prediction horizon.

Every row carries ``prediction_timestamp`` (kickoff minus the horizon) and every feature is
derived from data stamped before it. Feature families are recorded in a sidecar JSON so
ablations and the market-free / market-aware split are explicit.
"""
from __future__ import annotations
import json
import logging
import time
from pathlib import Path
import polars as pl
from app.nfl.config import paths, FEATURE_VERSION, HORIZONS
from app.nfl.features.rolling import prepare_team_games, rolling_features
from app.nfl.features.ratings import elo_features, margin_ratings, efficiency_ratings
from app.nfl.features.qb import qb_features
from app.nfl.features.availability import availability_features
from app.nfl.features.context import game_context, coaching_features, referee_features, market_features

log = logging.getLogger(__name__)
TARGETS = ["home_score", "away_score", "margin", "total_points", "home_win"]
ID_COLUMNS = ["game_id", "season", "week", "game_type", "kickoff_utc", "prediction_timestamp", "home_team", "away_team", "completed"]

# Team-level features that get explicit home-minus-away differences and matchup products.
DIFF_KEYS = [
    "elo_pre", "srs", "off_adj_epa_play", "def_adj_epa_play", "off_adj_epa_pass", "def_adj_epa_pass", "off_adj_epa_rush", "def_adj_epa_rush", "off_adj_success_rate", "def_adj_success_rate", "off_adj_points_drive", "def_adj_points_drive",
    "off_adj_epa_neutral", "def_adj_epa_neutral", "off_adj_explosive20", "def_adj_explosive20", "epa_play__ewm", "epa_play_allowed__ewm", "epa_play__std", "epa_play_allowed__std", "epa_play__prev", "epa_play_allowed__prev",
    "points__ewm", "points_allowed__ewm", "margin__ewm", "margin__l8", "margin__prev", "success_rate__ewm", "success_rate_allowed__ewm", "epa_pass__ewm", "epa_pass_allowed__ewm", "epa_rush__ewm", "epa_rush_allowed__ewm",
    "qb_epa_career", "qb_epa_season", "qb_epa_last4", "qb_epa_ewm", "qb_cpoe_career", "value_lost_off", "value_lost_def", "inj_lost_QB", "inj_lost_OL", "rest", "travel", "tz_shift", "sec_per_play_neutral__ewm", "pass_rate_neutral__ewm", "proe__ewm",
    "coach_tenure_games", "games_played_std", "wins__std", "win_pct__prev", "sack_rate__ewm", "sack_rate_allowed__ewm", "explosive20__ewm", "explosive20_allowed__ewm", "three_out_rate__ewm", "three_out_rate_allowed__ewm", "turnover_drive__ewm", "turnover_drive_allowed__ewm",
]


def family_of(name: str) -> str:
    base = name.removeprefix("home_").removeprefix("away_").removeprefix("diff_").removeprefix("mx_")
    if name.startswith("mkt_"):
        return "market"
    if name.startswith("mx_"):
        return "matchup"
    if base.startswith("elo"):
        return "elo"
    if base.startswith("srs"):
        return "ratings_margin"
    if base.startswith(("off_adj", "def_adj", "league_", "eff_games")):
        return "ratings_efficiency"
    if base.startswith("qb_"):
        return "qb"
    if base.startswith(("inj_", "starters_", "reserve_", "value_lost", "injury_", "roster_source", "dnp_count", "qb_starter_")):
        return "availability"
    if base.startswith("ref_"):
        return "officials"
    if base.startswith("coach_"):
        return "coaching"
    if base.startswith("wx_"):
        return "weather"
    if base.startswith(("rest", "travel", "log_travel", "tz_shift", "body_clock", "short_week", "bye")) or base in ("rest_diff",):
        return "rest_travel"
    if base.startswith("ctx_"):
        return "schedule"
    if "_allowed__" in base:
        return "team_form_defense"
    if "__" in base or base.startswith(("games_played", "wins__")):
        return "team_form_offense"
    return "other"


def build_features(horizon: str = "pregame", root: Path | None = None, write: bool = True) -> pl.DataFrame:
    p = paths(root)
    hours = HORIZONS[horizon]
    t0 = time.time()
    n = p.normalized
    games = pl.read_parquet(n / "games.parquet").filter(pl.col("kickoff_utc").is_not_null())
    team_games = pl.read_parquet(n / "team_games.parquet")
    qb_games = pl.read_parquet(n / "qb_games.parquet")
    players = pl.read_parquet(n / "players.parquet")
    injuries = pl.read_parquet(n / "injuries.parquet") if (n / "injuries.parquet").exists() else pl.DataFrame()
    snaps = pl.read_parquet(n / "snaps.parquet") if (n / "snaps.parquet").exists() else pl.DataFrame()
    rosters = pl.read_parquet(n / "rosters.parquet") if (n / "rosters.parquet").exists() else pl.DataFrame()
    depth = pl.read_parquet(n / "depth_charts.parquet") if (n / "depth_charts.parquet").exists() else pl.DataFrame()

    # Team-game rows for future games (no play-by-play yet) so rolling windows can be evaluated for them.
    played_ids = set(team_games["game_id"].to_list())
    future = games.filter(~pl.col("game_id").is_in(list(played_ids)))
    stub_rows = []
    for side, other in (("home", "away"), ("away", "home")):
        stub_rows.append(future.select(["game_id", pl.col(f"{side}_team").alias("team"), pl.col(f"{other}_team").alias("opponent"), pl.lit(side == "home").alias("is_home"), "season", "week", "kickoff_utc", "game_type", "playoff"]))
    stubs = pl.concat(stub_rows)
    tg_all = pl.concat([team_games, stubs], how="diagonal_relaxed")
    prep = prepare_team_games(tg_all, games)
    prep = prep.with_columns([pl.when(pl.col(c).is_between(15, 50)).then(pl.col(c)).otherwise(None).alias(c) for c in ("sec_per_play_neutral", "sec_per_play") if c in prep.columns])
    roll = rolling_features(prep)
    log.info("rolling features %s in %.1fs", roll.shape, time.time() - t0)
    elo = elo_features(games)
    srs = margin_ratings(games)
    eff = efficiency_ratings(games, prep.filter(pl.col("game_id").is_in(list(played_ids))))
    log.info("ratings done %.1fs", time.time() - t0)
    qb = qb_features(games, qb_games, players, horizon=horizon, horizon_hours=hours)
    avail = availability_features(games, injuries, snaps, rosters, depth, horizon_hours=hours) if injuries.height else None
    ctx = game_context(games)
    coach = coaching_features(games)
    ref = referee_features(games, team_games)
    mkt = market_features(games)
    log.info("side features done %.1fs", time.time() - t0)

    frame = games.select(["game_id", "season", "week", "game_type", "kickoff_utc", "home_team", "away_team", "completed", "home_score", "away_score", "margin", "total_points"])
    frame = frame.with_columns([(pl.col("kickoff_utc") - pl.duration(hours=hours)).alias("prediction_timestamp"), pl.when(pl.col("completed")).then((pl.col("home_score") > pl.col("away_score")).cast(pl.Float64)).otherwise(None).alias("home_win")])
    for side in ("home", "away"):
        team_col = f"{side}_team"
        block = roll.drop(["season", "week", "kickoff_utc"])
        frame = frame.join(block.rename({c: f"{side}_{c}" for c in block.columns if c not in ("game_id", "team")}).rename({"team": team_col}), on=["game_id", team_col], how="left")
        s = srs.rename({"srs": f"{side}_srs", "srs_hfa": "srs_hfa", "srs_games": "srs_games"}).rename({"team": team_col})
        frame = frame.join(s.select(["season", "week", team_col, f"{side}_srs"] + (["srs_hfa", "srs_games"] if side == "home" else [])), on=["season", "week", team_col], how="left")
        e = eff.rename({c: f"{side}_{c}" for c in eff.columns if c not in ("season", "week", "team")}).rename({"team": team_col})
        e = e.drop([c for c in e.columns if c.startswith(f"{side}_league_") and side == "away"] + ([f"{side}_eff_games"] if side == "away" else []))
        frame = frame.join(e, on=["season", "week", team_col], how="left")
        q = qb.rename({c: f"{side}_{c}" for c in qb.columns if c not in ("game_id", "team")}).rename({"team": team_col})
        frame = frame.join(q, on=["game_id", team_col], how="left")
        if avail is not None:
            a = avail.rename({c: f"{side}_{c}" for c in avail.columns if c not in ("game_id", "team")}).rename({"team": team_col})
            frame = frame.join(a, on=["game_id", team_col], how="left")
        c = coach.rename({col: f"{side}_{col}" for col in coach.columns if col not in ("game_id", "team")}).rename({"team": team_col})
        frame = frame.join(c, on=["game_id", team_col], how="left")
    frame = frame.join(elo, on="game_id", how="left").join(ctx.drop(["season", "week", "kickoff_utc"]), on="game_id", how="left").join(ref, on="game_id", how="left").join(mkt, on="game_id", how="left")
    frame = frame.rename({"home_league_epa_play": "league_epa_play", "home_league_epa_pass": "league_epa_pass", "home_league_epa_rush": "league_epa_rush", "home_league_success_rate": "league_success_rate", "home_league_points_drive": "league_points_drive", "home_league_epa_neutral": "league_epa_neutral", "home_league_explosive20": "league_explosive20", "home_eff_games": "eff_games"})
    frame = frame.with_columns([pl.col("home_elo_pre").alias("home_elo_pre"), pl.col("away_elo_pre").alias("away_elo_pre")])
    # Differences and matchup interactions.
    diffs = []
    for key in DIFF_KEYS:
        h, a = f"home_{key}", f"away_{key}"
        if h in frame.columns and a in frame.columns:
            diffs.append((pl.col(h) - pl.col(a)).alias(f"diff_{key}"))
    frame = frame.with_columns(diffs)
    mx = []
    def add_mx(name: str, expr: pl.Expr, cols: list[str]):
        if all(c in frame.columns for c in cols):
            mx.append(expr.alias(name))
    add_mx("mx_home_pass", pl.col("home_off_adj_epa_pass") + pl.col("away_def_adj_epa_pass"), ["home_off_adj_epa_pass", "away_def_adj_epa_pass"])
    add_mx("mx_away_pass", pl.col("away_off_adj_epa_pass") + pl.col("home_def_adj_epa_pass"), ["away_off_adj_epa_pass", "home_def_adj_epa_pass"])
    add_mx("mx_home_rush", pl.col("home_off_adj_epa_rush") + pl.col("away_def_adj_epa_rush"), ["home_off_adj_epa_rush", "away_def_adj_epa_rush"])
    add_mx("mx_away_rush", pl.col("away_off_adj_epa_rush") + pl.col("home_def_adj_epa_rush"), ["away_off_adj_epa_rush", "home_def_adj_epa_rush"])
    add_mx("mx_home_total", pl.col("home_off_adj_epa_play") + pl.col("away_def_adj_epa_play"), ["home_off_adj_epa_play", "away_def_adj_epa_play"])
    add_mx("mx_away_total", pl.col("away_off_adj_epa_play") + pl.col("home_def_adj_epa_play"), ["away_off_adj_epa_play", "home_def_adj_epa_play"])
    add_mx("mx_net", (pl.col("home_off_adj_epa_play") + pl.col("away_def_adj_epa_play")) - (pl.col("away_off_adj_epa_play") + pl.col("home_def_adj_epa_play")), ["home_off_adj_epa_play", "away_def_adj_epa_play", "away_off_adj_epa_play", "home_def_adj_epa_play"])
    add_mx("mx_home_points_drive", pl.col("home_off_adj_points_drive") + pl.col("away_def_adj_points_drive"), ["home_off_adj_points_drive", "away_def_adj_points_drive"])
    add_mx("mx_away_points_drive", pl.col("away_off_adj_points_drive") + pl.col("home_def_adj_points_drive"), ["away_off_adj_points_drive", "home_def_adj_points_drive"])
    add_mx("mx_home_explosive", pl.col("home_explosive20__ewm") + pl.col("away_explosive20_allowed__ewm"), ["home_explosive20__ewm", "away_explosive20_allowed__ewm"])
    add_mx("mx_away_explosive", pl.col("away_explosive20__ewm") + pl.col("home_explosive20_allowed__ewm"), ["away_explosive20__ewm", "home_explosive20_allowed__ewm"])
    add_mx("mx_home_pressure", pl.col("home_sack_rate__ewm") + pl.col("away_sack_rate_allowed__ewm"), ["home_sack_rate__ewm", "away_sack_rate_allowed__ewm"])
    add_mx("mx_away_pressure", pl.col("away_sack_rate__ewm") + pl.col("home_sack_rate_allowed__ewm"), ["away_sack_rate__ewm", "home_sack_rate_allowed__ewm"])
    add_mx("mx_home_qb_vs_pass_d", pl.col("home_qb_epa_career") + pl.col("away_def_adj_epa_pass"), ["home_qb_epa_career", "away_def_adj_epa_pass"])
    add_mx("mx_away_qb_vs_pass_d", pl.col("away_qb_epa_career") + pl.col("home_def_adj_epa_pass"), ["away_qb_epa_career", "home_def_adj_epa_pass"])
    add_mx("mx_pace", pl.col("home_sec_per_play_neutral__ewm") + pl.col("away_sec_per_play_neutral__ewm"), ["home_sec_per_play_neutral__ewm", "away_sec_per_play_neutral__ewm"])
    add_mx("mx_pass_rate", pl.col("home_pass_rate_neutral__ewm") + pl.col("away_pass_rate_neutral__ewm"), ["home_pass_rate_neutral__ewm", "away_pass_rate_neutral__ewm"])
    add_mx("mx_home_rz", pl.col("home_rz_td_rate__ewm") + pl.col("away_rz_td_rate_allowed__ewm"), ["home_rz_td_rate__ewm", "away_rz_td_rate_allowed__ewm"])
    add_mx("mx_away_rz", pl.col("away_rz_td_rate__ewm") + pl.col("home_rz_td_rate_allowed__ewm"), ["away_rz_td_rate__ewm", "home_rz_td_rate_allowed__ewm"])
    add_mx("mx_home_third", pl.col("home_third_conv__ewm") + pl.col("away_third_conv_allowed__ewm"), ["home_third_conv__ewm", "away_third_conv_allowed__ewm"])
    add_mx("mx_away_third", pl.col("away_third_conv__ewm") + pl.col("home_third_conv_allowed__ewm"), ["away_third_conv__ewm", "home_third_conv_allowed__ewm"])
    add_mx("mx_wind_pass", pl.col("wx_wind_over10") * (pl.col("home_pass_rate_neutral__ewm") + pl.col("away_pass_rate_neutral__ewm")), ["wx_wind_over10", "home_pass_rate_neutral__ewm", "away_pass_rate_neutral__ewm"])
    add_mx("mx_elo_x_qbchange", pl.col("elo_diff") * (pl.col("home_qb_changed") - pl.col("away_qb_changed")), ["elo_diff", "home_qb_changed", "away_qb_changed"])
    frame = frame.with_columns(mx)
    frame = frame.sort(["kickoff_utc", "game_id"])
    numeric_types = (pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Int16, pl.Int8, pl.UInt32, pl.UInt64, pl.Boolean)
    feature_cols = [c for c, t in zip(frame.columns, frame.dtypes) if c not in ID_COLUMNS + TARGETS + ["home_score", "away_score", "margin", "total_points"] and not c.endswith("_qb_id") and t in numeric_types]
    families = {c: family_of(c) for c in feature_cols}
    meta = {"feature_version": FEATURE_VERSION, "horizon": horizon, "horizon_hours": hours, "rows": frame.height, "features": len(feature_cols), "families": {f: sorted(c for c, fam in families.items() if fam == f) for f in sorted(set(families.values()))}, "built_in_seconds": round(time.time() - t0, 1),
            "notes": ["weather uses observed game conditions from nflverse (labelled 'observed'; not a pregame forecast)", "market features are closing lines without timestamps; only the market-aware model may use them", "injury reports without a timestamp (2009, 2025+) are assumed public 48h before kickoff"]}
    if write:
        folder = p.features / FEATURE_VERSION
        folder.mkdir(parents=True, exist_ok=True)
        frame.write_parquet(folder / f"games_{horizon}.parquet")
        (folder / f"meta_{horizon}.json").write_text(json.dumps(meta, indent=2))
    log.info("features %s: %s rows x %s features in %.1fs", horizon, frame.height, len(feature_cols), time.time() - t0)
    return frame


def load_features(horizon: str = "pregame", root: Path | None = None) -> tuple[pl.DataFrame, dict]:
    folder = paths(root).features / FEATURE_VERSION
    return pl.read_parquet(folder / f"games_{horizon}.parquet"), json.loads((folder / f"meta_{horizon}.json").read_text())


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    for h in (sys.argv[1:] or ["pregame"]):
        build_features(h)
