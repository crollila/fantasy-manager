"""Rolling / recency team features from prior games only.

For every (game, team) row the windows use games with an earlier kickoff. The current game
is excluded by construction (``shift(1)`` inside each team's chronological sequence), and a
team never has two games at the same timestamp, so ordering by kickoff is unambiguous.

Windows: last 1/3/5/8 games (any season), season-to-date (``std``), previous full season
(``prev``), exponentially weighted (``ewm``, half-life 4 games) and last 16 games (``l16``).
Defensive metrics are the opponent's offensive production in the same game (``*_allowed``).
"""
from __future__ import annotations
import polars as pl

# Metrics that receive the full set of windows. Others get ``ewm`` and ``std`` only.
CORE_METRICS = [
    "points", "points_allowed", "margin", "epa_play", "epa_pass", "epa_rush", "epa_neutral", "epa_pass_neutral", "epa_rush_neutral", "epa_early", "epa_late", "epa_rz",
    "success_rate", "success_pass", "success_rush", "success_early", "success_neutral", "third_conv", "ypp", "ypa", "ypc", "explosive20", "explosive_pass", "explosive_rush",
    "pass_rate_neutral", "proe", "cpoe", "sack_rate", "qb_hit_rate", "int_rate", "fumble_lost_rate", "giveaways", "points_drive", "epa_drive", "yards_drive", "plays_drive",
    "td_drive", "three_out_rate", "turnover_drive", "rz_td_rate", "sec_per_play_neutral", "plays", "drives", "penalty_yards", "st_epa", "fg_made_rate", "first_down_rate",
]
EXTENDED_METRICS = [
    "epa_third", "epa_gl", "epa_leading", "epa_trailing", "epa_tied", "epa_two_min", "epa_short", "epa_long", "success_late", "success_rz", "success_third", "fourth_conv", "fourth_attempts",
    "air_yards_att", "yac_comp", "explosive10", "explosive40", "deep_rate", "epa_deep", "short_pass_epa", "pass_rate", "pass_rate_close", "pass_rate_early", "xpass_mean", "comp_pct",
    "scramble_rate", "fumble_rate", "tfl_rate", "stuff_rate", "shotgun_rate", "no_huddle_rate", "xyac_epa_mean", "yac_epa_mean", "air_epa_mean", "penalties", "false_starts", "off_holding",
    "penalties_drawn", "penalty_yards_drawn", "dpi_drawn", "fg_epa", "punt_epa", "punt_distance", "fg_distance_mean", "rz_trip_rate", "punt_drive", "fg_drive", "downs_drive", "garbage_share", "sec_per_play", "neutral_plays",
]
ALLOWED_METRICS = [m for m in CORE_METRICS if m not in ("points", "points_allowed", "margin", "penalty_yards", "st_epa", "fg_made_rate", "plays", "drives")] + [
    "epa_third", "epa_gl", "explosive10", "deep_rate", "epa_deep", "air_yards_att", "yac_comp", "comp_pct", "stuff_rate", "tfl_rate", "fourth_conv", "penalties_drawn", "dpi_drawn", "rz_trip_rate", "pass_rate", "xyac_epa_mean",
]
WINDOWS = {"l1": 1, "l3": 3, "l5": 5, "l8": 8, "l16": 16}


def prepare_team_games(team_games: pl.DataFrame, games: pl.DataFrame) -> pl.DataFrame:
    """Attach scores and the opponent's offensive line (defensive 'allowed' metrics)."""
    scores = games.select(["game_id", "home_team", "away_team", "home_score", "away_score", "completed"])
    tg = team_games.join(scores.select(["game_id", "home_score", "away_score", "completed"]), on="game_id", how="left")
    tg = tg.with_columns([
        pl.when(pl.col("is_home")).then(pl.col("home_score")).otherwise(pl.col("away_score")).cast(pl.Float64).alias("points"),
        pl.when(pl.col("is_home")).then(pl.col("away_score")).otherwise(pl.col("home_score")).cast(pl.Float64).alias("points_allowed"),
        (pl.col("fg_made") / pl.when(pl.col("fg_att") > 0).then(pl.col("fg_att")).otherwise(None)).alias("fg_made_rate"),
    ]).with_columns((pl.col("points") - pl.col("points_allowed")).alias("margin"), (pl.col("points") > pl.col("points_allowed")).cast(pl.Float64).alias("won"))
    allowed_cols = [m for m in ALLOWED_METRICS if m in tg.columns]
    opp = tg.select(["game_id", "team"] + allowed_cols).rename({"team": "opponent"} | {m: f"{m}_allowed" for m in allowed_cols})
    tg = tg.join(opp, on=["game_id", "opponent"], how="left")
    return tg.sort(["team", "kickoff_utc", "game_id"])


def _window_exprs(metric: str, windows: dict[str, int], full: bool) -> list[pl.Expr]:
    prior = pl.col(f"__prior_{metric}")  # value from the previous game (already shifted within team)
    present = prior.is_not_null().cast(pl.Float64)
    filled = prior.fill_null(0.0)
    exprs = []
    if full:
        for name, size in windows.items():
            num = filled.rolling_sum(window_size=size, min_samples=1).over("team")
            den = present.rolling_sum(window_size=size, min_samples=1).over("team")
            exprs.append(pl.when(den > 0).then(num / den).otherwise(None).alias(f"{metric}__{name}"))
    exprs.append(prior.ewm_mean(half_life=4.0, ignore_nulls=True, adjust=True).over("team").alias(f"{metric}__ewm"))
    # season-to-date: prior game values within the same season
    num_s = pl.col(f"__prior_s_{metric}").fill_null(0.0).cum_sum().over(["team", "season"])
    den_s = pl.col(f"__prior_s_{metric}").is_not_null().cast(pl.Float64).cum_sum().over(["team", "season"])
    exprs.append(pl.when(den_s > 0).then(num_s / den_s).otherwise(None).alias(f"{metric}__std"))
    return exprs


def rolling_features(tg: pl.DataFrame, core: list[str] | None = None, extended: list[str] | None = None) -> pl.DataFrame:
    """Return (game_id, team) rows with windowed features from strictly prior games."""
    core = [m for m in (core or CORE_METRICS) if m in tg.columns]
    extended = [m for m in (extended or EXTENDED_METRICS) if m in tg.columns]
    allowed = [f"{m}_allowed" for m in ALLOWED_METRICS if f"{m}_allowed" in tg.columns]
    metrics_full = core + [a for a in allowed if a.removesuffix("_allowed") in CORE_METRICS]
    metrics_light = extended + [a for a in allowed if a.removesuffix("_allowed") not in CORE_METRICS]
    all_metrics = metrics_full + metrics_light
    tg = tg.sort(["team", "kickoff_utc", "game_id"])
    # Only completed regular+post season games feed the windows; an unplayed (future) game row
    # carries nulls and therefore never contributes. Its own features still come from the past.
    played = pl.col("completed").fill_null(False)
    shifted = [pl.when(played).then(pl.col(m)).otherwise(None).shift(1).over("team").alias(f"__prior_{m}") for m in all_metrics]
    shifted_season = [pl.when(played).then(pl.col(m)).otherwise(None).shift(1).over(["team", "season"]).alias(f"__prior_s_{m}") for m in all_metrics]
    shifted += [
        pl.when(played).then(pl.col("won")).otherwise(None).shift(1).over("team").alias("__prior_won"),
        played.cast(pl.Float64).shift(1).over("team").fill_null(0.0).alias("__prior_played"),
    ]
    shifted_season += [
        pl.when(played).then(pl.col("won")).otherwise(None).shift(1).over(["team", "season"]).alias("__prior_s_won"),
        played.cast(pl.Float64).shift(1).over(["team", "season"]).fill_null(0.0).alias("__prior_s_played"),
    ]
    tg = tg.with_columns(shifted + shifted_season)
    exprs = []
    for m in metrics_full:
        exprs += _window_exprs(m, WINDOWS, True)
    for m in metrics_light:
        exprs += _window_exprs(m, WINDOWS, False)
    # Experience counters and recent record.
    exprs += [
        pl.col("__prior_s_played").cum_sum().over(["team", "season"]).alias("games_played_std"),
        pl.col("__prior_played").cum_sum().over("team").alias("games_played_total"),
        pl.col("__prior_won").fill_null(0.0).rolling_sum(window_size=5, min_samples=1).over("team").alias("wins__l5"),
        pl.col("__prior_s_won").fill_null(0.0).cum_sum().over(["team", "season"]).alias("wins__std"),
    ]
    out = tg.with_columns(exprs)
    # Previous full season averages.
    season_means = tg.filter(played).group_by(["team", "season"]).agg([pl.col(m).mean().alias(f"{m}__prev") for m in all_metrics] + [pl.col("won").mean().alias("win_pct__prev"), pl.len().alias("games__prev")])
    season_means = season_means.with_columns((pl.col("season") + 1).alias("season")).drop("games__prev")
    out = out.join(season_means, on=["team", "season"], how="left")
    # Variability / trend measures for a few headline metrics.
    for m in ("epa_play", "margin", "points", "epa_play_allowed"):
        if f"__prior_{m}" in out.columns:
            out = out.with_columns([
                pl.col(f"__prior_{m}").rolling_std(window_size=8, min_samples=3).over("team").alias(f"{m}__sd8"),
                (pl.col(f"{m}__l3") - pl.col(f"{m}__l8")).alias(f"{m}__trend"),
            ])
    keep = ["game_id", "team", "season", "week", "kickoff_utc", "games_played_std", "games_played_total"] + [c for c in out.columns if "__" in c and not c.startswith("__")]
    return out.select(keep)
