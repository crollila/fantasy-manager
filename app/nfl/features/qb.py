"""Quarterback quality and availability features.

Per-QB running totals are stored as *post-game states* stamped with that game's kickoff; a
game at prediction time T uses the expected starter's latest state with kickoff < T (as-of
join). Shrinkage pulls small samples toward a replacement-level prior so two bad games do
not erase years of evidence, and rookies/backups without history get explicit flags.

Expected starter by horizon:
* ``pregame``: the quarterback listed on the schedule (the starter known once inactives are
  announced ~90 minutes before kickoff; nflverse fills projected starters for future games).
* ``early``: the team's previous game starter (no in-week injury information).
"""
from __future__ import annotations
import polars as pl

PRIOR_EPA = -0.03          # replacement-level EPA per dropback
K_CAREER = 200.0           # dropbacks of prior weight
K_SEASON = 150.0
K_RECENT = 120.0
PRIOR_CPOE = -1.5
PRIOR_SACK = 0.075
PRIOR_INT = 0.03
PRIOR_SUCCESS = 0.44
PRIOR_ANYA = 5.2


def qb_states(qb_games: pl.DataFrame) -> pl.DataFrame:
    """Cumulative post-game state per QB (stamped with the kickoff of the game just played)."""
    q = qb_games.filter(pl.col("kickoff_utc").is_not_null()).sort(["qb_id", "kickoff_utc", "game_id"])
    q = q.with_columns([
        (pl.col("cpoe") * pl.col("attempts")).alias("cpoe_w"), (pl.col("success_rate") * pl.col("dropbacks")).alias("succ_w"),
        pl.col("cpoe").is_not_null().cast(pl.Float64).mul(pl.col("attempts")).alias("cpoe_n"),
        (pl.col("dropbacks") >= 15).cast(pl.Float64).alias("start_like"),
        (pl.col("pass_yards") + 20 * pl.col("pass_td") - 45 * pl.col("interceptions")).alias("anya_num"),
        (pl.col("epa_dropback_neutral") * pl.col("dropbacks")).alias("epa_neutral_w"),
    ])
    sums = ["dropbacks", "attempts", "epa_total", "cpoe_w", "cpoe_n", "succ_w", "sacks", "interceptions", "start_like", "anya_num", "scrambles", "scramble_epa", "qb_hits", "epa_neutral_w"]
    q = q.with_columns([pl.col(c).fill_null(0.0).cum_sum().over("qb_id").alias(f"car_{c}") for c in sums])
    q = q.with_columns([pl.col(c).fill_null(0.0).cum_sum().over(["qb_id", "season"]).alias(f"sea_{c}") for c in ("dropbacks", "epa_total", "cpoe_w", "cpoe_n", "succ_w", "sacks", "interceptions", "attempts", "start_like")])
    q = q.with_columns([pl.col(c).fill_null(0.0).rolling_sum(window_size=4, min_samples=1).over("qb_id").alias(f"l4_{c}") for c in ("dropbacks", "epa_total", "cpoe_w", "cpoe_n", "sacks", "attempts")])
    q = q.with_columns([
        pl.col("epa_dropback").ewm_mean(half_life=6.0, ignore_nulls=True, adjust=True).over("qb_id").alias("ewm_epa_dropback"),
        pl.len().over("qb_id").alias("_n"), pl.int_range(pl.len()).over("qb_id").add(1).alias("games_seen"),
        pl.col("team").alias("state_team"), pl.col("game_id").alias("state_game"),
    ])
    prev = q.group_by(["qb_id", "season"]).agg([pl.col("dropbacks").sum().alias("prev_dropbacks"), pl.col("epa_total").sum().alias("prev_epa"), pl.col("cpoe_w").sum().alias("prev_cpoe_w"), pl.col("cpoe_n").sum().alias("prev_cpoe_n")]).with_columns((pl.col("season") + 1).alias("season"))
    q = q.join(prev, on=["qb_id", "season"], how="left")
    shrunk = lambda num, n, k, prior: (pl.col(num) + k * prior) / (pl.col(n) + k)  # noqa: E731
    state = q.select([
        pl.col("qb_id"), pl.col("kickoff_utc").alias("state_kickoff"), pl.col("season").alias("state_season"), "state_team", "state_game", "games_seen",
        shrunk("car_epa_total", "car_dropbacks", K_CAREER, PRIOR_EPA).alias("qb_epa_career"),
        shrunk("sea_epa_total", "sea_dropbacks", K_SEASON, PRIOR_EPA).alias("qb_epa_season_raw"),
        shrunk("l4_epa_total", "l4_dropbacks", K_RECENT, PRIOR_EPA).alias("qb_epa_last4_raw"),
        pl.when(pl.col("prev_dropbacks").is_not_null()).then((pl.col("prev_epa") + K_SEASON * PRIOR_EPA) / (pl.col("prev_dropbacks") + K_SEASON)).otherwise(None).alias("qb_epa_prev_season"),
        pl.col("ewm_epa_dropback").alias("qb_epa_ewm"),
        ((pl.col("car_cpoe_w") + 100 * PRIOR_CPOE) / (pl.col("car_cpoe_n") + 100)).alias("qb_cpoe_career"),
        ((pl.col("sea_cpoe_w") + 100 * PRIOR_CPOE) / (pl.col("sea_cpoe_n") + 100)).alias("qb_cpoe_season"),
        ((pl.col("car_sacks") + 100 * PRIOR_SACK) / (pl.col("car_dropbacks") + 100)).alias("qb_sack_rate_career"),
        ((pl.col("car_interceptions") + 100 * PRIOR_INT) / (pl.col("car_attempts") + 100)).alias("qb_int_rate_career"),
        ((pl.col("car_succ_w") + 100 * PRIOR_SUCCESS) / (pl.col("car_dropbacks") + 100)).alias("qb_success_career"),
        ((pl.col("car_anya_num") + 100 * PRIOR_ANYA) / (pl.col("car_dropbacks") + 100)).alias("qb_anya_career"),
        ((pl.col("car_epa_neutral_w") + K_CAREER * PRIOR_EPA) / (pl.col("car_dropbacks") + K_CAREER)).alias("qb_epa_neutral_career"),
        (pl.col("car_scrambles") / pl.col("car_dropbacks").clip(1, None)).alias("qb_scramble_rate_career"),
        (pl.col("car_qb_hits") / pl.col("car_dropbacks").clip(1, None)).alias("qb_hit_rate_career"),
        pl.col("car_dropbacks").alias("qb_dropbacks_career"), pl.col("sea_dropbacks").alias("qb_dropbacks_season"), pl.col("car_start_like").alias("qb_starts_career"), pl.col("sea_start_like").alias("qb_starts_season"),
    ])
    # Blend season/recent toward career so early-season values do not overreact.
    state = state.with_columns([
        ((pl.col("qb_epa_season_raw") * pl.col("qb_dropbacks_season") + pl.col("qb_epa_career") * K_SEASON) / (pl.col("qb_dropbacks_season") + K_SEASON)).alias("qb_epa_season"),
        pl.col("qb_epa_last4_raw").alias("qb_epa_last4"),
    ]).drop(["qb_epa_season_raw", "qb_epa_last4_raw"])
    return state.sort(["qb_id", "state_kickoff"])


def team_starters(qb_games: pl.DataFrame) -> pl.DataFrame:
    """Starter (most dropbacks) per (game_id, team) and the previous starter for that team."""
    s = qb_games.sort(["game_id", "team", "dropbacks"], descending=[False, False, True]).unique(subset=["game_id", "team"], keep="first")
    s = s.select(["game_id", "team", "season", "kickoff_utc", pl.col("qb_id").alias("starter_id"), pl.col("dropbacks").alias("starter_dropbacks")]).sort(["team", "kickoff_utc"])
    s = s.with_columns([pl.col("starter_id").shift(1).over("team").alias("prev_starter_id"), pl.col("kickoff_utc").shift(1).over("team").alias("prev_kickoff")])
    return s


def qb_features(games: pl.DataFrame, qb_games: pl.DataFrame, players: pl.DataFrame, horizon: str = "pregame", horizon_hours: float = 1.0) -> pl.DataFrame:
    """One row per (game_id, team) with expected-starter features at the horizon."""
    states = qb_states(qb_games)
    starters = team_starters(qb_games)
    sides = []
    for side, other in (("home", "away"), ("away", "home")):
        sides.append(games.select([
            "game_id", "season", "week", "kickoff_utc", pl.col(f"{side}_team").alias("team"), pl.col(f"{side}_qb_id").alias("schedule_qb_id"),
        ]))
    frame = pl.concat(sides)
    frame = frame.join(starters.select(["game_id", "team", "starter_id", "prev_starter_id"]), on=["game_id", "team"], how="left")
    # Previous starter as of this game for teams whose row has no qb_games entry (future games):
    last_known = starters.sort(["team", "kickoff_utc"]).select(["team", pl.col("kickoff_utc").alias("prev_kickoff"), pl.col("starter_id").alias("last_starter_id")])
    frame = frame.sort("kickoff_utc")
    frame = frame.with_columns((pl.col("kickoff_utc") - pl.duration(hours=horizon_hours)).alias("prediction_ts"))
    frame = frame.join_asof(last_known.sort("prev_kickoff"), left_on="prediction_ts", right_on="prev_kickoff", by="team", strategy="backward", check_sortedness=False)
    if horizon == "pregame":
        expected = pl.when(pl.col("schedule_qb_id").is_not_null()).then(pl.col("schedule_qb_id")).otherwise(pl.col("last_starter_id"))
    else:
        expected = pl.col("last_starter_id")
    frame = frame.with_columns(expected.alias("qb_id"))
    frame = frame.sort("prediction_ts")
    frame = frame.join_asof(states.sort("state_kickoff"), left_on="prediction_ts", right_on="state_kickoff", by="qb_id", strategy="backward", check_sortedness=False)
    info = players.select([pl.col("gsis_id").alias("qb_id"), pl.col("birth_date").alias("qb_birth"), pl.col("draft_round").alias("qb_draft_round"), pl.col("draft_pick").alias("qb_draft_pick"), pl.col("rookie_season").alias("qb_rookie_season")])
    frame = frame.join(info, on="qb_id", how="left")
    frame = frame.with_columns([
        pl.col("qb_id").is_null().cast(pl.Float64).alias("qb_unknown"),
        (pl.col("qb_dropbacks_career").fill_null(0.0) == 0).cast(pl.Float64).alias("qb_no_history"),
        (pl.col("qb_id") != pl.col("last_starter_id")).cast(pl.Float64).fill_null(1.0).alias("qb_changed"),
        (pl.col("state_team") != pl.col("team")).cast(pl.Float64).fill_null(1.0).alias("qb_new_to_team"),
        pl.col("qb_dropbacks_career").fill_null(0.0).log1p().alias("qb_log_dropbacks"),
        pl.col("qb_draft_round").cast(pl.Float64).fill_null(8.0).alias("qb_draft_round"), pl.col("qb_draft_pick").cast(pl.Float64).fill_null(260.0).alias("qb_draft_pick"),
        ((pl.col("kickoff_utc").dt.year() - pl.col("qb_birth").str.slice(0, 4).cast(pl.Int32, strict=False)).cast(pl.Float64)).alias("qb_age"),
        (pl.col("season") - pl.col("qb_rookie_season").cast(pl.Int32)).cast(pl.Float64).alias("qb_seasons_exp"),
    ])
    # Unknown/no-history quarterbacks: replacement priors instead of nulls for the headline metric.
    frame = frame.with_columns([
        pl.col("qb_epa_career").fill_null(PRIOR_EPA), pl.col("qb_epa_season").fill_null(pl.col("qb_epa_career")), pl.col("qb_epa_last4").fill_null(pl.col("qb_epa_career")), pl.col("qb_epa_ewm").fill_null(pl.col("qb_epa_career")),
        pl.col("qb_epa_prev_season").fill_null(pl.col("qb_epa_career")), pl.col("qb_cpoe_career").fill_null(PRIOR_CPOE), pl.col("qb_sack_rate_career").fill_null(PRIOR_SACK), pl.col("qb_int_rate_career").fill_null(PRIOR_INT),
        pl.col("qb_success_career").fill_null(PRIOR_SUCCESS), pl.col("qb_anya_career").fill_null(PRIOR_ANYA), pl.col("qb_epa_neutral_career").fill_null(PRIOR_EPA),
        pl.col("qb_dropbacks_career").fill_null(0.0), pl.col("qb_dropbacks_season").fill_null(0.0), pl.col("qb_starts_career").fill_null(0.0), pl.col("qb_starts_season").fill_null(0.0), pl.col("qb_cpoe_season").fill_null(pl.col("qb_cpoe_career")),
    ])
    cols = ["game_id", "team", "qb_id", "qb_epa_career", "qb_epa_season", "qb_epa_last4", "qb_epa_ewm", "qb_epa_prev_season", "qb_cpoe_career", "qb_cpoe_season", "qb_sack_rate_career", "qb_int_rate_career", "qb_success_career",
            "qb_anya_career", "qb_epa_neutral_career", "qb_scramble_rate_career", "qb_hit_rate_career", "qb_dropbacks_career", "qb_dropbacks_season", "qb_starts_career", "qb_starts_season", "qb_unknown", "qb_no_history", "qb_changed", "qb_new_to_team",
            "qb_log_dropbacks", "qb_draft_round", "qb_draft_pick", "qb_age", "qb_seasons_exp"]
    return frame.select(cols)
