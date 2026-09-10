"""Player availability expressed as expected value lost, not as a count of injured players.

Sources: official injury reports (2009+, timestamped 2010-2024), weekly roster status
(2002+, reserve/PUP/suspension) and snap counts (2012+) for player value. A player's value is
the snap share carried in recent games; a listed player with no snap history falls back to
depth-chart rank and finally to a small default.

Miss probabilities by report status come from the 2013-2015 development seasons (players on a
roster with snap data): out 99.8% miss, doubtful 99.5%, questionable ~40% overall (68% when
they did not practice, 33% when limited), probable/none ~5%.
"""
from __future__ import annotations
import polars as pl

MISS = {("out", None): 1.0, ("doubtful", None): 0.99, ("questionable", "did not participate in practice"): 0.68, ("questionable", "limited participation in practice"): 0.33,
        ("questionable", "full participation in practice"): 0.25, ("questionable", None): 0.40, ("probable", None): 0.05, (None, "did not participate in practice"): 0.35, (None, "limited participation in practice"): 0.07, (None, None): 0.05}
GROUPS = {"QB": "QB", "RB": "RB", "FB": "RB", "HB": "RB", "WR": "WR", "TE": "TE", "T": "OL", "G": "OL", "C": "OL", "OL": "OL", "OT": "OL", "OG": "OL", "LT": "OL", "RT": "OL", "LG": "OL", "RG": "OL",
          "DE": "DL", "DT": "DL", "NT": "DL", "DL": "DL", "EDGE": "DL", "LB": "LB", "ILB": "LB", "OLB": "LB", "MLB": "LB", "CB": "DB", "S": "DB", "SS": "DB", "FS": "DB", "DB": "DB", "SAF": "DB", "K": "ST", "P": "ST", "LS": "ST", "PK": "ST"}
OFFENSE = ("QB", "RB", "WR", "TE", "OL")
DEFENSE = ("DL", "LB", "DB")


def _group(position: str | None) -> str:
    return GROUPS.get(str(position or "").upper(), "OTHER")


def miss_probability(status: str | None, practice: str | None) -> float:
    status = (status or "").strip().lower() or None
    practice = (practice or "").strip().lower() or None
    if status in ("out", "doubtful", "probable") or (status == "questionable" and practice is None):
        return MISS.get((status, None), 0.4)
    key = (status if status == "questionable" else None, practice if practice in ("did not participate in practice", "limited participation in practice", "full participation in practice") else None)
    return MISS.get(key, MISS.get((status, None), 0.05))


VALUE_SCHEMA = {"gsis_id": pl.String, "value_kickoff": pl.Datetime("us", "UTC"), "value": pl.Float64, "value_team": pl.String, "snap_position": pl.String}


def player_values(snaps: pl.DataFrame, games: pl.DataFrame) -> pl.DataFrame:
    """Post-game snap-share state per player (kickoff-stamped) for as-of lookups."""
    if snaps.height == 0 or "gsis_id" not in snaps.columns:
        return pl.DataFrame(schema=VALUE_SCHEMA)
    s = snaps.filter(pl.col("gsis_id").is_not_null()).join(games.select(["game_id", "kickoff_utc"]), on="game_id", how="inner")
    if s.height == 0:
        return pl.DataFrame(schema=VALUE_SCHEMA)
    s = s.with_columns([
        pl.when(pl.col("position").map_elements(_group, return_dtype=pl.String).is_in(list(OFFENSE))).then(pl.col("offense_pct")).when(pl.col("position").map_elements(_group, return_dtype=pl.String).is_in(list(DEFENSE))).then(pl.col("defense_pct")).otherwise(pl.max_horizontal("offense_pct", "defense_pct", "st_pct")).fill_null(0.0).alias("share"),
    ]).sort(["gsis_id", "kickoff_utc"])
    s = s.with_columns(pl.col("share").rolling_mean(window_size=4, min_samples=1).over("gsis_id").alias("value"))
    return s.select([pl.col("gsis_id"), pl.col("kickoff_utc").alias("value_kickoff"), "value", pl.col("team").alias("value_team"), pl.col("position").alias("snap_position")]).sort("value_kickoff")


def availability_features(games: pl.DataFrame, injuries: pl.DataFrame, snaps: pl.DataFrame, rosters: pl.DataFrame, depth: pl.DataFrame, horizon_hours: float = 1.0) -> pl.DataFrame:
    """One row per (game_id, team) with expected value lost by position group."""
    sides = []
    for side in ("home", "away"):
        sides.append(games.select(["game_id", "season", "week", "game_type", "kickoff_utc", pl.col(f"{side}_team").alias("team")]))
    frame = pl.concat(sides).with_columns((pl.col("kickoff_utc") - pl.duration(hours=horizon_hours)).alias("prediction_ts"))
    values = player_values(snaps, games)

    # ---- injury reports -------------------------------------------------------------
    inj = injuries.filter(pl.col("game_type").fill_null("REG") == "REG") if "game_type" in injuries.columns else injuries
    inj = inj.join(frame.select(["game_id", "season", "week", "team", "kickoff_utc", "prediction_ts"]), on=["season", "week", "team"], how="inner")
    # Known-at: the record timestamp when present, otherwise assume the report was public 48 hours before kickoff.
    inj = inj.with_columns(pl.when(pl.col("date_modified").is_not_null()).then(pl.col("date_modified")).otherwise(pl.col("kickoff_utc") - pl.duration(hours=48)).alias("known_at"))
    inj = inj.filter(pl.col("known_at") <= pl.col("prediction_ts"))
    inj = inj.with_columns([
        pl.struct(["report_status", "practice_status"]).map_elements(lambda s: miss_probability(s["report_status"], s["practice_status"]), return_dtype=pl.Float64).alias("p_miss"),
        pl.col("position").map_elements(_group, return_dtype=pl.String).alias("grp"),
    ]).filter(pl.col("grp") != "OTHER")
    inj = inj.sort("prediction_ts").join_asof(values, left_on="prediction_ts", right_on="value_kickoff", by="gsis_id", strategy="backward", check_sortedness=False)
    # Depth-chart fallback for value when there is no snap history (or before 2012).
    if depth.height:
        weekly = depth.filter(pl.col("as_of").is_null()).select(["season", "week", "team", "gsis_id", pl.col("rank").alias("depth_rank")]).unique(subset=["season", "week", "team", "gsis_id"], keep="first")
        inj = inj.join(weekly, on=["season", "week", "team", "gsis_id"], how="left")
    else:
        inj = inj.with_columns(pl.lit(None, dtype=pl.Int32).alias("depth_rank"))
    inj = inj.with_columns(pl.when(pl.col("value").is_not_null()).then(pl.col("value")).when(pl.col("depth_rank") == 1).then(0.85).when(pl.col("depth_rank") == 2).then(0.35).when(pl.col("depth_rank").is_not_null()).then(0.10).otherwise(0.25).alias("value_used"))
    inj = inj.with_columns((pl.col("p_miss") * pl.col("value_used")).alias("exp_lost"))
    agg = inj.group_by(["game_id", "team"]).agg(
        [pl.col("exp_lost").filter(pl.col("grp") == g).sum().alias(f"inj_lost_{g}") for g in OFFENSE + DEFENSE + ("ST",)] + [
            pl.col("exp_lost").filter(pl.col("grp").is_in(list(OFFENSE))).sum().alias("inj_lost_off"), pl.col("exp_lost").filter(pl.col("grp").is_in(list(DEFENSE))).sum().alias("inj_lost_def"),
            ((pl.col("p_miss") >= 0.9) & (pl.col("value_used") >= 0.5) & pl.col("grp").is_in(list(OFFENSE))).sum().alias("starters_out_off"), ((pl.col("p_miss") >= 0.9) & (pl.col("value_used") >= 0.5) & pl.col("grp").is_in(list(DEFENSE))).sum().alias("starters_out_def"),
            ((pl.col("p_miss") < 0.9) & (pl.col("p_miss") >= 0.2) & (pl.col("value_used") >= 0.5)).sum().alias("starters_questionable"),
            ((pl.col("p_miss") >= 0.9) & (pl.col("grp") == "QB") & (pl.col("value_used") >= 0.5)).sum().alias("qb_starter_listed_out"),
            ((pl.col("p_miss") < 0.9) & (pl.col("p_miss") >= 0.2) & (pl.col("grp") == "QB") & (pl.col("value_used") >= 0.5)).sum().alias("qb_starter_questionable"),
            pl.len().alias("injury_listings"), (pl.col("practice_status") == "did not participate in practice").sum().alias("dnp_count"),
        ])
    frame = frame.join(agg, on=["game_id", "team"], how="left")

    # ---- reserve lists from weekly rosters ---------------------------------------------
    if rosters.height:
        res = rosters.filter(pl.col("status").is_in(["RES", "PUP", "SUS", "NON", "RSN", "RET"]) & (pl.col("game_type").fill_null("REG") == "REG"))
        res = res.join(frame.select(["game_id", "season", "week", "team", "prediction_ts"]), on=["season", "week", "team"], how="inner")
        res = res.with_columns(pl.col("position").map_elements(_group, return_dtype=pl.String).alias("grp")).filter(pl.col("grp") != "OTHER")
        res = res.sort("prediction_ts").join_asof(values, left_on="prediction_ts", right_on="value_kickoff", by="gsis_id", strategy="backward", check_sortedness=False)
        # Only players who carried a real role recently (value >= 0.3) and whose snap state is from this or last season count.
        res = res.filter(pl.col("value").is_not_null() & (pl.col("value") >= 0.3) & ((pl.col("prediction_ts") - pl.col("value_kickoff")).dt.total_days() <= 400))
        ragg = res.group_by(["game_id", "team"]).agg([
            pl.col("value").filter(pl.col("grp").is_in(list(OFFENSE))).sum().alias("reserve_lost_off"), pl.col("value").filter(pl.col("grp").is_in(list(DEFENSE))).sum().alias("reserve_lost_def"),
            pl.col("value").filter(pl.col("grp") == "QB").sum().alias("reserve_lost_QB"), pl.col("value").filter(pl.col("grp") == "OL").sum().alias("reserve_lost_OL"), pl.len().alias("reserve_players"),
        ])
        frame = frame.join(ragg, on=["game_id", "team"], how="left")
    else:
        frame = frame.with_columns([pl.lit(None, dtype=pl.Float64).alias(c) for c in ("reserve_lost_off", "reserve_lost_def", "reserve_lost_QB", "reserve_lost_OL", "reserve_players")])
    # Coverage flags: 0 counts are meaningful only where the source exists.
    inj_seasons = set(injuries["season"].unique().to_list()) if injuries.height else set()
    ros_seasons = set(rosters["season"].unique().to_list()) if rosters.height else set()
    frame = frame.with_columns([
        pl.col("season").is_in(list(inj_seasons)).cast(pl.Float64).alias("injury_source_available"),
        pl.col("season").is_in(list(ros_seasons)).cast(pl.Float64).alias("roster_source_available"),
    ])
    fill_inj = [c for c in frame.columns if c.startswith(("inj_lost", "starters_", "qb_starter_", "injury_listings", "dnp_count"))]
    fill_res = [c for c in frame.columns if c.startswith("reserve_")]
    frame = frame.with_columns([pl.when(pl.col("injury_source_available") == 1).then(pl.col(c).fill_null(0.0)).otherwise(None).alias(c) for c in fill_inj] + [pl.when(pl.col("roster_source_available") == 1).then(pl.col(c).fill_null(0.0)).otherwise(None).alias(c) for c in fill_res])
    frame = frame.with_columns([
        (pl.col("inj_lost_off").fill_null(0.0) + pl.col("reserve_lost_off").fill_null(0.0)).alias("value_lost_off"), (pl.col("inj_lost_def").fill_null(0.0) + pl.col("reserve_lost_def").fill_null(0.0)).alias("value_lost_def"),
    ])
    return frame.drop(["season", "week", "game_type", "kickoff_utc", "prediction_ts"])
