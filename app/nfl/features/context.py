"""Schedule, rest, travel, stadium, weather, coaching, officiating and market context."""
from __future__ import annotations
import polars as pl

US_ZONES = {"America/New_York", "America/Chicago", "America/Denver", "America/Phoenix", "America/Los_Angeles", "America/Indiana/Indianapolis"}


def game_context(games: pl.DataFrame) -> pl.DataFrame:
    g = games.sort(["kickoff_utc", "game_id"])
    roof = pl.col("roof").fill_null("").str.to_lowercase()
    indoor = roof.is_in(["dome", "closed"])
    surface = pl.col("surface").fill_null("").str.to_lowercase().str.strip_chars()
    ctx = g.select([
        "game_id", "season", "week", "kickoff_utc",
        pl.col("season").cast(pl.Float64).alias("ctx_season"), pl.col("week").cast(pl.Float64).alias("ctx_week"), pl.col("playoff").cast(pl.Float64).alias("ctx_playoff"),
        pl.col("neutral").cast(pl.Float64).alias("ctx_neutral"), pl.col("div_game").cast(pl.Float64).alias("ctx_div_game"), pl.col("conf_game").cast(pl.Float64).alias("ctx_conf_game"),
        indoor.cast(pl.Float64).alias("ctx_indoor"), (roof == "dome").cast(pl.Float64).alias("ctx_dome"), (roof == "open").cast(pl.Float64).alias("ctx_retractable_open"), (roof == "").cast(pl.Float64).alias("ctx_roof_unknown"),
        surface.str.contains("grass").cast(pl.Float64).alias("ctx_grass"), (surface == "").cast(pl.Float64).alias("ctx_surface_unknown"),
        pl.when(indoor).then(70.0).otherwise(pl.col("temp")).alias("wx_temp"), pl.when(indoor).then(0.0).otherwise(pl.col("wind")).alias("wx_wind"),
        (pl.col("temp").is_null() & ~indoor).cast(pl.Float64).alias("wx_missing"),
        pl.col("kickoff_local_hour").alias("ctx_local_hour"), (pl.col("kickoff_local_hour") >= 19).cast(pl.Float64).alias("ctx_primetime"),
        (pl.col("weekday") == "Thursday").cast(pl.Float64).alias("ctx_thursday"), (pl.col("weekday") == "Monday").cast(pl.Float64).alias("ctx_monday"), (pl.col("weekday") == "Saturday").cast(pl.Float64).alias("ctx_saturday"),
        (~pl.col("venue_tz").is_in(list(US_ZONES))).cast(pl.Float64).alias("ctx_international"),
        pl.col("home_rest").cast(pl.Float64).alias("home_rest"), pl.col("away_rest").cast(pl.Float64).alias("away_rest"),
        pl.col("home_travel_miles").alias("home_travel"), pl.col("away_travel_miles").alias("away_travel"),
        (pl.col("venue_utc_offset") - pl.col("home_utc_offset")).alias("home_tz_shift"), (pl.col("venue_utc_offset") - pl.col("away_utc_offset")).alias("away_tz_shift"),
    ])
    ctx = ctx.with_columns([
        (pl.col("home_rest") - pl.col("away_rest")).alias("rest_diff"), (pl.col("home_rest") <= 5).cast(pl.Float64).alias("home_short_week"), (pl.col("away_rest") <= 5).cast(pl.Float64).alias("away_short_week"),
        (pl.col("home_rest") >= 13).cast(pl.Float64).alias("home_bye"), (pl.col("away_rest") >= 13).cast(pl.Float64).alias("away_bye"),
        pl.col("home_travel").log1p().alias("home_log_travel"), pl.col("away_travel").log1p().alias("away_log_travel"),
        # body clock: kickoff hour in the team's home time zone
        (pl.col("ctx_local_hour") - pl.col("home_tz_shift")).alias("home_body_clock_hour"), (pl.col("ctx_local_hour") - pl.col("away_tz_shift")).alias("away_body_clock_hour"),
        pl.when(pl.col("wx_wind").is_not_null()).then((pl.col("wx_wind") - 10).clip(0, None)).otherwise(None).alias("wx_wind_over10"),
        pl.when(pl.col("wx_wind").is_not_null()).then((pl.col("wx_wind") - 15).clip(0, None)).otherwise(None).alias("wx_wind_over15"),
        pl.when(pl.col("wx_temp").is_not_null()).then((40 - pl.col("wx_temp")).clip(0, None)).otherwise(None).alias("wx_cold"),
        pl.when(pl.col("wx_temp").is_not_null()).then((pl.col("wx_temp") <= 32).cast(pl.Float64)).otherwise(None).alias("wx_freezing"),
        pl.when(pl.col("wx_temp").is_not_null()).then((pl.col("wx_temp") - 85).clip(0, None)).otherwise(None).alias("wx_heat"),
    ])
    return ctx


def coaching_features(games: pl.DataFrame) -> pl.DataFrame:
    """Head-coach tenure and change flags from schedule coach names (all seasons)."""
    rows = []
    for side in ("home", "away"):
        rows.append(games.select(["game_id", "kickoff_utc", "season", pl.col(f"{side}_team").alias("team"), pl.col(f"{side}_coach").alias("coach"), pl.col("completed")]))
    f = pl.concat(rows).sort(["team", "kickoff_utc"])
    f = f.with_columns([
        pl.col("coach").shift(1).over("team").alias("prev_coach"),
        pl.col("coach").fill_null("?").alias("coach_f"),
    ])
    f = f.with_columns((pl.col("coach_f") != pl.col("prev_coach").fill_null("?")).cast(pl.Float64).alias("coach_change"))
    # Tenure: games coached for this team before this one (streak of the same coach).
    f = f.with_columns(pl.col("coach_change").cum_sum().over("team").alias("stint"))
    f = f.with_columns([
        pl.int_range(pl.len()).over(["team", "stint"]).cast(pl.Float64).alias("coach_tenure_games"),
        pl.int_range(pl.len()).over(["coach_f"]).cast(pl.Float64).alias("coach_career_games"),
    ])
    first_season = f.group_by(["team", "stint"]).agg(pl.col("season").min().alias("stint_first_season"))
    f = f.join(first_season, on=["team", "stint"], how="left").with_columns((pl.col("season") == pl.col("stint_first_season")).cast(pl.Float64).alias("coach_first_season"))
    return f.select(["game_id", "team", "coach_tenure_games", "coach_career_games", "coach_first_season", pl.col("coach_change").alias("coach_changed_since_last")])


def referee_features(games: pl.DataFrame, team_games: pl.DataFrame) -> pl.DataFrame:
    """Referee crew tendencies from prior games only (penalties and points per game, shrunk)."""
    per_game = team_games.group_by("game_id").agg([pl.col("penalties").sum().alias("g_pen"), pl.col("penalty_yards").sum().alias("g_pen_yds"), pl.col("dpi_drawn").sum().alias("g_dpi"), pl.col("off_holding").sum().alias("g_hold")])
    g = games.select(["game_id", "kickoff_utc", "referee", "total_points", "completed"]).join(per_game, on="game_id", how="left").sort(["kickoff_utc", "game_id"])
    g = g.with_columns(pl.col("referee").fill_null("unknown"))
    played = pl.col("completed").fill_null(False) & pl.col("g_pen").is_not_null()
    prior = [pl.when(played).then(pl.col(c)).otherwise(None).shift(1).over("referee").alias(f"__p_{c}") for c in ("g_pen", "g_pen_yds", "g_dpi", "g_hold", "total_points")]
    g = g.with_columns(prior)
    league = {"g_pen": 12.5, "g_pen_yds": 105.0, "g_dpi": 1.6, "g_hold": 2.2, "total_points": 43.0}
    exprs = []
    for c, mean in league.items():
        num = pl.col(f"__p_{c}").fill_null(0.0).cum_sum().over("referee")
        den = pl.col(f"__p_{c}").is_not_null().cast(pl.Float64).cum_sum().over("referee")
        exprs.append(((num + 20 * mean) / (den + 20)).alias(f"ref_{c}"))
    exprs.append(pl.col("__p_g_pen").is_not_null().cast(pl.Float64).cum_sum().over("referee").alias("ref_games"))
    return g.with_columns(exprs).select(["game_id", "ref_g_pen", "ref_g_pen_yds", "ref_g_dpi", "ref_g_hold", "ref_total_points", "ref_games"])


def no_vig_probability(home_ml: pl.Expr, away_ml: pl.Expr) -> pl.Expr:
    def implied(ml: pl.Expr) -> pl.Expr:
        return pl.when(ml < 0).then(-ml / (-ml + 100)).otherwise(100 / (ml + 100))
    h, a = implied(home_ml), implied(away_ml)
    return h / (h + a)


def market_features(games: pl.DataFrame) -> pl.DataFrame:
    """Closing-line market features (market-aware models only)."""
    m = games.select(["game_id", "spread_line", "total_line", "home_moneyline", "away_moneyline"])
    return m.with_columns([
        pl.col("spread_line").alias("mkt_spread"), pl.col("total_line").alias("mkt_total"),
        ((pl.col("total_line") + pl.col("spread_line")) / 2).alias("mkt_home_implied"), ((pl.col("total_line") - pl.col("spread_line")) / 2).alias("mkt_away_implied"),
        pl.when(pl.col("home_moneyline").is_not_null() & pl.col("away_moneyline").is_not_null()).then(no_vig_probability(pl.col("home_moneyline"), pl.col("away_moneyline"))).otherwise(None).alias("mkt_home_prob"),
    ]).select(["game_id", "mkt_spread", "mkt_total", "mkt_home_implied", "mkt_away_implied", "mkt_home_prob"])
