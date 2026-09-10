"""Normalized point-in-time data model built from the immutable raw store.

Outputs (Parquet under ``normalized/``):

* ``games``            one row per game with UTC kickoff, canonical franchise codes, scores,
                       context (roof, surface, rest, coaches, referee, venue) and raw market lines.
* ``plays/<season>``   trimmed play-by-play with canonical codes and derived flags.
* ``team_games``       one row per (game, team) of what that team's OFFENSE and SPECIAL TEAMS did.
                       Defensive "allowed" numbers are the opponent's row of the same game.
* ``qb_games``         one row per (game, passer) with dropback-level production.
* ``kicker_games``     one row per (game, kicker) with field-goal attempts by distance.
* ``injuries``, ``snaps``, ``depth_charts``, ``rosters``, ``players`` with stable GSIS ids.

Nothing here looks across games; point-in-time windows are assembled in the feature layer.
"""
from __future__ import annotations
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo
import polars as pl
from app.nfl.config import paths, FIRST_SEASON
from app.nfl.ingest import RawStore, utcnow
from app.nfl.reference.teams import canonical, venue_for_game, home_venue, haversine_miles, utc_offset_hours, DIVISIONS

log = logging.getLogger(__name__)
ET = ZoneInfo("America/New_York")

PLAY_COLUMNS = [
    "game_id", "play_id", "season", "week", "season_type", "posteam", "defteam", "home_team", "away_team", "qtr", "down", "ydstogo", "yardline_100", "goal_to_go",
    "game_seconds_remaining", "half_seconds_remaining", "score_differential", "posteam_score", "defteam_score", "posteam_score_post", "wp", "play_type", "pass", "rush", "qb_dropback", "qb_scramble", "qb_kneel", "qb_spike",
    "shotgun", "no_huddle", "epa", "qb_epa", "success", "yards_gained", "air_yards", "yards_after_catch", "cpoe", "xpass", "pass_oe", "pass_length", "complete_pass", "incomplete_pass", "interception", "sack", "qb_hit",
    "fumble", "fumble_lost", "fumble_forced", "touchdown", "pass_touchdown", "rush_touchdown", "first_down", "third_down_converted", "third_down_failed", "fourth_down_converted", "fourth_down_failed",
    "penalty", "penalty_team", "penalty_yards", "penalty_type", "field_goal_attempt", "field_goal_result", "kick_distance", "punt_attempt", "kickoff_attempt", "extra_point_attempt", "extra_point_result",
    "two_point_attempt", "two_point_conv_result", "special_teams_play", "return_yards", "fixed_drive", "fixed_drive_result", "series", "series_success", "passer_player_id", "rusher_player_id", "receiver_player_id",
    "kicker_player_id", "punter_player_id", "aborted_play", "play_deleted", "tackled_for_loss", "timeout", "xyac_epa", "air_epa", "yac_epa", "td_team", "run_location", "run_gap", "drive_inside20", "tackle_with_assist",
]


def _read(path: Path, columns: list[str] | None = None) -> pl.DataFrame:
    frame = pl.read_parquet(path)
    if columns:
        missing = [c for c in columns if c not in frame.columns]
        frame = frame.select([c for c in columns if c in frame.columns])
        for c in missing:
            frame = frame.with_columns(pl.lit(None, dtype=pl.Float64).alias(c))
        frame = frame.select(columns)
    return frame


def kickoff_utc(gameday: str | None, gametime: str | None) -> tuple[datetime | None, bool]:
    """Kickoff in UTC from the ET schedule fields. Missing times default to 13:00 ET (flagged)."""
    if not gameday:
        return None, False
    imputed = not gametime or str(gametime) in ("None", "nan", "")
    time = "13:00" if imputed else str(gametime)
    try:
        local = datetime.fromisoformat(f"{gameday}T{time}").replace(tzinfo=ET)
    except ValueError:
        return None, False
    return local.astimezone(timezone.utc), imputed


def build_games(raw: RawStore, out: Path) -> pl.DataFrame:
    schedule = pl.read_parquet(raw.path("schedules", "games.parquet"))
    rows = []
    for r in schedule.to_dicts():
        home, away = canonical(r["home_team"]), canonical(r["away_team"])
        ko, imputed = kickoff_utc(r.get("gameday"), r.get("gametime"))
        season = int(r["season"])
        neutral = str(r.get("location") or "Home").lower() == "neutral"
        venue = venue_for_game(home, season, r.get("stadium"), neutral)
        home_home = home_venue(home, season)
        away_home = home_venue(away, season)
        completed = r.get("home_score") is not None and r.get("away_score") is not None
        row = {
            "game_id": r["game_id"], "season": season, "week": int(r["week"]), "game_type": r["game_type"], "playoff": r["game_type"] != "REG",
            "gameday": r.get("gameday"), "weekday": r.get("weekday"), "gametime_et": r.get("gametime"), "kickoff_utc": ko, "kickoff_imputed": imputed,
            "home_team": home, "away_team": away, "home_team_raw": r["home_team"], "away_team_raw": r["away_team"],
            "home_score": r.get("home_score"), "away_score": r.get("away_score"), "completed": completed,
            "margin": (r["home_score"] - r["away_score"]) if completed else None, "total_points": (r["home_score"] + r["away_score"]) if completed else None,
            "overtime": r.get("overtime"), "neutral": neutral, "div_game": bool(r.get("div_game")), "conf_game": DIVISIONS.get(home, "?")[:3] == DIVISIONS.get(away, "?")[:3],
            "roof": (r.get("roof") or "").strip() or None, "surface": (r.get("surface") or "").strip() or None, "temp": r.get("temp"), "wind": r.get("wind"),
            "home_rest": r.get("home_rest"), "away_rest": r.get("away_rest"), "home_qb_id": r.get("home_qb_id"), "away_qb_id": r.get("away_qb_id"), "home_qb_name": r.get("home_qb_name"), "away_qb_name": r.get("away_qb_name"),
            "home_coach": r.get("home_coach"), "away_coach": r.get("away_coach"), "referee": r.get("referee"), "stadium": r.get("stadium"), "stadium_id": r.get("stadium_id"),
            "venue_lat": venue["lat"], "venue_lon": venue["lon"], "venue_tz": venue["tz"], "venue_match": venue.get("matched"),
            "home_travel_miles": haversine_miles(home_home["lat"], home_home["lon"], venue["lat"], venue["lon"]) if neutral else 0.0,
            "away_travel_miles": haversine_miles(away_home["lat"], away_home["lon"], venue["lat"], venue["lon"]),
            "home_tz": home_home["tz"], "away_tz": away_home["tz"],
            "spread_line": r.get("spread_line"), "total_line": r.get("total_line"), "home_moneyline": r.get("home_moneyline"), "away_moneyline": r.get("away_moneyline"),
            "home_spread_odds": r.get("home_spread_odds"), "away_spread_odds": r.get("away_spread_odds"), "over_odds": r.get("over_odds"), "under_odds": r.get("under_odds"),
            "espn_id": str(r.get("espn")).removesuffix(".0") if r.get("espn") is not None else None, "old_game_id": str(r.get("old_game_id")) if r.get("old_game_id") is not None else None, "pfr_id": r.get("pfr"),
        }
        if ko is not None:
            row["venue_utc_offset"] = utc_offset_hours(venue["tz"], ko)
            row["home_utc_offset"] = utc_offset_hours(home_home["tz"], ko)
            row["away_utc_offset"] = utc_offset_hours(away_home["tz"], ko)
            row["kickoff_local_hour"] = (ko.astimezone(ZoneInfo(venue["tz"])).hour + ko.astimezone(ZoneInfo(venue["tz"])).minute / 60) if venue["tz"] else None
        rows.append(row)
    games = pl.DataFrame(rows, infer_schema_length=None).sort(["kickoff_utc", "game_id"])
    games = games.with_columns(pl.col("kickoff_utc").cast(pl.Datetime("us", "UTC")))
    games.write_parquet(out / "games.parquet")
    return games


def build_plays(raw: RawStore, out: Path, seasons: list[int] | None = None) -> list[int]:
    folder = out / "plays"
    folder.mkdir(parents=True, exist_ok=True)
    built = []
    for path in raw.files("pbp"):
        season = int(path.stem.split("_")[-1])
        if seasons and season not in seasons:
            continue
        target = folder / f"{season}.parquet"
        if target.exists() and target.stat().st_mtime >= path.stat().st_mtime:
            built.append(season)
            continue
        frame = _read(path, PLAY_COLUMNS)
        frame = frame.with_columns([
            pl.col("posteam").map_elements(canonical, return_dtype=pl.String).alias("posteam"),
            pl.col("defteam").map_elements(canonical, return_dtype=pl.String).alias("defteam"),
            pl.col("home_team").map_elements(canonical, return_dtype=pl.String).alias("home_team"),
            pl.col("away_team").map_elements(canonical, return_dtype=pl.String).alias("away_team"),
            pl.col("penalty_team").map_elements(canonical, return_dtype=pl.String).alias("penalty_team"),
            pl.col("td_team").map_elements(canonical, return_dtype=pl.String).alias("td_team"),
        ])
        frame = frame.filter(pl.col("play_deleted").fill_null(0) != 1)
        frame.write_parquet(target)
        built.append(season)
        log.info("plays %s: %s rows", season, frame.height)
    return sorted(built)


def _flag(col: str) -> pl.Expr:
    return pl.col(col).cast(pl.Float64).fill_null(0) == 1


def team_game_frame(plays: pl.DataFrame) -> pl.DataFrame:
    """Offense + special teams + penalty production per (game_id, team)."""
    p = plays.filter(pl.col("posteam").is_not_null() & (pl.col("posteam") != ""))
    scrim = (_flag("pass") | _flag("rush")) & pl.col("epa").is_not_null() & ~_flag("qb_kneel") & ~_flag("qb_spike")
    is_pass = scrim & _flag("pass")
    is_rush = scrim & _flag("rush")
    wp = pl.col("wp")
    neutral = scrim & wp.is_between(0.10, 0.90)
    early = scrim & pl.col("down").is_in([1, 2])
    late = scrim & pl.col("down").is_in([3, 4])
    third = scrim & (pl.col("down") == 3)
    fourth = scrim & (pl.col("down") == 4)
    rz = scrim & (pl.col("yardline_100") <= 20)
    gl = scrim & (pl.col("yardline_100") <= 5)
    leading = scrim & (pl.col("score_differential") > 0)
    trailing = scrim & (pl.col("score_differential") < 0)
    tied = scrim & (pl.col("score_differential") == 0)
    two_min = scrim & (pl.col("half_seconds_remaining") <= 120)
    short = scrim & (pl.col("ydstogo") <= 3)
    long_ = scrim & (pl.col("ydstogo") >= 8)
    close_state = scrim & (pl.col("score_differential").abs() <= 7) & (pl.col("game_seconds_remaining") > 120)
    yards = pl.col("yards_gained").cast(pl.Float64)
    epa = pl.col("epa")
    success = pl.col("success").cast(pl.Float64)
    deep = is_pass & (pl.col("air_yards") >= 20)
    st = _flag("special_teams_play") | _flag("field_goal_attempt") | _flag("punt_attempt") | _flag("kickoff_attempt") | _flag("extra_point_attempt")
    fg = _flag("field_goal_attempt")
    fg_made = fg & (pl.col("field_goal_result") == "made")
    dist = pl.col("kick_distance").cast(pl.Float64)
    pen = _flag("penalty") & (pl.col("penalty_team") == pl.col("posteam"))
    pen_def = _flag("penalty") & (pl.col("penalty_team") == pl.col("defteam"))

    def mean(expr: pl.Expr, cond: pl.Expr) -> pl.Expr:
        return expr.filter(cond).mean()

    def rate(cond_num: pl.Expr, cond_den: pl.Expr) -> pl.Expr:
        return cond_num.cast(pl.Float64).sum() / pl.when(cond_den.cast(pl.Float64).sum() > 0).then(cond_den.cast(pl.Float64).sum()).otherwise(None)

    aggs = {
        "plays": scrim.sum(), "dropbacks": is_pass.sum(), "rushes": is_rush.sum(), "neutral_plays": neutral.sum(),
        "epa_play": mean(epa, scrim), "epa_pass": mean(epa, is_pass), "epa_rush": mean(epa, is_rush), "epa_early": mean(epa, early), "epa_late": mean(epa, late),
        "epa_third": mean(epa, third), "epa_rz": mean(epa, rz), "epa_gl": mean(epa, gl), "epa_neutral": mean(epa, neutral), "epa_pass_neutral": mean(epa, neutral & is_pass), "epa_rush_neutral": mean(epa, neutral & is_rush),
        "epa_leading": mean(epa, leading), "epa_trailing": mean(epa, trailing), "epa_tied": mean(epa, tied), "epa_two_min": mean(epa, two_min), "epa_short": mean(epa, short), "epa_long": mean(epa, long_),
        "epa_total": epa.filter(scrim).sum(), "epa_pass_total": epa.filter(is_pass).sum(), "epa_rush_total": epa.filter(is_rush).sum(),
        "success_rate": mean(success, scrim), "success_pass": mean(success, is_pass), "success_rush": mean(success, is_rush), "success_early": mean(success, early), "success_late": mean(success, late), "success_neutral": mean(success, neutral),
        "success_rz": mean(success, rz), "success_third": mean(success, third),
        "third_conv": rate(_flag("third_down_converted"), _flag("third_down_converted") | _flag("third_down_failed")), "fourth_conv": rate(_flag("fourth_down_converted"), _flag("fourth_down_converted") | _flag("fourth_down_failed")),
        "fourth_attempts": (_flag("fourth_down_converted") | _flag("fourth_down_failed")).sum(),
        "ypp": mean(yards, scrim), "ypa": mean(yards, is_pass), "ypc": mean(yards, is_rush), "air_yards_att": mean(pl.col("air_yards"), is_pass & pl.col("air_yards").is_not_null()), "yac_comp": mean(pl.col("yards_after_catch"), is_pass & _flag("complete_pass")),
        "explosive10": rate(scrim & (yards >= 10), scrim), "explosive20": rate(scrim & (yards >= 20), scrim), "explosive40": rate(scrim & (yards >= 40), scrim), "explosive_pass": rate(is_pass & (yards >= 20), is_pass), "explosive_rush": rate(is_rush & (yards >= 10), is_rush),
        "deep_rate": rate(deep, is_pass & pl.col("air_yards").is_not_null()), "epa_deep": mean(epa, deep), "short_pass_epa": mean(epa, is_pass & (pl.col("air_yards") < 10)),
        "pass_rate": rate(is_pass, scrim), "pass_rate_neutral": rate(neutral & is_pass, neutral), "pass_rate_close": rate(close_state & is_pass, close_state), "pass_rate_early": rate(early & is_pass, early),
        "proe": mean(pl.col("pass_oe"), scrim & pl.col("pass_oe").is_not_null()), "xpass_mean": mean(pl.col("xpass"), scrim & pl.col("xpass").is_not_null()),
        "cpoe": mean(pl.col("cpoe"), is_pass & pl.col("cpoe").is_not_null()), "comp_pct": rate(is_pass & _flag("complete_pass"), is_pass & (_flag("complete_pass") | _flag("incomplete_pass") | _flag("interception"))),
        "sacks_taken": (is_pass & _flag("sack")).sum(), "sack_rate": rate(is_pass & _flag("sack"), is_pass), "qb_hits_taken": (is_pass & _flag("qb_hit")).sum(), "qb_hit_rate": rate(is_pass & _flag("qb_hit"), is_pass),
        "scrambles": (is_pass & _flag("qb_scramble")).sum(), "scramble_rate": rate(is_pass & _flag("qb_scramble"), is_pass),
        "interceptions": (scrim & _flag("interception")).sum(), "fumbles_lost": (scrim & _flag("fumble_lost")).sum(), "fumbles": (scrim & _flag("fumble")).sum(), "giveaways": (scrim & (_flag("interception") | _flag("fumble_lost"))).sum(),
        "int_rate": rate(is_pass & _flag("interception"), is_pass), "fumble_rate": rate(scrim & _flag("fumble"), scrim), "fumble_lost_rate": rate(scrim & _flag("fumble_lost"), scrim),
        "td_pass": (is_pass & _flag("pass_touchdown")).sum(), "td_rush": (is_rush & _flag("rush_touchdown")).sum(), "first_downs": (scrim & _flag("first_down")).sum(), "first_down_rate": rate(scrim & _flag("first_down"), scrim),
        "tfl_rate": rate(is_rush & _flag("tackled_for_loss"), is_rush), "stuff_rate": rate(is_rush & (yards <= 0), is_rush), "shotgun_rate": rate(scrim & _flag("shotgun"), scrim), "no_huddle_rate": rate(scrim & _flag("no_huddle"), scrim),
        "xyac_epa_mean": mean(pl.col("xyac_epa"), is_pass & pl.col("xyac_epa").is_not_null()), "yac_epa_mean": mean(pl.col("yac_epa"), is_pass & _flag("complete_pass") & pl.col("yac_epa").is_not_null()), "air_epa_mean": mean(pl.col("air_epa"), is_pass & pl.col("air_epa").is_not_null()),
        "penalties": pen.sum(), "penalty_yards": pl.col("penalty_yards").filter(pen).sum(), "false_starts": (pen & (pl.col("penalty_type") == "False Start")).sum(), "off_holding": (pen & (pl.col("penalty_type") == "Offensive Holding")).sum(),
        "penalties_drawn": pen_def.sum(), "penalty_yards_drawn": pl.col("penalty_yards").filter(pen_def).sum(), "dpi_drawn": (pen_def & (pl.col("penalty_type") == "Defensive Pass Interference")).sum(), "def_holding_drawn": (pen_def & (pl.col("penalty_type") == "Defensive Holding")).sum(),
        "fg_att": fg.sum(), "fg_made": fg_made.sum(), "fg_att_0_39": (fg & (dist < 40)).sum(), "fg_made_0_39": (fg_made & (dist < 40)).sum(), "fg_att_40_49": (fg & dist.is_between(40, 49.99)).sum(), "fg_made_40_49": (fg_made & dist.is_between(40, 49.99)).sum(),
        "fg_att_50": (fg & (dist >= 50)).sum(), "fg_made_50": (fg_made & (dist >= 50)).sum(), "fg_distance_mean": mean(dist, fg), "xp_att": _flag("extra_point_attempt").sum(), "xp_made": (_flag("extra_point_attempt") & (pl.col("extra_point_result") == "good")).sum(),
        "punts": _flag("punt_attempt").sum(), "punt_distance": mean(dist, _flag("punt_attempt")), "st_epa": epa.filter(st & epa.is_not_null()).sum(), "fg_epa": epa.filter(fg & epa.is_not_null()).sum(), "punt_epa": epa.filter(_flag("punt_attempt") & epa.is_not_null()).sum(),
        "two_pt_att": _flag("two_point_attempt").sum(),
        "wp_mean": mean(wp, scrim), "garbage_share": rate(scrim & ~wp.is_between(0.10, 0.90), scrim),
    }
    base = p.group_by(["game_id", "posteam"]).agg([expr.alias(name) for name, expr in aggs.items()]).rename({"posteam": "team"})

    # Drive-level summaries (offense drives).
    d = p.filter(pl.col("fixed_drive").is_not_null())
    drive = d.group_by(["game_id", "posteam", "fixed_drive"]).agg([
        scrim.sum().alias("d_plays"), pl.col("fixed_drive_result").drop_nulls().first().alias("d_result"),
        (pl.col("posteam_score_post").max() - pl.col("posteam_score").min()).alias("d_points_raw"),
        yards.filter(scrim).sum().alias("d_yards"), (scrim & _flag("first_down")).sum().alias("d_first_downs"),
        (pl.col("yardline_100").filter(scrim).min() <= 20).alias("d_rz"), epa.filter(scrim).sum().alias("d_epa"),
        pl.col("game_seconds_remaining").filter(scrim).max().alias("d_start"), pl.col("game_seconds_remaining").filter(scrim).min().alias("d_end"),
        pl.col("wp").filter(scrim).first().alias("d_wp_start"),
    ]).filter(pl.col("d_plays") > 0)
    drive = drive.with_columns([
        pl.when(pl.col("d_result") == "Touchdown").then(pl.col("d_points_raw").clip(6, 8)).when(pl.col("d_result") == "Field goal").then(3.0).otherwise(0.0).alias("d_points"),
        (pl.col("d_result") == "Touchdown").cast(pl.Float64).alias("d_td"), (pl.col("d_result") == "Field goal").cast(pl.Float64).alias("d_fg"), (pl.col("d_result") == "Punt").cast(pl.Float64).alias("d_punt"),
        pl.col("d_result").is_in(["Turnover", "Opp touchdown"]).cast(pl.Float64).alias("d_turnover"), (pl.col("d_result") == "Turnover on downs").cast(pl.Float64).alias("d_downs"),
        ((pl.col("d_result") == "Punt") & (pl.col("d_first_downs") == 0)).cast(pl.Float64).alias("d_three_out"), (pl.col("d_start") - pl.col("d_end")).alias("d_seconds"),
    ])
    drives = drive.group_by(["game_id", "posteam"]).agg([
        pl.len().alias("drives"), pl.col("d_points").mean().alias("points_drive"), pl.col("d_points").sum().alias("drive_points"), pl.col("d_epa").mean().alias("epa_drive"), pl.col("d_yards").mean().alias("yards_drive"),
        pl.col("d_plays").mean().alias("plays_drive"), pl.col("d_td").mean().alias("td_drive"), pl.col("d_fg").mean().alias("fg_drive"), pl.col("d_punt").mean().alias("punt_drive"), pl.col("d_turnover").mean().alias("turnover_drive"),
        pl.col("d_downs").mean().alias("downs_drive"), pl.col("d_three_out").mean().alias("three_out_rate"), pl.col("d_rz").cast(pl.Float64).mean().alias("rz_trip_rate"),
        (pl.col("d_td").filter(pl.col("d_rz")).sum() / pl.when(pl.col("d_rz").sum() > 0).then(pl.col("d_rz").sum()).otherwise(None)).alias("rz_td_rate"),
        ((pl.col("d_seconds").filter((pl.col("d_plays") >= 3) & pl.col("d_wp_start").is_between(0.15, 0.85)).sum()) / pl.when((pl.col("d_plays").filter((pl.col("d_plays") >= 3) & pl.col("d_wp_start").is_between(0.15, 0.85)) - 1).sum() > 0).then((pl.col("d_plays").filter((pl.col("d_plays") >= 3) & pl.col("d_wp_start").is_between(0.15, 0.85)) - 1).sum()).otherwise(None)).alias("sec_per_play_neutral"),
        ((pl.col("d_seconds").filter(pl.col("d_plays") >= 3).sum()) / pl.when((pl.col("d_plays").filter(pl.col("d_plays") >= 3) - 1).sum() > 0).then((pl.col("d_plays").filter(pl.col("d_plays") >= 3) - 1).sum()).otherwise(None)).alias("sec_per_play"),
    ]).rename({"posteam": "team"})
    frame = base.join(drives, on=["game_id", "team"], how="left")
    meta = p.group_by("game_id").agg([pl.col("season").first(), pl.col("week").first(), pl.col("season_type").first(), pl.col("home_team").first(), pl.col("away_team").first()])
    frame = frame.join(meta, on="game_id", how="left")
    frame = frame.with_columns([
        pl.when(pl.col("team") == pl.col("home_team")).then(pl.col("away_team")).otherwise(pl.col("home_team")).alias("opponent"),
        (pl.col("team") == pl.col("home_team")).alias("is_home"),
    ])
    return frame


def qb_game_frame(plays: pl.DataFrame) -> pl.DataFrame:
    p = plays.filter(pl.col("posteam").is_not_null() & _flag("qb_dropback") & ~_flag("qb_kneel") & ~_flag("qb_spike") & pl.col("epa").is_not_null())
    p = p.with_columns(pl.when(pl.col("passer_player_id").is_not_null()).then(pl.col("passer_player_id")).when(_flag("qb_scramble")).then(pl.col("rusher_player_id")).otherwise(None).alias("qb_id"))
    p = p.filter(pl.col("qb_id").is_not_null())
    attempt = _flag("pass") & ~_flag("sack") & ~_flag("qb_scramble")
    qb_epa = pl.when(pl.col("qb_epa").is_not_null()).then(pl.col("qb_epa")).otherwise(pl.col("epa"))
    frame = p.group_by(["game_id", "posteam", "qb_id"]).agg([
        pl.len().alias("dropbacks"), attempt.sum().alias("attempts"), (attempt & _flag("complete_pass")).sum().alias("completions"),
        qb_epa.sum().alias("epa_total"), qb_epa.mean().alias("epa_dropback"), pl.col("success").cast(pl.Float64).mean().alias("success_rate"),
        pl.col("cpoe").filter(attempt & pl.col("cpoe").is_not_null()).mean().alias("cpoe"), pl.col("air_yards").filter(attempt & pl.col("air_yards").is_not_null()).mean().alias("adot"),
        pl.col("yards_gained").filter(attempt).sum().alias("pass_yards"), (attempt & _flag("pass_touchdown")).sum().alias("pass_td"), (attempt & _flag("interception")).sum().alias("interceptions"),
        _flag("sack").sum().alias("sacks"), _flag("qb_hit").sum().alias("qb_hits"), _flag("qb_scramble").sum().alias("scrambles"), pl.col("epa").filter(_flag("qb_scramble")).sum().alias("scramble_epa"),
        (pl.col("air_yards") >= 20).filter(attempt).cast(pl.Float64).mean().alias("deep_rate"), qb_epa.filter(pl.col("wp").is_between(0.10, 0.90)).mean().alias("epa_dropback_neutral"),
        pl.col("season").first(), pl.col("week").first(), pl.col("season_type").first(), pl.col("home_team").first(), pl.col("away_team").first(),
    ]).rename({"posteam": "team"})
    frame = frame.with_columns([
        pl.when(pl.col("team") == pl.col("home_team")).then(pl.col("away_team")).otherwise(pl.col("home_team")).alias("opponent"),
        (pl.col("sacks") / pl.col("dropbacks")).alias("sack_rate"), (pl.col("interceptions") / pl.when(pl.col("attempts") > 0).then(pl.col("attempts")).otherwise(None)).alias("int_rate"),
        (pl.col("completions") / pl.when(pl.col("attempts") > 0).then(pl.col("attempts")).otherwise(None)).alias("comp_pct"),
        ((pl.col("pass_yards") + 20 * pl.col("pass_td") - 45 * pl.col("interceptions")) / pl.when(pl.col("dropbacks") > 0).then(pl.col("dropbacks")).otherwise(None)).alias("anya"),
    ])
    return frame


def kicker_game_frame(plays: pl.DataFrame) -> pl.DataFrame:
    fg = plays.filter(_flag("field_goal_attempt") & pl.col("kicker_player_id").is_not_null())
    made = pl.col("field_goal_result") == "made"
    dist = pl.col("kick_distance").cast(pl.Float64)
    return fg.group_by(["game_id", "posteam", "kicker_player_id"]).agg([
        pl.len().alias("fg_att"), made.sum().alias("fg_made"), dist.mean().alias("fg_distance_mean"), dist.filter(made).max().alias("fg_longest_made"),
        (dist >= 50).sum().alias("fg_att_50"), (made & (dist >= 50)).sum().alias("fg_made_50"), pl.col("season").first(), pl.col("week").first(),
    ]).rename({"posteam": "team", "kicker_player_id": "kicker_id"})


def build_team_and_qb_games(out: Path, seasons: list[int]) -> None:
    """Per-season aggregates are cached; a season is re-aggregated only when its plays file changed."""
    teams, qbs, kickers = [], [], []
    parts = out / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    for season in seasons:
        source = out / "plays" / f"{season}.parquet"
        cached = {kind: parts / f"{kind}_{season}.parquet" for kind in ("team", "qb", "kicker")}
        if all(c.exists() and c.stat().st_mtime >= source.stat().st_mtime for c in cached.values()):
            teams.append(pl.read_parquet(cached["team"]))
            qbs.append(pl.read_parquet(cached["qb"]))
            kickers.append(pl.read_parquet(cached["kicker"]))
            continue
        plays = pl.read_parquet(source)
        t, q, k = team_game_frame(plays), qb_game_frame(plays), kicker_game_frame(plays)
        t.write_parquet(cached["team"])
        q.write_parquet(cached["qb"])
        k.write_parquet(cached["kicker"])
        teams.append(t)
        qbs.append(q)
        kickers.append(k)
        log.info("aggregated season %s", season)
    team_games = pl.concat(teams, how="diagonal_relaxed")
    games = pl.read_parquet(out / "games.parquet").select(["game_id", "kickoff_utc", "game_type", "playoff"])
    team_games = team_games.join(games, on="game_id", how="left").sort(["kickoff_utc", "game_id", "team"])
    team_games.write_parquet(out / "team_games.parquet")
    qb_games = pl.concat(qbs, how="diagonal_relaxed").join(games, on="game_id", how="left").sort(["kickoff_utc", "game_id", "team", "dropbacks"])
    qb_games.write_parquet(out / "qb_games.parquet")
    pl.concat(kickers, how="diagonal_relaxed").join(games, on="game_id", how="left").sort(["kickoff_utc", "game_id"]).write_parquet(out / "kicker_games.parquet")


def build_players(raw: RawStore, out: Path) -> pl.DataFrame:
    players = pl.read_parquet(raw.path("players", "players.parquet"))
    keep = [c for c in ("gsis_id", "display_name", "position", "position_group", "birth_date", "height", "weight", "rookie_season", "last_season", "draft_year", "draft_round", "draft_pick", "draft_team", "pfr_id", "espn_id", "esb_id", "college_name", "years_of_experience", "status", "latest_team") if c in players.columns]
    frame = players.select(keep).filter(pl.col("gsis_id").is_not_null()).unique(subset=["gsis_id"], keep="first")
    frame.write_parquet(out / "players.parquet")
    return frame


def build_injuries(raw: RawStore, out: Path) -> pl.DataFrame:
    frames = []
    for path in raw.files("injuries"):
        f = pl.read_parquet(path)
        cols = {c: c for c in ("season", "game_type", "team", "week", "gsis_id", "position", "full_name", "report_primary_injury", "report_status", "practice_status")}
        f = f.select([pl.col(c) for c in cols if c in f.columns] + ([pl.col("date_modified").cast(pl.Datetime("us", "UTC")).alias("date_modified")] if "date_modified" in f.columns else [pl.lit(None, dtype=pl.Datetime("us", "UTC")).alias("date_modified")]))
        frames.append(f)
    if not frames:
        return pl.DataFrame()
    inj = pl.concat(frames, how="diagonal_relaxed")
    inj = inj.with_columns([
        pl.col("team").map_elements(canonical, return_dtype=pl.String).alias("team"),
        pl.col("report_status").cast(pl.String).str.to_lowercase().alias("report_status"),
        pl.col("practice_status").cast(pl.String).str.to_lowercase().alias("practice_status"),
        pl.col("season").cast(pl.Int32), pl.col("week").cast(pl.Int32),
    ]).filter(pl.col("gsis_id").is_not_null())
    inj.write_parquet(out / "injuries.parquet")
    return inj


def build_snaps(raw: RawStore, out: Path, players: pl.DataFrame) -> pl.DataFrame:
    frames = [pl.read_parquet(p) for p in raw.files("snap_counts")]
    if not frames:
        return pl.DataFrame()
    snaps = pl.concat(frames, how="diagonal_relaxed")
    snaps = snaps.with_columns([pl.col("team").map_elements(canonical, return_dtype=pl.String).alias("team"), pl.col("opponent").map_elements(canonical, return_dtype=pl.String).alias("opponent")])
    idmap = players.select(["pfr_id", "gsis_id"]).filter(pl.col("pfr_id").is_not_null()).unique(subset=["pfr_id"], keep="first")
    snaps = snaps.join(idmap, left_on="pfr_player_id", right_on="pfr_id", how="left")
    # nflverse game_id for snap counts is already the nflverse id (e.g. 2024_01_BAL_KC); relocations use historical codes in the id string only.
    snaps.write_parquet(out / "snaps.parquet")
    return snaps


def build_depth_charts(raw: RawStore, out: Path) -> pl.DataFrame:
    frames = []
    for path in raw.files("depth_charts"):
        f = pl.read_parquet(path)
        if "dt" in f.columns:  # 2025+ timestamped format
            g = f.select([
                pl.col("dt").str.to_datetime("%Y-%m-%dT%H:%M:%SZ", time_zone="UTC").alias("as_of"), pl.col("team"), pl.col("gsis_id"), pl.col("pos_abb").alias("slot"), pl.col("pos_grp").alias("unit_raw"), pl.col("pos_rank").cast(pl.Int32).alias("rank"), pl.col("player_name").alias("full_name"),
            ]).with_columns([pl.lit(None, dtype=pl.Int32).alias("week"), pl.lit(int(path.stem.split("_")[-1]), dtype=pl.Int32).alias("season")])
        else:
            if "game_type" in f.columns:
                f = f.filter(pl.col("game_type").fill_null("REG") == "REG")
            g = f.select([
                pl.lit(None, dtype=pl.Datetime("us", "UTC")).alias("as_of"), pl.col("club_code").alias("team"), pl.col("gsis_id"), pl.col("depth_position").alias("slot"), pl.col("formation").alias("unit_raw"), pl.col("depth_team").cast(pl.Int32, strict=False).alias("rank"), pl.col("full_name"),
                pl.col("week").cast(pl.Int32), pl.col("season").cast(pl.Int32),
            ])
        frames.append(g)
    if not frames:
        return pl.DataFrame()
    depth = pl.concat(frames, how="diagonal_relaxed")
    depth = depth.with_columns([pl.col("team").map_elements(canonical, return_dtype=pl.String).alias("team"), pl.col("unit_raw").cast(pl.String).str.to_lowercase().alias("unit_raw")])
    depth = depth.with_columns(pl.when(pl.col("unit_raw").str.contains("off")).then(pl.lit("offense")).when(pl.col("unit_raw").str.contains("def") | pl.col("unit_raw").str.contains(" d")).then(pl.lit("defense")).when(pl.col("unit_raw").str.contains("special") | pl.col("unit_raw").str.contains("st")).then(pl.lit("special")).otherwise(pl.lit("unknown")).alias("unit")).filter(pl.col("gsis_id").is_not_null())
    depth.write_parquet(out / "depth_charts.parquet")
    return depth


def build_rosters(raw: RawStore, out: Path) -> pl.DataFrame:
    frames = []
    for path in raw.files("weekly_rosters"):
        f = pl.read_parquet(path)
        cols = [c for c in ("season", "week", "game_type", "team", "gsis_id", "position", "depth_chart_position", "status", "years_exp", "entry_year", "rookie_year", "draft_number", "birth_date", "full_name") if c in f.columns]
        frames.append(f.select(cols).with_columns([pl.col("season").cast(pl.Int32), pl.col("week").cast(pl.Int32)]))
    if not frames:
        return pl.DataFrame()
    rosters = pl.concat(frames, how="diagonal_relaxed").with_columns(pl.col("team").map_elements(canonical, return_dtype=pl.String).alias("team")).filter(pl.col("gsis_id").is_not_null())
    rosters.write_parquet(out / "rosters.parquet")
    return rosters


def quality_report(out: Path) -> dict:
    games = pl.read_parquet(out / "games.parquet")
    team_games = pl.read_parquet(out / "team_games.parquet")
    completed = games.filter(pl.col("completed"))
    report = {
        "generated_at": utcnow(), "games": games.height, "completed_games": completed.height, "seasons": [int(games["season"].min()), int(games["season"].max())],
        "duplicate_game_ids": int(games.height - games["game_id"].n_unique()), "missing_kickoff": int(games["kickoff_utc"].null_count()), "imputed_kickoff_times": int(games["kickoff_imputed"].sum()),
        "future_completed_games": int(completed.filter(pl.col("kickoff_utc") > datetime.now(timezone.utc)).height),
        "impossible_scores": int(completed.filter((pl.col("home_score") < 0) | (pl.col("away_score") < 0) | (pl.col("home_score") > 80) | (pl.col("away_score") > 80)).height),
        "same_team_games": int(games.filter(pl.col("home_team") == pl.col("away_team")).height),
        "team_codes": sorted(set(games["home_team"].to_list()) | set(games["away_team"].to_list())),
        "team_game_rows": team_games.height, "team_games_per_completed_game": round(team_games.height / max(1, completed.filter(pl.col("season") <= team_games["season"].max()).height), 3),
        "team_games_missing_join": int(team_games.filter(pl.col("kickoff_utc").is_null()).height),
        "market_coverage": {"spread": round(1 - completed["spread_line"].null_count() / completed.height, 4), "total": round(1 - completed["total_line"].null_count() / completed.height, 4), "moneyline": round(1 - completed["home_moneyline"].null_count() / completed.height, 4)},
        "weather_coverage": round(1 - completed["temp"].null_count() / completed.height, 4), "neutral_venue_matches": games.filter(pl.col("neutral")).group_by("venue_match").len().to_dicts(),
    }
    (paths().reports / "data_quality.json").write_text(json.dumps(report, indent=2, default=str))
    return report


def normalize_all(seasons: list[int] | None = None, root: Path | None = None) -> dict:
    p = paths(root)
    raw = RawStore(p.raw)
    out = p.normalized
    build_games(raw, out)
    built = build_plays(raw, out, seasons)
    build_team_and_qb_games(out, built)
    players = build_players(raw, out)
    build_injuries(raw, out)
    build_snaps(raw, out, players)
    build_depth_charts(raw, out)
    build_rosters(raw, out)
    return quality_report(out)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    print(json.dumps(normalize_all(), indent=2, default=str))
