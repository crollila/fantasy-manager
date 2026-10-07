"""Power rankings from the champion forecasting model.

Each team's rating is the margin the champion's market-free ensemble expects it to win by
against an average NFL team on a neutral field. For a week, every team's pregame state
(Elo, margin and efficiency ratings, recent form, quarterback, injuries, coaching) is
paired against every other team's, in both home/away orientations, with neutral game
context (league-median weather, rest, officials). Averaging both orientations cancels
home-field advantage, so the rating is a property of the team, not of its schedule.
"""
from __future__ import annotations
import json
import logging
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import polars as pl
from app.nfl.config import paths
from app.nfl.features.builder import load_features, add_matchup_columns, family_of
from app.nfl.models.ensemble import apply_weights
from app.nfl.models.calibration import margin_sd, win_probability_from_margin, ProbabilityCalibrator
from app.nfl.models.registry import Registry
from app.nfl.reference.teams import TEAMS, DISPLAY

log = logging.getLogger(__name__)
ELO_HFA = 55.0  # matches app.nfl.features.ratings.elo_features
CONTEXT_FAMILIES = ("rest_travel", "weather", "schedule", "officials")


def _side_columns(frame: pl.DataFrame) -> list[str]:
    return [c.removeprefix("home_") for c in frame.columns if c.startswith("home_") and f"away_{c.removeprefix('home_')}" in frame.columns and c not in ("home_team", "home_score", "home_win")]


def team_states(frame: pl.DataFrame, season: int, week: int) -> dict[str, dict]:
    """Each team's side features entering ``week``: from its game that week, or its next game
    after a bye. Rolling-form values that are only evaluated for a team's next game are taken
    from its latest row where they exist."""
    keys = _side_columns(frame)
    rows = frame.filter(pl.col("season") == season).sort(["week", "kickoff_utc"]).to_dicts()
    states = {}
    for team in TEAMS:
        mine = [(r, "home" if r["home_team"] == team else "away") for r in rows if team in (r["home_team"], r["away_team"])]
        if not mine:
            continue
        current = next(((r, s) for r, s in mine if r["week"] >= week), mine[-1])
        row, side = current
        state = {k: row.get(f"{side}_{k}") for k in keys}
        for r, s in reversed([m for m in mine if m[0]["week"] < row["week"]]):
            missing = [k for k, v in state.items() if v is None and "__" in k]
            if not missing:
                break
            for k in missing:
                state[k] = r.get(f"{s}_{k}")
        states[team] = state | {"_week": row["week"], "_game_id": row["game_id"]}
    return states


def neutral_context(frame: pl.DataFrame, season: int, week: int, keys: list[str]) -> dict:
    """League-median game context for the week (falls back to the whole season)."""
    pool = frame.filter((pl.col("season") == season) & (pl.col("week") == week))
    if pool.height == 0:
        pool = frame.filter(pl.col("season") == season)
    out = {}
    for c in frame.columns:
        if c.startswith(("home_", "away_", "diff_", "mx_")) or c in ("elo_diff", "elo_prob_home", "elo_margin") or c.startswith("mkt_"):
            continue
        if pool.schema[c] in (pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Int16, pl.Int8, pl.Boolean):
            out[c] = pool[c].cast(pl.Float64).median()
    for k in keys:  # rest and travel describe the fixture, not the team
        if family_of(f"home_{k}") in CONTEXT_FAMILIES:
            out[f"home_{k}"] = out[f"away_{k}"] = pl.concat([pool[f"home_{k}"].cast(pl.Float64), pool[f"away_{k}"].cast(pl.Float64)]).median()
    out["rest_diff"] = 0.0
    out["ctx_neutral"] = 0.0
    return out


def matchup_frame(states: dict[str, dict], context: dict, keys: list[str]) -> pl.DataFrame:
    teams = sorted(states)
    records = []
    for h in teams:
        for a in teams:
            if h == a:
                continue
            rec = {"home_team": h, "away_team": a} | {k: v for k, v in context.items()}
            for k in keys:
                if f"home_{k}" in context:
                    continue
                rec[f"home_{k}"] = states[h].get(k)
                rec[f"away_{k}"] = states[a].get(k)
            diff = (rec.get("home_elo_pre") or 1500.0) - (rec.get("away_elo_pre") or 1500.0) + ELO_HFA
            rec |= {"elo_diff": diff, "elo_prob_home": 1.0 / (1.0 + 10 ** (-diff / 400.0)), "elo_margin": diff / 25.0}
            records.append(rec)
    frame = pl.DataFrame(records, infer_schema_length=None)
    numeric = [c for c, t in frame.schema.items() if c not in ("home_team", "away_team")]
    frame = frame.with_columns([pl.col(c).cast(pl.Float64, strict=False) for c in numeric])
    return add_matchup_columns(frame)


def predict_pairs(frame: pl.DataFrame, artifact: dict) -> pd.DataFrame:
    system = artifact["systems"]["ensemble_market_free"]
    learners = artifact["models"]["learners"]
    needed = {c for c in system["weights_margin"]["columns"] + system["weights_total"]["columns"] if c in learners}
    preds = pd.DataFrame({"home_team": frame["home_team"].to_list(), "away_team": frame["away_team"].to_list()})
    cache = {}
    for key in needed:
        entry = learners[key]
        cols = entry["columns"]
        if entry["feature_set"] not in cache:
            missing = [c for c in cols if c not in frame.columns]
            cache[entry["feature_set"]] = frame.with_columns([pl.lit(None, dtype=pl.Float64).alias(c) for c in missing]).select([pl.col(c).cast(pl.Float64) for c in cols]).to_numpy()
        preds[key] = entry["learner"].predict(cache[entry["feature_set"]])
    base = artifact["models"].get("baselines", {"elo_slope": 0.04, "avg_total": 44.0})
    preds["elo__margin"] = frame["elo_diff"].to_numpy() * base["elo_slope"]
    preds["elo__total_points"] = base["avg_total"]
    preds["margin"] = apply_weights(preds, system["weights_margin"])
    preds["total"] = apply_weights(preds, system["weights_total"])
    residual = system["residual"]
    sd = margin_sd(residual, preds["total"].to_numpy())
    p = win_probability_from_margin(preds["margin"].to_numpy(), sd, residual["t_df"], residual["tie_rate"])
    preds["home_win"] = ProbabilityCalibrator.from_dict(system["calibrator"]).predict(p)
    return preds


def ratings(preds: pd.DataFrame) -> dict[str, dict]:
    out = {}
    for team in sorted(set(preds["home_team"])):
        home = preds[preds["home_team"] == team]
        away = preds[preds["away_team"] == team]
        margin = (home["margin"].mean() - away["margin"].mean()) / 2
        scored = (((home["total"] + home["margin"]) / 2).mean() + ((away["total"] - away["margin"]) / 2).mean()) / 2
        allowed = (((home["total"] - home["margin"]) / 2).mean() + ((away["total"] + away["margin"]) / 2).mean()) / 2
        win = (home["home_win"].mean() + (1 - away["home_win"]).mean()) / 2
        out[team] = {"rating": float(margin), "points_for": float(scored), "points_against": float(allowed), "win_vs_average": float(win)}
    league_for = np.mean([v["points_for"] for v in out.values()])
    for v in out.values():
        v["offense"] = v["points_for"] - league_for
        v["defense"] = league_for - v["points_against"]
    return out


def records(season: int, through_week: int) -> dict[str, dict]:
    games = pl.read_parquet(paths().normalized / "games.parquet").filter((pl.col("season") == season) & (pl.col("week") < through_week) & pl.col("home_score").is_not_null() & (pl.col("game_type") == "REG"))
    out = {t: {"wins": 0, "losses": 0, "ties": 0, "points_for": 0, "points_against": 0, "results": []} for t in TEAMS}
    for g in games.sort("week").to_dicts():
        for team, opp, pf, pa, home in ((g["home_team"], g["away_team"], g["home_score"], g["away_score"], True), (g["away_team"], g["home_team"], g["away_score"], g["home_score"], False)):
            r = out.setdefault(team, {"wins": 0, "losses": 0, "ties": 0, "points_for": 0, "points_against": 0, "results": []})
            r["wins" if pf > pa else "losses" if pf < pa else "ties"] += 1
            r["points_for"] += pf
            r["points_against"] += pa
            r["results"].append({"week": g["week"], "opponent": DISPLAY.get(opp, opp), "home": home, "points_for": pf, "points_against": pa, "result": "W" if pf > pa else "L" if pf < pa else "T"})
    return out


def current_week(frame: pl.DataFrame, season: int) -> int:
    pending = frame.filter((pl.col("season") == season) & (pl.col("game_type") == "REG") & ~pl.col("completed"))
    return int(pending["week"].min()) if pending.height else int(frame.filter(pl.col("season") == season)["week"].max()) + 1


def rank_week(frame: pl.DataFrame, artifact: dict, season: int, week: int) -> dict[str, dict]:
    keys = _side_columns(frame)
    states = team_states(frame, season, week)
    table = ratings(predict_pairs(matchup_frame(states, neutral_context(frame, season, week, keys), keys), artifact))
    order = sorted(table, key=lambda t: -table[t]["rating"])
    for i, t in enumerate(order, 1):
        table[t]["rank"] = i
        table[t]["elo"] = states[t].get("elo_pre")
        table[t]["srs"] = states[t].get("srs")
        table[t]["qb_epa"] = states[t].get("qb_epa_season")
    return table


def power_rankings(season: int | None = None, history: bool = True) -> dict:
    """Rankings entering the next unplayed week, plus every earlier week of the season."""
    registry = Registry()
    champion = registry.champion("game_forecast")
    if champion is None:
        return {"status": "no champion model registered", "teams": []}
    frame, _ = load_features("pregame")
    season = season or int(frame.filter(pl.col("completed"))["season"].max())
    week = current_week(frame, season)
    artifact = registry.load(champion["model_id"])
    weeks = list(range(1, week + 1)) if history else [week - 1, week]
    by_week = {w: rank_week(frame, artifact, season, w) for w in weeks if w >= 1}
    now = by_week[week]
    record = records(season, week)
    games = pl.read_parquet(paths().normalized / "games.parquet").filter((pl.col("season") == season) & (pl.col("week") >= week)).sort("kickoff_utc").to_dicts()
    teams = []
    for team, r in sorted(now.items(), key=lambda kv: kv[1]["rank"]):
        prev = by_week.get(week - 1, {}).get(team)
        nxt = next((g for g in games if team in (g["home_team"], g["away_team"])), None)
        rec = record.get(team, {})
        teams.append({"team": DISPLAY.get(team, team), "rank": r["rank"], "previous_rank": prev["rank"] if prev else None, "change": (prev["rank"] - r["rank"]) if prev else None,
                      "rating": round(r["rating"], 2), "offense": round(r["offense"], 2), "defense": round(r["defense"], 2), "win_vs_average": round(r["win_vs_average"], 4),
                      "elo": round(r["elo"], 0) if r["elo"] is not None else None, "srs": round(r["srs"], 2) if r["srs"] is not None else None,
                      "wins": rec.get("wins", 0), "losses": rec.get("losses", 0), "ties": rec.get("ties", 0), "points_for": rec.get("points_for", 0), "points_against": rec.get("points_against", 0),
                      "recent": rec.get("results", [])[-5:],
                      "next_game": None if nxt is None else {"week": nxt["week"], "opponent": DISPLAY.get(o := (nxt["away_team"] if nxt["home_team"] == team else nxt["home_team"]), o), "home": nxt["home_team"] == team, "kickoff": nxt["kickoff_utc"].isoformat() if nxt["kickoff_utc"] else None},
                      "history": [{"week": w, "rank": by_week[w][team]["rank"], "rating": round(by_week[w][team]["rating"], 2)} for w in sorted(by_week) if team in by_week[w]]})
    return {"season": season, "week": week, "model_id": champion["model_id"], "generated_at": datetime.now(timezone.utc).isoformat(), "teams": teams,
            "method": "Expected point margin against an average NFL team on a neutral field, from the champion market-free ensemble: every team's current pregame state is paired against every other team in both home/away orientations with league-median game context, so home-field advantage and schedule cancel out. Offense/defense are expected points scored/allowed relative to league average. Week N is the ranking entering week N."}


def cached_power_rankings(refresh: bool = False) -> dict:
    """Recomputed only when the champion or the feature table changes."""
    p = paths()
    feature_file = p.features / "fv1" / "games_pregame.parquet"
    try:
        champion = Registry().champion("game_forecast") or {}
        key = f"{champion.get('model_id')}|{feature_file.stat().st_mtime_ns}"
    except Exception:  # noqa: BLE001
        key = None
    cache = p.reports / "power_rankings.json"
    if not refresh and key and cache.exists():
        try:
            data = json.loads(cache.read_text())
            if data.get("cache_key") == key:
                return data
        except ValueError:
            pass
    data = power_rankings() | {"cache_key": key}
    cache.write_text(json.dumps(data, default=str))
    return data
