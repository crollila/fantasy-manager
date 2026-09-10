"""Live forecasts for upcoming games from the champion artifact.

Steps: refresh point-in-time features (already includes scheduled future games), run every
production base learner, blend with the frozen ensemble weights, derive calibrated win
probabilities and a Monte Carlo score distribution, attach market comparisons and the
strongest drivers, and persist machine-readable outputs.
"""
from __future__ import annotations
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import polars as pl
from app.nfl.config import paths, FEATURE_VERSION, SEED
from app.nfl.features.builder import build_features, load_features
from app.nfl.models.prepare import to_matrix
from app.nfl.models.ensemble import apply_weights
from app.nfl.models.calibration import margin_sd, win_probability_from_margin, ProbabilityCalibrator
from app.nfl.models.simulate import simulate_game
from app.nfl.models.registry import Registry
from app.nfl.reference.teams import DISPLAY

log = logging.getLogger(__name__)
FAMILY_LABELS = {"elo": "Team strength (Elo)", "ratings_margin": "Team strength (margin ratings)", "ratings_efficiency": "Opponent-adjusted efficiency", "team_form_offense": "Recent offensive form", "team_form_defense": "Recent defensive form", "qb": "Quarterback",
                 "availability": "Injuries / availability", "rest_travel": "Rest and travel", "schedule": "Schedule context", "weather": "Weather", "coaching": "Coaching", "officials": "Officials", "matchup": "Matchup interactions", "market": "Betting market"}


def _contributions(model_entry: dict, X: np.ndarray, families: dict) -> list[dict]:
    """Per-feature contributions: SHAP values for boosting members, coefficient x standardised value for ridge."""
    learner = model_entry["learner"]
    if not hasattr(learner, "contributions"):
        return []
    try:
        contrib = np.asarray(learner.contributions(X))
    except Exception:  # noqa: BLE001 - explanations are best effort
        return []
    return [{"feature": c, "value": float(v)} for c, v in zip(model_entry["columns"], contrib[0])]


def explain(artifact: dict, system: dict, X_by_set: dict, families: dict, feature_values: dict) -> dict:
    per_family: dict[str, float] = {}
    top: dict[str, float] = {}
    weights = system["weights_margin"]["weights"]
    for key, weight in weights.items():
        entry = artifact["models"]["learners"].get(key)
        if not entry or weight <= 0:
            continue
        for item in _contributions(entry, X_by_set[entry["feature_set"]], families):
            fam = families.get(item["feature"], "other")
            per_family[fam] = per_family.get(fam, 0.0) + weight * item["value"]
            top[item["feature"]] = top.get(item["feature"], 0.0) + weight * item["value"]
    ranked = sorted(top.items(), key=lambda kv: -abs(kv[1]))
    positives = [{"feature": f, "points": round(v, 2), "family": FAMILY_LABELS.get(families.get(f, "other"), "Other"), "value": feature_values.get(f)} for f, v in ranked if v > 0][:6]
    negatives = [{"feature": f, "points": round(v, 2), "family": FAMILY_LABELS.get(families.get(f, "other"), "Other"), "value": feature_values.get(f)} for f, v in ranked if v < 0][:6]
    return {"by_family_points": {FAMILY_LABELS.get(k, k): round(v, 2) for k, v in sorted(per_family.items(), key=lambda kv: -abs(kv[1]))}, "home_positive": positives, "home_negative": negatives,
            "note": "Margin contributions (home minus away points) from the learned members (SHAP for boosting, coefficient x standardised value for ridge), weighted by ensemble weight. Descriptive, not causal."}


def forecast_games(frame: pl.DataFrame, meta: dict, artifact: dict, n_sims: int = 50000, market_aware: bool = True) -> list[dict]:
    families = {c: fam for fam, cols in meta["families"].items() for c in cols}
    learners = artifact["models"]["learners"]
    sets_needed = {e["feature_set"]: e["columns"] for e in learners.values()}
    X_by_set = {name: to_matrix(frame, cols) for name, cols in sets_needed.items()}
    preds = pd.DataFrame({"game_id": frame["game_id"].to_list()})
    for key, entry in learners.items():
        offset = frame[entry["offset"]].cast(pl.Float64).to_numpy().astype(float) if entry.get("offset") else 0.0
        preds[key] = entry["learner"].predict(X_by_set[entry["feature_set"]]) + offset
    # Analytic baseline members used by the ensembles.
    baselines = artifact["models"].get("baselines", {"elo_slope": 0.04, "avg_total": 44.0})
    preds["elo__margin"] = frame["elo_diff"].cast(pl.Float64).to_numpy().astype(float) * baselines["elo_slope"]
    preds["elo__total_points"] = baselines["avg_total"]
    preds["market__margin"] = frame["mkt_spread"].cast(pl.Float64).to_numpy().astype(float)
    preds["market__total_points"] = frame["mkt_total"].cast(pl.Float64).to_numpy().astype(float)
    preds["market__home_win"] = frame["mkt_home_prob"].cast(pl.Float64).to_numpy().astype(float)
    results = []
    rows = frame.to_dicts()
    for i, row in enumerate(rows):
        game = {"game_id": row["game_id"], "season": row["season"], "week": row["week"], "game_type": row["game_type"], "kickoff_utc": row["kickoff_utc"].isoformat() if row["kickoff_utc"] else None, "home_team": row["home_team"], "away_team": row["away_team"],
                "home_team_display": DISPLAY.get(row["home_team"], row["home_team"]), "away_team_display": DISPLAY.get(row["away_team"], row["away_team"]), "prediction_timestamp": datetime.now(timezone.utc).isoformat(), "feature_version": FEATURE_VERSION, "model_id": artifact["model_id"]}
        one = preds.iloc[[i]]
        for label, system_name in (("market_free", "ensemble_market_free"), ("market_aware", "ensemble_market_aware")):
            system = artifact["systems"][system_name]
            if label == "market_aware" and (not market_aware or not np.isfinite(row.get("mkt_spread") or np.nan)):
                game[label] = {"available": False, "reason": "no market line available"}
                continue
            margin = float(apply_weights(one, system["weights_margin"])[0])
            total = float(apply_weights(one, system["weights_total"])[0])
            residual = system["residual"]
            sd = float(margin_sd(residual, np.array([total]))[0])
            p_margin = float(win_probability_from_margin(np.array([margin]), np.array([sd]), residual["t_df"], residual["tie_rate"])[0])
            p = p_margin
            if system["probability_source"] != "margin_only" and system["classifiers"]:
                p_clf = float(one[system["classifiers"]].to_numpy().astype(float).mean())
                p = p_clf if system["probability_source"] == "classifier_only" else 0.5 * p_clf + 0.5 * p_margin
            p = float(ProbabilityCalibrator.from_dict(system["calibrator"]).predict(np.array([p]))[0])
            sim = simulate_game(margin, total, residual, n=n_sims, seed=SEED + i)
            # Keep the calibrated probability as the headline; simulation supplies the distribution shape.
            home_pts, away_pts = (total + margin) / 2, (total - margin) / 2
            block = {"available": True, "expected_home_points": round(home_pts, 2), "expected_away_points": round(away_pts, 2), "expected_margin": round(margin, 2), "expected_total": round(total, 2),
                     "displayed_score": {"home": int(round(home_pts)), "away": int(round(away_pts))}, "home_win_probability": round(p, 4), "away_win_probability": round(1 - p - sim["tie"], 4), "tie_probability": round(sim["tie"], 4),
                     "margin_sd": round(sd, 2), "total_sd": round(sim["total_sd"], 2), "home_score_interval_80": [round(v, 1) for v in sim["home_interval_80"]], "away_score_interval_80": [round(v, 1) for v in sim["away_interval_80"]],
                     "home_score_interval_50": [round(v, 1) for v in sim["home_interval_50"]], "away_score_interval_50": [round(v, 1) for v in sim["away_interval_50"]], "margin_interval_80": [round(v, 1) for v in sim["margin_interval_80"]], "total_interval_80": [round(v, 1) for v in sim["total_interval_80"]],
                     "simulated_home_win": round(sim["home_win"], 4), "upset_probability": round(min(p, 1 - p), 4), "most_likely_scores": sim["most_likely_scores"], "spread_cover_probabilities": sim["spread_cover"], "total_over_probabilities": sim["total_over"],
                     "margin_distribution": sim["margin_distribution"], "total_distribution": sim["total_distribution"], "base_predictions": {k: round(float(one[k].iloc[0]), 3) for k in system["weights_margin"]["columns"] + system["weights_total"]["columns"] + system["classifiers"] if k in one.columns},
                     "probability_source": system["probability_source"], "calibration": system["calibrator"]["method"], "simulations": n_sims}
            game[label] = block
        spread, total_line, mkt_prob = row.get("mkt_spread"), row.get("mkt_total"), row.get("mkt_home_prob")
        if spread is not None and np.isfinite(spread):
            free = game["market_free"]
            game["market"] = {"spread_home_margin": spread, "total": total_line, "no_vig_home_probability": mkt_prob, "source": "nflverse schedule snapshot (latest available line; not timestamped)",
                              "model_fair_spread": free["expected_margin"], "spread_difference": round(free["expected_margin"] - spread, 2), "model_fair_total": free["expected_total"], "total_difference": round(free["expected_total"] - (total_line or np.nan), 2) if total_line is not None else None,
                              "model_win_probability": free["home_win_probability"], "probability_difference": round(free["home_win_probability"] - mkt_prob, 4) if mkt_prob is not None and np.isfinite(mkt_prob) else None}
        # Context and drivers.
        feature_values = {c: (round(float(row[c]), 4) if isinstance(row.get(c), (int, float)) and row.get(c) is not None and np.isfinite(row[c]) else None) for c in families}
        game["context"] = {"home_qb": row.get("home_qb_id"), "away_qb": row.get("away_qb_id"), "home_qb_epa_career": feature_values.get("home_qb_epa_career"), "away_qb_epa_career": feature_values.get("away_qb_epa_career"), "home_elo": feature_values.get("home_elo_pre"), "away_elo": feature_values.get("away_elo_pre"),
                           "home_value_lost_offense": feature_values.get("home_value_lost_off"), "away_value_lost_offense": feature_values.get("away_value_lost_off"), "home_value_lost_defense": feature_values.get("home_value_lost_def"), "away_value_lost_defense": feature_values.get("away_value_lost_def"),
                           "home_rest": feature_values.get("home_rest"), "away_rest": feature_values.get("away_rest"), "away_travel_miles": feature_values.get("away_travel"), "temperature_f": feature_values.get("wx_temp"), "wind_mph": feature_values.get("wx_wind"), "indoor": feature_values.get("ctx_indoor"), "neutral_site": feature_values.get("ctx_neutral")}
        try:
            game["drivers"] = explain(artifact, artifact["systems"]["ensemble_market_free"], {k: v[[i]] for k, v in X_by_set.items()}, families, feature_values)
        except Exception as exc:  # noqa: BLE001 - explanations are best effort
            game["drivers"] = {"error": str(exc)}
        results.append(game)
    return results


def upcoming_games(frame: pl.DataFrame, season: int | None, week: int | None, as_of: datetime) -> pl.DataFrame:
    f = frame.filter(~pl.col("completed").fill_null(False))
    if season is not None:
        f = f.filter(pl.col("season") == season)
    if week is None:
        future = f.filter(pl.col("kickoff_utc") > as_of)
        if future.height == 0:
            return future
        week = int(future.sort("kickoff_utc")["week"][0])
        season = int(future.sort("kickoff_utc")["season"][0])
        f = f.filter((pl.col("season") == season) & (pl.col("week") == week))
    else:
        f = f.filter(pl.col("week") == week)
    return f.sort("kickoff_utc")


def run(season: int | None = None, week: int | None = None, rebuild: bool = True, n_sims: int = 50000, root: Path | None = None, as_of: datetime | None = None) -> dict:
    p = paths(root)
    as_of = as_of or datetime.now(timezone.utc)
    registry = Registry(root)
    champion = registry.champion("game_forecast")
    if champion is None:
        raise RuntimeError("No champion model registered; run `python -m app.nfl train` first")
    artifact = registry.load(champion["model_id"])
    frame = build_features("pregame", root) if rebuild else load_features("pregame", root)[0]
    meta = load_features("pregame", root)[1]
    games = upcoming_games(frame, season, week, as_of)
    if games.height == 0:
        return {"status": "no upcoming games", "as_of": as_of.isoformat()}
    forecasts = forecast_games(games, meta, artifact, n_sims=n_sims)
    for g in forecasts:
        kickoff = datetime.fromisoformat(g["kickoff_utc"]) if g["kickoff_utc"] else None
        g["kicked_off_before_prediction"] = bool(kickoff and kickoff <= as_of)
    season_v, week_v = int(games["season"][0]), int(games["week"][0])
    out = {"status": "ok", "season": season_v, "week": week_v, "generated_at": as_of.isoformat(), "model_id": champion["model_id"], "feature_version": FEATURE_VERSION, "games": forecasts, "sources_note": "nflverse schedules/pbp/injuries/depth charts/rosters; observed weather unavailable pregame (schedule fields empty until game day)"}
    folder = p.predictions / "current"
    stem = f"{season_v}_week{week_v:02d}"
    (folder / f"{stem}.json").write_text(json.dumps(out, indent=2, default=str))
    flat = pd.DataFrame([{
        "game_id": g["game_id"], "kickoff_utc": g["kickoff_utc"], "home_team": g["home_team"], "away_team": g["away_team"], "model_id": g["model_id"], "prediction_timestamp": g["prediction_timestamp"],
        "free_home_points": g["market_free"]["expected_home_points"], "free_away_points": g["market_free"]["expected_away_points"], "free_margin": g["market_free"]["expected_margin"], "free_total": g["market_free"]["expected_total"], "free_home_win_prob": g["market_free"]["home_win_probability"], "free_margin_sd": g["market_free"]["margin_sd"],
        "aware_home_points": g.get("market_aware", {}).get("expected_home_points"), "aware_away_points": g.get("market_aware", {}).get("expected_away_points"), "aware_margin": g.get("market_aware", {}).get("expected_margin"), "aware_total": g.get("market_aware", {}).get("expected_total"), "aware_home_win_prob": g.get("market_aware", {}).get("home_win_probability"),
        "market_spread": g.get("market", {}).get("spread_home_margin"), "market_total": g.get("market", {}).get("total"), "market_home_prob": g.get("market", {}).get("no_vig_home_probability"), "kicked_off_before_prediction": g["kicked_off_before_prediction"],
    } for g in forecasts])
    flat.to_parquet(folder / f"{stem}.parquet", index=False)
    flat.to_csv(folder / f"{stem}.csv", index=False)
    (folder / "latest.json").write_text(json.dumps(out, indent=2, default=str))
    return out


def render(out: dict) -> str:
    if out.get("status") != "ok":
        return json.dumps(out)
    lines = [f"NFL forecasts  season {out['season']} week {out['week']}  (model {out['model_id']}, generated {out['generated_at'][:19]}Z)", ""]
    for g in out["games"]:
        f = g["market_free"]
        h, a = g["home_team_display"], g["away_team_display"]
        lines.append(f"{a} @ {h}   kickoff {g['kickoff_utc'][:16]}Z" + ("   [kicked off before this run; not archived]" if g.get("kicked_off_before_prediction") else ""))
        lines.append(f"  Expected score:        {h} {f['expected_home_points']:.1f}  -  {a} {f['expected_away_points']:.1f}      displayed {h} {f['displayed_score']['home']} - {a} {f['displayed_score']['away']}")
        lines.append(f"  Win probability:       {h} {f['home_win_probability']*100:.1f}%   {a} {f['away_win_probability']*100:.1f}%   (tie {f['tie_probability']*100:.1f}%)")
        lines.append(f"  Expected margin:       {h} {f['expected_margin']:+.1f}     expected total {f['expected_total']:.1f}")
        lines.append(f"  80% intervals:         {h} {f['home_score_interval_80'][0]:.0f}-{f['home_score_interval_80'][1]:.0f}   {a} {f['away_score_interval_80'][0]:.0f}-{f['away_score_interval_80'][1]:.0f}   margin sd {f['margin_sd']:.1f}  total sd {f['total_sd']:.1f}")
        if g.get("market"):
            m = g["market"]
            aw = g.get("market_aware", {})
            lines.append(f"  Market spread {m['spread_home_margin']:+.1f} vs model fair {m['model_fair_spread']:+.1f} (diff {m['spread_difference']:+.1f}); market total {m['total']} vs model {m['model_fair_total']:.1f} (diff {m['total_difference']:+.1f}); market prob {m['no_vig_home_probability'] if m['no_vig_home_probability'] is None else round(m['no_vig_home_probability']*100,1)}% vs model {m['model_win_probability']*100:.1f}%")
            if aw.get("available"):
                lines.append(f"  Market-aware model:    {h} {aw['expected_home_points']:.1f} - {a} {aw['expected_away_points']:.1f}, {h} win {aw['home_win_probability']*100:.1f}%")
        drivers = g.get("drivers", {})
        if drivers.get("by_family_points"):
            fam = ", ".join(f"{k} {v:+.1f}" for k, v in list(drivers["by_family_points"].items())[:5])
            lines.append(f"  Drivers (pts to {h}): {fam}")
        lines.append("")
    return "\n".join(lines)
