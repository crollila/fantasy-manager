"""Is the forecasting engine actually learning from each week? A continuous, out-of-sample check.

For every completed week of the season, three versions of the forecast are scored on the
same games, each using only a model trained before that week:

- live:            the model the app was using that week (refit through the previous week)
                   with up-to-date team data -- what the app actually does;
- no refit:        the season's first model (never refit) with up-to-date team data, isolating
                   what the weekly refit adds;
- no new results:  the same live model but with every result-driven input (Elo, margin and
                   efficiency ratings, recent form, quarterback stats) frozen at each team's
                   state entering week 1 -- a forecaster that keeps "making the same guesses".

If the live version does not beat "no new results" as the season goes on, the engine is not
learning from the games it has seen.
"""
from __future__ import annotations
import json
import re
from datetime import datetime, timezone
import numpy as np
import polars as pl
from app.nfl.config import paths
from app.nfl.features.builder import load_features, add_matchup_columns, family_of
from app.nfl.models.registry import Registry
from app.nfl.power import team_states, predict_pairs, _side_columns, ELO_HFA
from app.storage import ROOT

FROZEN_FAMILIES = {"elo", "ratings_margin", "ratings_efficiency", "team_form_offense", "team_form_defense", "qb"}
HISTORY_FILE = ROOT / "app" / "nfl" / "reference" / "data" / "learning_audit.json"
VARIANTS = ("live", "no_refit", "no_new_results")


def frozen_rows(frame: pl.DataFrame, rows: pl.DataFrame, season: int) -> pl.DataFrame:
    """``rows`` with every result-driven side feature replaced by the team's week-1 state."""
    keys = [k for k in _side_columns(frame) if family_of(f"home_{k}") in FROZEN_FAMILIES]
    states = team_states(frame, season, 1)
    out = []
    for r in rows.to_dicts():
        for side in ("home", "away"):
            state = states.get(r[f"{side}_team"], {})
            for k in keys:
                r[f"{side}_{k}"] = state.get(k)
        diff = (r.get("home_elo_pre") or 1500.0) - (r.get("away_elo_pre") or 1500.0) + (0.0 if (r.get("ctx_neutral") or 0) > 0.5 else ELO_HFA)
        r |= {"elo_diff": diff, "elo_prob_home": 1.0 / (1.0 + 10 ** (-diff / 400.0)), "elo_margin": diff / 25.0}
        out.append(r)
    return add_matchup_columns(pl.DataFrame(out, schema=rows.schema, strict=False))


def score(p: np.ndarray, margin_pred: np.ndarray, margin: np.ndarray) -> dict | None:
    ok = margin != 0
    if not ok.any():
        return None
    y = (margin[ok] > 0).astype(float)
    q = np.clip(p[ok], 1e-4, 1 - 1e-4)
    return {"games": int(ok.sum()), "log_loss": float(-np.mean(y * np.log(q) + (1 - y) * np.log(1 - q))), "brier": float(np.mean((q - y) ** 2)),
            "accuracy": float(np.mean((q > 0.5) == (y == 1))), "margin_mae": float(np.mean(np.abs(margin - margin_pred)))}


def _through(card: dict) -> tuple[int, int] | None:
    """Last (season, week) a model was trained on: the registry stamp, or the game id named in
    the notes of refits registered before stamps existed. None means a pre-season build."""
    through = card.get("trained_through") or {}
    if through.get("season") is not None:
        return int(through["season"]), int(through["week"])
    found = re.search(r"through (\d{4})_(\d{2})_", str(card.get("notes") or ""))
    return (int(found.group(1)), int(found.group(2))) if found else None


def _trained_before(card: dict, season: int, week: int) -> bool:
    through = _through(card)
    return through is None or through < (season, week)


def learning_check(season: int | None = None) -> dict:
    registry = Registry()
    cards = [m for m in registry.read()["models"] if m.get("kind") == "game_forecast"]
    frame, _ = load_features("pregame")
    season = season or int(frame.filter(pl.col("completed"))["season"].max())
    done = frame.filter((pl.col("season") == season) & pl.col("completed") & pl.col("margin").is_not_null() & (pl.col("game_type") == "REG"))
    lineage = cards  # registration order is chronological
    loaded: dict[str, dict] = {}

    def load(model_id: str) -> dict:
        if model_id not in loaded:
            loaded[model_id] = registry.load(model_id)
        return loaded[model_id]

    # The season's first model: the latest one trained before any game of this season.
    first_week = int(done["week"].min()) if done.height else 1
    eligible = [c for c in lineage if _trained_before(c, season, first_week)]
    if not eligible or done.height == 0:
        return {"season": season, "weeks": [], "status": "no completed games or models yet", "history": _history()}
    baseline = eligible[-1]["model_id"]
    weeks = []
    pooled = {v: {"p": [], "m": [], "y": []} for v in VARIANTS}
    for week in sorted(set(done["week"].to_list())):
        rows = done.filter(pl.col("week") == week)
        before = [c for c in lineage if _trained_before(c, season, week)]
        live_id = before[-1]["model_id"] if before else baseline
        margin = rows["margin"].cast(pl.Float64).to_numpy()
        preds = {"live": predict_pairs(rows, load(live_id)), "no_refit": predict_pairs(rows, load(baseline)), "no_new_results": predict_pairs(frozen_rows(frame, rows, season), load(live_id))}
        entry = {"week": week, "games": rows.height, "live_model": live_id}
        for v, p in preds.items():
            entry[v] = score(p["home_win"].to_numpy(), p["margin"].to_numpy(), margin)
            pooled[v]["p"].append(p["home_win"].to_numpy()); pooled[v]["m"].append(p["margin"].to_numpy()); pooled[v]["y"].append(margin)
        entry["refit_probability_change"] = float(np.mean(np.abs(preds["live"]["home_win"].to_numpy() - preds["no_refit"]["home_win"].to_numpy())))
        entry["data_probability_change"] = float(np.mean(np.abs(preds["live"]["home_win"].to_numpy() - preds["no_new_results"]["home_win"].to_numpy())))
        weeks.append(entry)
    season_scores = {v: score(np.concatenate(d["p"]), np.concatenate(d["m"]), np.concatenate(d["y"])) for v, d in pooled.items()}
    return {"season": season, "generated_at": datetime.now(timezone.utc).isoformat(), "baseline_model": baseline, "weeks": weeks, "season_scores": season_scores,
            "models_this_season": [{"model_id": c["model_id"], "trained_through_week": _through(c)[1], "created_at": c.get("created_at")} for c in lineage if (_through(c) or (0,))[0] == season],
            "history": _history()}


def _history() -> dict | None:
    try:
        return json.loads(HISTORY_FILE.read_text())
    except (OSError, ValueError):
        return None


def cached_learning_check() -> dict:
    p = paths()
    feature_file = p.features / "fv1" / "games_pregame.parquet"
    try:
        key = f"{(Registry().champion('game_forecast') or {}).get('model_id')}|{feature_file.stat().st_mtime_ns}"
    except Exception:  # noqa: BLE001
        key = None
    cache = p.reports / "learning_check.json"
    if key and cache.exists():
        try:
            data = json.loads(cache.read_text())
            if data.get("cache_key") == key:
                return data | {"history": _history()}
        except ValueError:
            pass
    data = learning_check() | {"cache_key": key}
    cache.write_text(json.dumps({k: v for k, v in data.items() if k != "history"}, default=str))
    return data
