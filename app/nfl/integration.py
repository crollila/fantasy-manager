"""Bridge between the forecasting engine and the existing Fantasy Manager application.

``engine_forecasts`` refreshes the engine's data (cheap when the raw store is fresh), builds
point-in-time features for the scheduled games and returns forecasts in the shape the
existing ledger (``app.tracking``) and UI already understand, with the full engine output
attached under ``engine``. Any failure degrades to ``None`` so the legacy ridge model keeps
producing picks; the failure reason is reported in the intelligence sources list.
"""
from __future__ import annotations
import logging
from datetime import datetime, timezone
import numpy as np
from app.nfl.config import paths
from app.nfl.models.registry import Registry

log = logging.getLogger(__name__)


def champion_available() -> bool:
    try:
        return Registry().champion("game_forecast") is not None
    except Exception:  # noqa: BLE001
        return False


def refresh_engine_data(season: int | None = None) -> dict:
    from app.nfl.ingest import ingest
    from app.nfl.normalize import normalize_all
    from app.nfl.features.builder import build_features
    rows = ingest(season=season)
    normalize_all()
    frame = build_features("pregame")
    return {"assets": len(rows), "problems": [r for r in rows if r["status"] not in ("ok", "unavailable")], "games": frame.height}


def engine_forecasts(game_ids: list[str], n_sims: int = 20000, refresh: bool = True, season: int | None = None) -> tuple[dict[str, dict], dict]:
    """Return {game_id: legacy-shaped forecast} for the requested upcoming games plus a status row."""
    status = {"name": "NFL forecasting engine", "status": "not configured", "url": "docs/ML_SYSTEM.md"}
    try:
        import polars as pl
        from app.nfl.predict import forecast_games
        from app.nfl.features.builder import load_features
        registry = Registry()
        champion = registry.champion("game_forecast")
        if champion is None:
            status["note"] = "No champion model registered; run `python -m app.nfl train`"
            return {}, status
        if refresh:
            refresh_engine_data(season)
        frame, meta = load_features("pregame")
        rows = frame.filter(pl.col("game_id").is_in(game_ids))
        if rows.height == 0:
            status.update(status="no games", note="Requested games are not in the schedule feature table")
            return {}, status
        artifact = registry.load(champion["model_id"])
        forecasts = forecast_games(rows, meta, artifact, n_sims=n_sims)
        out = {g["game_id"]: legacy_shape(g) for g in forecasts}
        status.update(status="ok", model_id=champion["model_id"], games=len(out), fetched_at=datetime.now(timezone.utc).isoformat(), holdout=(champion.get("metrics", {}).get("holdout", {}) or {}))
        return out, status
    except Exception as exc:  # noqa: BLE001 - never block the legacy path
        log.exception("engine forecasts failed")
        status.update(status="unavailable", error=str(exc))
        return {}, status


def legacy_shape(g: dict) -> dict:
    """Map an engine forecast onto the keys used by app.tracking / the React UI."""
    f = g["market_free"]
    home, away = f["expected_home_points"], f["expected_away_points"]
    margin, total = f["expected_margin"], f["expected_total"]
    hp, ap, tie = f["home_win_probability"], f["away_win_probability"], f["tie_probability"]
    market = g.get("market", {})
    line = market.get("spread_home_margin")
    market_total = market.get("total")
    ats = None if line is None else ("home" if margin > line else "away" if margin < line else "pass")
    ou = None if market_total is None else ("over" if total > market_total else "under" if total < market_total else "pass")
    drivers = g.get("drivers", {})
    reasons = []
    for item in drivers.get("home_positive", [])[:3]:
        reasons.append(f"{g['home_team_display']} +{item['points']:.1f}: {item['family']} ({item['feature']})")
    for item in drivers.get("home_negative", [])[:3]:
        reasons.append(f"{g['home_team_display']} {item['points']:.1f}: {item['family']} ({item['feature']})")
    return {
        "model_version": f"engine:{g['model_id']}", "home_score": round(home, 2), "away_score": round(away, 2), "home_win_probability": hp, "away_win_probability": ap, "tie_probability": tie,
        "pick": g["home_team_display"] if hp >= ap else g["away_team_display"], "margin": margin, "total": total, "margin_p10": f["margin_interval_80"][0], "margin_p90": f["margin_interval_80"][1], "margin_sd": f["margin_sd"],
        "market_home_margin": line, "market_total": market_total, "ats_pick": ats, "total_pick": ou, "market_source": market.get("source", "unavailable"),
        # Same shape the UI and the learning layer expect: per side, a baseline and named factors.
        "components": {"home": {"baseline": round(home, 2), "factors": drivers.get("by_family_points", {}), "inputs": {}}, "away": {"baseline": round(away, 2), "factors": {k: -v for k, v in drivers.get("by_family_points", {}).items()}, "inputs": {}}, "base_predictions": f.get("base_predictions", {}), "note": drivers.get("note")},
        "reasons": reasons,
        "engine": {k: v for k, v in g.items() if k not in ("drivers",)} | {"drivers": {k: v for k, v in drivers.items() if k != "note"}},
        "warning": "Independent market-free ensemble; market-aware companion and uncertainty in `engine`. No demonstrated edge over the closing line.",
    }


def latest_predictions() -> dict | None:
    import json
    path = paths().predictions / "current" / "latest.json"
    return json.loads(path.read_text()) if path.exists() else None


def engine_health() -> dict:
    """Versions of the packaged ML libraries and the registered champion (used by the frozen smoke test)."""
    import importlib
    versions = {}
    for name in ("lightgbm", "xgboost", "polars", "duckdb", "sklearn", "numpy"):
        try:
            versions[name] = importlib.import_module(name).__version__
        except Exception as exc:  # noqa: BLE001
            versions[name] = f"unavailable: {exc}"
    champion = None
    try:
        champion = Registry().champion("game_forecast")
    except Exception as exc:  # noqa: BLE001
        versions["registry_error"] = str(exc)
    return {"libraries": versions, "champion": champion["model_id"] if champion else None, "trained_through": (champion or {}).get("notes"), "data_root": str(paths().root)}


def champion_summary() -> dict | None:
    import json
    p = paths()
    path = p.reports / "champion_report.json"
    if not path.exists():
        # Packaged installs carry the champion but not the full report: summarise the registry card.
        champion = Registry().champion("game_forecast")
        if champion is None:
            return None
        metrics = champion.get("metrics", {})
        return {"model_id": champion["model_id"], "created_at": champion.get("created_at"), "notes": champion.get("notes"), "holdout": {k: v for k, v in metrics.get("holdout", {}).items() if isinstance(v, dict) and "games" in v} or {"ensemble_market_free": metrics.get("holdout", {})}, "source": "registry card"}
    rep = json.loads(path.read_text())
    hold = rep.get("holdout", {})
    keep = ("games", "score_mae", "margin_mae", "total_mae", "log_loss", "brier", "ece", "accuracy", "coverage_80_margin")
    return {"model_id": rep.get("model_id"), "holdout_seasons": rep.get("holdout_seasons"), "development_seasons": rep.get("development_seasons"),
            "holdout": {name: {k: v for k, v in r.items() if k in keep} for name, r in hold.items() if isinstance(r, dict) and "games" in r},
            "promotion": rep.get("promotion")}
