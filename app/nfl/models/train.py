"""Champion assembly: base models -> ensemble -> uncertainty -> calibration -> registry.

All ensemble weights, residual distributions and probability calibrators are fitted on
development-season out-of-fold predictions. The holdout seasons are scored once with the
frozen configuration. Production artifacts are then refit on every completed game.
"""
from __future__ import annotations
import json
import logging
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import polars as pl
from app.nfl.config import paths, FEATURE_VERSION, DEVELOPMENT_TEST_SEASONS, HOLDOUT_SEASONS, SEED
from app.nfl.features.builder import load_features
from app.nfl.models.backtest import ModelSpec, run_specs, system_table, season_breakdown, segment_breakdown, save_backtest
from app.nfl.models.ensemble import fit_weights, apply_weights
from app.nfl.models.calibration import fit_residual_model, margin_sd, win_probability_from_margin, select_calibrator, ProbabilityCalibrator
from app.nfl.models.metrics import evaluate, bootstrap_difference, reliability_table, log_loss
from app.nfl.models.prepare import feature_sets, to_matrix, season_weights
from app.nfl.models.learners import make_learner
from app.nfl.models.registry import Registry

log = logging.getLogger(__name__)


@dataclass
class ChampionConfig:
    name: str = "nfl-forecast"
    # Base members may be learned specs or analytic baselines ('elo', 'market') whose columns the
    # backtest and the live predictor both provide.
    free_bases: tuple[str, ...] = ("ridge_compact", "lgbm_compact", "xgb_compact", "lgbm_compact_hl8", "lgbm_small", "elo")
    aware_bases: tuple[str, ...] = ("market", "ridge_resid_mkt", "lgbm_resid_mkt", "ridge_compact_mkt", "lgbm_compact_mkt")
    free_classifiers: tuple[str, ...] = ("logit_compact",)
    aware_classifiers: tuple[str, ...] = ("market", "lgbm_clf_compact_mkt")
    prob_blend: float = 0.5       # weight on classifier probability vs margin-derived probability
    specs: list = field(default_factory=list)


def _target_cols(bases: tuple[str, ...], target: str) -> list[str]:
    return [f"{b}__{target}" for b in bases]


def fit_system(raw: pd.DataFrame, bases: tuple[str, ...], classifiers: tuple[str, ...], prob_blend: float, label: str) -> dict:
    """Fit ensemble weights, residual model and calibrator on OOF predictions of the development seasons."""
    dev = raw[raw["season"].isin(DEVELOPMENT_TEST_SEASONS)].copy()
    margin = (dev["actual_home_points"] - dev["actual_away_points"]).to_numpy().astype(float)
    total = (dev["actual_home_points"] + dev["actual_away_points"]).to_numpy().astype(float)
    m_cols = [c for c in _target_cols(bases, "margin") if c in dev.columns]
    t_cols = [c for c in _target_cols(bases, "total_points") if c in dev.columns]
    w_margin = fit_weights(dev, m_cols, margin)
    w_total = fit_weights(dev, t_cols, total)
    # Simple average must be beaten, otherwise use it.
    for w in (w_margin, w_total):
        if w["simple_average_loss"] <= w["loss"] + 1e-4:
            k = len(w["columns"])
            w["weights"] = {c: 1.0 / k for c in w["columns"]}
            w["chosen"] = "simple_average"
        else:
            w["chosen"] = "constrained_blend"
    dev["ens_margin"] = apply_weights(dev, w_margin)
    dev["ens_total"] = apply_weights(dev, w_total)
    residual = fit_residual_model(dev, "ens_margin", "ens_total")
    # Probability: margin-derived (t distribution, heteroscedastic sd) optionally blended with a classifier.
    sd = margin_sd(residual, dev["ens_total"].to_numpy())
    p_margin = win_probability_from_margin(dev["ens_margin"].to_numpy(), sd, residual["t_df"], residual["tie_rate"])
    clf_cols = [f"{c}__home_win" for c in classifiers if f"{c}__home_win" in dev.columns]
    decided = margin != 0
    y = (margin > 0).astype(float)
    seasons = dev["season"].to_numpy()
    candidates = {"margin_only": p_margin}
    if clf_cols:
        p_clf = np.nanmean(dev[clf_cols].to_numpy().astype(float), axis=1)
        p_clf = np.where(np.isfinite(p_clf), p_clf, p_margin)
        candidates["classifier_only"] = p_clf
        candidates["blend"] = prob_blend * p_clf + (1 - prob_blend) * p_margin
    # Chronological selection: fit nothing, just compare OOF log loss on the later half of the dev seasons.
    later = seasons >= np.unique(seasons)[len(np.unique(seasons)) // 2]
    scores = {k: log_loss(v[decided & later], y[decided & later]) for k, v in candidates.items()}
    best = min(scores, key=scores.get)
    calibrator, cal_info = select_calibrator(candidates[best][decided], y[decided], seasons[decided])
    dev["ens_prob"] = calibrator.predict(candidates[best])
    return {"label": label, "bases": list(bases), "classifiers": clf_cols, "weights_margin": w_margin, "weights_total": w_total, "residual": residual, "probability_source": best, "probability_scores": scores, "calibrator": calibrator.to_dict(), "calibrator_info": cal_info, "dev_table": dev}


def apply_system(raw: pd.DataFrame, system: dict) -> pd.DataFrame:
    frame = raw.copy()
    frame["ens_margin"] = apply_weights(frame, system["weights_margin"])
    frame["ens_total"] = apply_weights(frame, system["weights_total"])
    residual = system["residual"]
    sd = margin_sd(residual, frame["ens_total"].to_numpy())
    p_margin = win_probability_from_margin(frame["ens_margin"].to_numpy(), sd, residual["t_df"], residual["tie_rate"])
    if system["probability_source"] != "margin_only" and system["classifiers"]:
        p_clf = np.nanmean(frame[system["classifiers"]].to_numpy().astype(float), axis=1)
        p_clf = np.where(np.isfinite(p_clf), p_clf, p_margin)
        p = p_clf if system["probability_source"] == "classifier_only" else 0.5 * p_clf + 0.5 * p_margin
    else:
        p = p_margin
    calibrator = ProbabilityCalibrator.from_dict(system["calibrator"])
    frame["ens_prob"] = calibrator.predict(p)
    frame["ens_margin_sd"] = sd
    return frame


def evaluate_system(frame: pd.DataFrame, name: str, model_version: str) -> tuple[pd.DataFrame, dict]:
    table = system_table(frame, name, "ens_margin", "ens_total", prob_col="ens_prob", model_version=model_version)
    table["pred_margin_sd"] = frame["ens_margin_sd"].to_numpy()
    return table, evaluate(table)


def comparison_tables(raw: pd.DataFrame, systems: dict[str, dict]) -> dict[str, pd.DataFrame]:
    """Backtest tables for baselines, base models and ensembles (same games)."""
    out = {}
    for name, m, t, p in [("baseline_hfa", "hfa__margin", "hfa__total_points", None), ("baseline_rolling", "rolling__margin", "rolling__total_points", None), ("baseline_elo", "elo__margin", "elo__total_points", None), ("market_closing", "market__margin", "market__total_points", "market__home_win_probability")]:
        if m in raw.columns:
            out[name] = system_table(raw, name, m, t, prob_col=p)
    for col in raw.columns:
        if col.endswith("__margin") and not col.startswith(("hfa", "rolling", "elo", "market")):
            base = col[: -len("__margin")]
            if f"{base}__total_points" in raw.columns:
                out[base] = system_table(raw, base, col, f"{base}__total_points")
    if "lgbm_compact_direct__home_score" in raw.columns:
        d = raw.copy()
        d["direct__margin"] = d["lgbm_compact_direct__home_score"] - d["lgbm_compact_direct__away_score"]
        d["direct__total_points"] = d["lgbm_compact_direct__home_score"] + d["lgbm_compact_direct__away_score"]
        out["lgbm_compact_direct"] = system_table(d, "lgbm_compact_direct", "direct__margin", "direct__total_points")
    for name, system in systems.items():
        applied = apply_system(raw, system)
        out[name], _ = evaluate_system(applied, name, name)
    return out


def train_production_models(config: ChampionConfig, specs: list[ModelSpec], root: Path | None = None) -> dict:
    """Refit every base learner on all completed games for live predictions."""
    frame, meta = load_features("pregame", root)
    done = frame.filter(pl.col("completed") & pl.col("margin").is_not_null())
    sets = feature_sets(meta, done.columns)
    latest = int(done["season"].max())
    learners = {}
    # Analytic baseline members: Elo slope (points per Elo point) and the league-average total.
    elo = done["elo_diff"].to_numpy().astype(float)
    margin = done["margin"].to_numpy().astype(float)
    ok = np.isfinite(elo)
    baselines = {"elo_slope": float(np.sum(elo[ok] * margin[ok]) / max(np.sum(elo[ok] ** 2), 1e-9)), "avg_total": float(done["total_points"].mean())}
    for spec in specs:
        cols = sets[spec.feature_set]
        X = to_matrix(done, cols)
        w = season_weights(done["season"].to_numpy(), latest + 1, spec.half_life)
        for target in spec.targets:
            y = done[target].to_numpy().astype(float)
            offset_col = spec.offsets.get(target)
            off = done[offset_col].to_numpy().astype(float) if offset_col else np.zeros(len(y))
            ok = np.isfinite(y) & np.isfinite(off) & ((done["margin"].to_numpy().astype(float) != 0) if target == "home_win" else True)
            learner = make_learner(spec.kind, **spec.params)
            learner.fit(X[ok], (y - off)[ok], w[ok], done["season"].to_numpy()[ok])
            learners[f"{spec.name}__{target}"] = {"learner": learner, "columns": cols, "feature_set": spec.feature_set, "kind": spec.kind, "market_aware": spec.market_aware, "offset": offset_col}
            log.info("production fit %s__%s on %s games", spec.name, target, int(ok.sum()))
    return {"learners": learners, "baselines": baselines, "trained_through_season": latest, "training_games": int(done.height), "feature_version": FEATURE_VERSION}


def refit_champion(root: Path | None = None) -> dict:
    """Refresh the champion's production learners on every completed game available today.

    The configuration (members, ensemble weights, residual model, calibrator) and the
    out-of-sample evaluation are unchanged; only the learners are refit, so the refreshed
    artifact inherits the champion's metrics and is promoted as the same champion lineage.
    """
    registry = Registry(root)
    champion = registry.champion("game_forecast")
    if champion is None:
        raise RuntimeError("No champion registered; run `python -m app.nfl train` first")
    artifact = registry.load(champion["model_id"])
    config = ChampionConfig(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in artifact["config"].items()})
    specs = [ModelSpec(**s) for s in artifact["specs"]]
    models = train_production_models(config, specs, root)
    frame, _ = load_features("pregame", root)
    done = frame.filter(pl.col("completed") & pl.col("margin").is_not_null())
    latest = done.sort("kickoff_utc").tail(1).to_dicts()[0]
    model_id = f"{config.name}-{FEATURE_VERSION}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    new_artifact = {"model_id": model_id, "systems": artifact["systems"], "models": models, "specs": artifact["specs"], "config": artifact["config"], "refit_of": champion["model_id"],
                    "trained_through": {"season": latest["season"], "week": latest["week"], "game_id": latest["game_id"], "kickoff_utc": str(latest["kickoff_utc"]), "games": int(done.height)}}
    notes = f"Production refit of {champion['model_id']} on {done.height} completed games through {latest['game_id']} (same configuration and out-of-sample metrics)."
    registry.register(model_id, "game_forecast", new_artifact, champion["metrics"], champion["config"], notes=notes)
    registry.promote("game_forecast", model_id, notes)
    return {"model_id": model_id, "refit_of": champion["model_id"], "trained_through": new_artifact["trained_through"], "training_games": models["training_games"]}


def build_champion(raw_dev: pd.DataFrame, raw_holdout: pd.DataFrame | None, config: ChampionConfig, specs: list[ModelSpec], root: Path | None = None, register: bool = True) -> dict:
    p = paths(root)
    t0 = time.time()
    systems = {
        "ensemble_market_free": fit_system(raw_dev, config.free_bases, config.free_classifiers, config.prob_blend, "market-free"),
        "ensemble_market_aware": fit_system(raw_dev, config.aware_bases, config.aware_classifiers, config.prob_blend, "market-aware"),
    }
    report = {"config": asdict(config), "development_seasons": list(DEVELOPMENT_TEST_SEASONS), "holdout_seasons": list(HOLDOUT_SEASONS), "systems": {}, "development": {}, "holdout": {}}
    dev_tables = comparison_tables(raw_dev, systems)
    for name, table in dev_tables.items():
        report["development"][name] = evaluate(table)
    for name, system in systems.items():
        report["systems"][name] = {k: v for k, v in system.items() if k not in ("dev_table", "residual")} | {"residual_summary": {k: v for k, v in system["residual"].items() if k != "residual_pool"}}
    combined_raw = raw_dev
    if raw_holdout is not None and len(raw_holdout):
        hold_tables = comparison_tables(raw_holdout, systems)
        for name, table in hold_tables.items():
            report["holdout"][name] = evaluate(table)
            report["holdout"][name]["by_season"] = season_breakdown(table).to_dict(orient="records")
            report["holdout"][name]["segments"] = segment_breakdown(table)
        # Paired weekly bootstrap of ensemble vs strongest simple baseline and vs the market.
        ens = hold_tables["ensemble_market_free"]
        for other in ("baseline_elo", "market_closing", "ensemble_market_aware", "lgbm_compact", "ridge_compact"):
            if other in hold_tables:
                o = hold_tables[other]
                a = np.abs((ens["actual_home_points"] - ens["actual_away_points"]) - ens["pred_margin"]).to_numpy()
                b = np.abs((o["actual_home_points"] - o["actual_away_points"]) - o["pred_margin"]).to_numpy()
                groups = (ens["season"].astype(str) + "_" + ens["week"].astype(str)).to_numpy()
                report["holdout"][f"bootstrap_free_vs_{other}"] = bootstrap_difference(a, b, groups)
        combined_raw = pd.concat([raw_dev, raw_holdout], ignore_index=True)
        for name, table in hold_tables.items():
            save_backtest(pd.concat([dev_tables[name], table], ignore_index=True), name, root)
        # Calibration diagnostics on holdout for the ensembles.
        for name in systems:
            t = hold_tables[name]
            decided = (t["actual_home_points"] != t["actual_away_points"]).to_numpy()
            report["holdout"][name]["reliability"] = reliability_table(t["pred_home_win_probability"].to_numpy()[decided], (t["actual_home_points"] > t["actual_away_points"]).to_numpy()[decided].astype(float))
    else:
        for name, table in dev_tables.items():
            save_backtest(table, name, root)
    # Production refit: residual pool and calibrator use every OOF season (dev + holdout) so live
    # uncertainty reflects the most recent scoring environment; weights stay as selected on dev.
    production_systems = {}
    for name, system in systems.items():
        refreshed = dict(system)
        applied = apply_system(combined_raw, system)
        refreshed["residual"] = fit_residual_model(applied, "ens_margin", "ens_total")
        production_systems[name] = {k: v for k, v in refreshed.items() if k != "dev_table"}
    models = train_production_models(config, specs, root)
    model_id = f"{config.name}-{FEATURE_VERSION}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    artifact = {"model_id": model_id, "systems": production_systems, "models": models, "specs": [asdict(s) for s in specs], "config": asdict(config)}
    metrics = {"development": {k: {m: v for m, v in r.items() if not isinstance(v, (list, dict))} for k, r in report["development"].items()}, "holdout": {k: {m: v for m, v in r.items() if not isinstance(v, (list, dict))} for k, r in report["holdout"].items() if isinstance(r, dict) and "games" in r}}
    if raw_holdout is not None and len(raw_holdout):
        metrics["holdout_summary"] = report["holdout"]["ensemble_market_free"]
    report["model_id"] = model_id
    report["elapsed_seconds"] = round(time.time() - t0, 1)
    (p.reports / "champion_report.json").write_text(json.dumps(report, indent=2, default=str))
    if register:
        registry = Registry(root)
        champion = registry.champion("game_forecast")
        record = registry.register(model_id, "game_forecast", artifact, metrics | {"holdout": {k: v for k, v in metrics.get("holdout", {}).get("ensemble_market_free", {}).items()}}, {"config": asdict(config), "specs": [asdict(s) for s in specs]})
        challenger = {"holdout": metrics.get("holdout", {}).get("ensemble_market_free", {})}
        if champion is not None:
            # Same holdout games: compare on stored metrics; bootstrap uses the saved backtest tables when available.
            try:
                prev = pd.read_parquet(p.predictions / "historical" / "backtest_ensemble_market_free.parquet")
                prev = prev[prev["season"].isin(HOLDOUT_SEASONS)]
                cur = hold_tables["ensemble_market_free"].set_index("game_id").loc[prev["game_id"]]
                a = np.abs((cur["actual_home_points"] - cur["actual_away_points"]) - cur["pred_margin"]).to_numpy()
                b = np.abs((prev["actual_home_points"] - prev["actual_away_points"]) - prev["pred_margin"]).to_numpy()
                challenger["bootstrap_vs_champion"] = bootstrap_difference(a, b, (prev["season"].astype(str) + "_" + prev["week"].astype(str)).to_numpy())
            except Exception as exc:  # noqa: BLE001
                log.warning("bootstrap vs champion unavailable: %s", exc)
        ok, reason = Registry.challenger_beats_champion(challenger, champion)
        if ok:
            registry.promote("game_forecast", model_id, reason)
        report["promotion"] = {"promoted": ok, "reason": reason, "model_id": model_id, "previous_champion": champion["model_id"] if champion else None}
        (p.reports / "champion_report.json").write_text(json.dumps(report, indent=2, default=str))
    return report
