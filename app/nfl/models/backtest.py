"""Walk-forward (expanding window) backtests on point-in-time features.

For every test season S the models are trained on completed games from seasons < S and
predict every game of season S. Predictions are persisted per game and per model so that
ensembles, calibration and reports are all fitted on genuinely out-of-sample values.
"""
from __future__ import annotations
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import pandas as pd
import polars as pl
from scipy.stats import norm
from app.nfl.config import paths, FEATURE_VERSION, DEVELOPMENT_TEST_SEASONS, HOLDOUT_SEASONS, MIN_TRAIN_SEASONS
from app.nfl.features.builder import load_features
from app.nfl.models.prepare import feature_sets, to_matrix, season_weights
from app.nfl.models.learners import make_learner
from app.nfl.models.metrics import evaluate

log = logging.getLogger(__name__)


@dataclass
class ModelSpec:
    name: str
    kind: str                     # ridge | lgbm | xgb | logit | lgbm_clf | baseline
    feature_set: str = "compact"  # key of prepare.feature_sets
    targets: tuple[str, ...] = ("margin", "total_points")
    params: dict = field(default_factory=dict)
    half_life: float | None = None  # season recency weighting (None = uniform)
    market_aware: bool = False
    min_season: int = 1999
    offsets: dict = field(default_factory=dict)  # target -> feature column subtracted before fitting and added back (residual-of-market models)

    @property
    def label(self) -> str:
        return self.name


DEFAULT_SPECS = [
    ModelSpec("ridge_compact", "ridge", "compact"),
    ModelSpec("lgbm_compact", "lgbm", "compact"),
    ModelSpec("lgbm_full", "lgbm", "full"),
    ModelSpec("ridge_compact_mkt", "ridge", "compact_market", market_aware=True),
    ModelSpec("lgbm_compact_mkt", "lgbm", "compact_market", market_aware=True),
]


def baseline_predictions(test: pl.DataFrame, train: pl.DataFrame) -> dict[str, np.ndarray]:
    """Analytic baselines evaluated on the test season using only training-season constants."""
    hfa = float(train.filter(~pl.col("neutral_flag"))["margin"].mean()) if "neutral_flag" in train.columns else float(train["margin"].mean())
    avg_total = float(train["total_points"].mean())
    out = {}
    n = test.height
    neutral = test["ctx_neutral"].to_numpy().astype(float)
    out["hfa__margin"] = np.where(neutral > 0, 0.0, hfa) * np.ones(n)
    out["hfa__total_points"] = np.full(n, avg_total)
    # Rolling scoring baseline: season-to-date points for/against (EWM fallback to previous season).
    hp = test["home_points__ewm"].to_numpy().astype(float)
    ha = test["home_points_allowed__ewm"].to_numpy().astype(float)
    ap = test["away_points__ewm"].to_numpy().astype(float)
    aa = test["away_points_allowed__ewm"].to_numpy().astype(float)
    league = avg_total / 2
    exp_home = np.nan_to_num((hp + aa) / 2, nan=league)
    exp_away = np.nan_to_num((ap + ha) / 2, nan=league)
    out["rolling__margin"] = exp_home - exp_away + np.where(neutral > 0, 0.0, hfa)
    out["rolling__total_points"] = exp_home + exp_away
    # Elo baseline: 25 Elo points per point of margin, scaled to fit training data.
    elo = test["elo_diff"].to_numpy().astype(float)
    tr_elo = train["elo_diff"].to_numpy().astype(float)
    tr_margin = train["margin"].to_numpy().astype(float)
    ok = np.isfinite(tr_elo)
    slope = float(np.sum(tr_elo[ok] * tr_margin[ok]) / max(np.sum(tr_elo[ok] ** 2), 1e-9))
    out["elo__margin"] = np.nan_to_num(elo * slope, nan=0.0)
    out["elo__total_points"] = np.full(n, avg_total)
    # Market baseline (closing lines): spread is the market home margin, total the market total.
    if "mkt_spread" in test.columns:
        out["market__margin"] = test["mkt_spread"].to_numpy().astype(float)
        out["market__total_points"] = test["mkt_total"].to_numpy().astype(float)
        out["market__home_win_probability"] = test["mkt_home_prob"].to_numpy().astype(float)
    return out


def run_specs(frame: pl.DataFrame, meta: dict, specs: list[ModelSpec], test_seasons: list[int], horizon: str = "pregame", log_progress: bool = True) -> tuple[pd.DataFrame, dict]:
    """Return the per-game prediction table (long-term persisted) and model diagnostics."""
    frame = frame.filter(pl.col("completed") & pl.col("margin").is_not_null()).sort(["kickoff_utc", "game_id"])
    sets = feature_sets(meta, frame.columns)
    diagnostics = {"seasons": {}, "importance": {}, "alphas": {}, "rounds": {}}
    rows = []
    for season in test_seasons:
        t0 = time.time()
        train = frame.filter(pl.col("season") < season)
        test = frame.filter(pl.col("season") == season)
        if train["season"].n_unique() < MIN_TRAIN_SEASONS or test.height == 0:
            continue
        base = pd.DataFrame({
            "game_id": test["game_id"].to_list(), "season": test["season"].to_list(), "week": test["week"].to_list(), "game_type": test["game_type"].to_list(),
            "kickoff_utc": test["kickoff_utc"].to_list(), "prediction_timestamp": test["prediction_timestamp"].to_list(), "home_team": test["home_team"].to_list(), "away_team": test["away_team"].to_list(),
            "actual_home_points": test["home_score"].to_numpy().astype(float), "actual_away_points": test["away_score"].to_numpy().astype(float),
            "market_spread": test["mkt_spread"].to_numpy().astype(float), "market_total": test["mkt_total"].to_numpy().astype(float), "market_home_prob": test["mkt_home_prob"].to_numpy().astype(float),
        })
        for k, v in baseline_predictions(test, train).items():
            base[k] = v
        if "market__home_win_probability" in base.columns:
            base["market__home_win"] = base["market__home_win_probability"]
        for spec in specs:
            cols = sets[spec.feature_set]
            tr = train.filter(pl.col("season") >= spec.min_season)
            Xtr = to_matrix(tr, cols)
            Xte = to_matrix(test, cols)
            w = season_weights(tr["season"].to_numpy(), season, spec.half_life)
            for target in spec.targets:
                ytr = tr[target].to_numpy().astype(float)
                offset_col = spec.offsets.get(target)
                off_tr = tr[offset_col].to_numpy().astype(float) if offset_col else np.zeros(len(ytr))
                off_te = test[offset_col].to_numpy().astype(float) if offset_col else np.zeros(test.height)
                if target == "home_win":
                    ok = np.isfinite(ytr) & (tr["margin"].to_numpy().astype(float) != 0)
                else:
                    ok = np.isfinite(ytr) & np.isfinite(off_tr)
                learner = make_learner(spec.kind, **spec.params)
                learner.fit(Xtr[ok], (ytr - off_tr)[ok], w[ok], tr["season"].to_numpy()[ok])
                base[f"{spec.name}__{target}"] = learner.predict(Xte) + off_te
                key = f"{spec.name}__{target}"
                if hasattr(learner, "best_alpha_"):
                    diagnostics["alphas"].setdefault(key, {})[season] = learner.best_alpha_
                if hasattr(learner, "best_rounds_"):
                    diagnostics["rounds"].setdefault(key, {})[season] = learner.best_rounds_
                if hasattr(learner, "importance") and season in (max(test_seasons), test_seasons[len(test_seasons) // 2]):
                    diagnostics["importance"].setdefault(key, {})[season] = learner.importance(cols, top=40)
        rows.append(base)
        diagnostics["seasons"][season] = {"train_games": int(train.height), "test_games": int(test.height), "seconds": round(time.time() - t0, 1)}
        if log_progress:
            log.info("season %s: %s train / %s test games in %.1fs", season, train.height, test.height, time.time() - t0)
    preds = pd.concat(rows, ignore_index=True)
    preds["feature_version"] = FEATURE_VERSION
    preds["horizon"] = horizon
    return preds, diagnostics


def derive_scores(preds: pd.DataFrame, margin_col: str, total_col: str) -> tuple[np.ndarray, np.ndarray]:
    m = preds[margin_col].to_numpy().astype(float)
    t = preds[total_col].to_numpy().astype(float)
    return (t + m) / 2, (t - m) / 2


def rolling_sigma(preds: pd.DataFrame, margin_col: str, min_seasons: int = 2, default: float = 13.5) -> np.ndarray:
    """Season-by-season margin residual sd using only earlier test seasons (out-of-sample)."""
    resid = (preds["actual_home_points"] - preds["actual_away_points"] - preds[margin_col]).to_numpy().astype(float)
    seasons = preds["season"].to_numpy()
    out = np.full(len(preds), default)
    for s in np.unique(seasons):
        prior = seasons < s
        if np.unique(seasons[prior]).size >= min_seasons:
            out[seasons == s] = np.sqrt(np.mean(resid[prior] ** 2))
    return out


def system_table(preds: pd.DataFrame, name: str, margin_col: str, total_col: str, prob_col: str | None = None, model_version: str = "") -> pd.DataFrame:
    """Standard backtest table for one forecasting system."""
    home, away = derive_scores(preds, margin_col, total_col)
    sd = rolling_sigma(preds, margin_col)
    table = preds[["game_id", "season", "week", "game_type", "kickoff_utc", "prediction_timestamp", "home_team", "away_team", "actual_home_points", "actual_away_points", "market_spread", "market_total", "market_home_prob", "feature_version", "horizon"]].copy()
    table["pred_home_points"] = home
    table["pred_away_points"] = away
    table["pred_margin"] = preds[margin_col].to_numpy().astype(float)
    table["pred_total"] = preds[total_col].to_numpy().astype(float)
    table["pred_margin_sd"] = sd
    if prob_col and prob_col in preds.columns:
        table["pred_home_win_probability"] = preds[prob_col].to_numpy().astype(float)
    else:
        table["pred_home_win_probability"] = norm.cdf(table["pred_margin"] / sd)
    table["system"] = name
    table["model_version"] = model_version or name
    return table


def season_breakdown(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for season, part in table.groupby("season"):
        rows.append({"season": int(season)} | evaluate(part))
    return pd.DataFrame(rows)


def segment_breakdown(table: pd.DataFrame) -> dict:
    t = table.copy()
    t["abs_spread"] = t["market_spread"].abs()
    segments = {
        "weeks_1_4": t[t["week"] <= 4], "weeks_5_plus": t[(t["week"] >= 5) & (t["game_type"] == "REG")], "playoffs": t[t["game_type"] != "REG"],
        "home_favorite": t[t["market_spread"] > 0], "road_favorite": t[t["market_spread"] < 0], "close_games_spread_lt_3": t[t["abs_spread"] < 3], "large_spread_ge_7": t[t["abs_spread"] >= 7],
    }
    return {name: evaluate(part) for name, part in segments.items() if len(part) >= 30}


def save_backtest(table: pd.DataFrame, name: str, root: Path | None = None) -> Path:
    p = paths(root)
    path = p.predictions / "historical" / f"backtest_{name}.parquet"
    table.to_parquet(path, index=False)
    return path


def development_holdout_split(table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    dev = table[table["season"].isin(DEVELOPMENT_TEST_SEASONS)]
    hold = table[table["season"].isin(HOLDOUT_SEASONS)]
    return dev, hold
