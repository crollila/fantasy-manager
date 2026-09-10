"""Named walk-forward experiments. Raw per-model predictions are persisted so later stages
(ensembling, calibration, ablations, reports) never refit on the test seasons."""
from __future__ import annotations
import json
import logging
import sys
import time
from pathlib import Path
from app.nfl.config import paths, DEVELOPMENT_TEST_SEASONS, HOLDOUT_SEASONS
from app.nfl.features.builder import load_features
from app.nfl.models.backtest import ModelSpec, run_specs

log = logging.getLogger(__name__)

DEVELOPMENT_SPECS = [
    ModelSpec("ridge_compact", "ridge", "compact"),
    ModelSpec("ridge_full", "ridge", "full"),
    ModelSpec("lgbm_compact", "lgbm", "compact"),
    ModelSpec("lgbm_full", "lgbm", "full"),
    ModelSpec("xgb_compact", "xgb", "compact"),
    ModelSpec("lgbm_core", "lgbm", "core"),
    ModelSpec("lgbm_compact_hl8", "lgbm", "compact", half_life=8.0),
    ModelSpec("ridge_compact_hl8", "ridge", "compact", half_life=8.0),
    ModelSpec("lgbm_compact_direct", "lgbm", "compact", targets=("home_score", "away_score")),
    ModelSpec("lgbm_clf_compact", "lgbm_clf", "compact", targets=("home_win",)),
    ModelSpec("logit_compact", "logit", "compact", targets=("home_win",)),
    ModelSpec("lgbm_small", "lgbm", "compact", params={"params": {"num_leaves": 7, "min_child_samples": 100, "feature_fraction": 0.25, "lambda_l2": 50.0}}),
    ModelSpec("ridge_compact_mkt", "ridge", "compact_market", market_aware=True),
    ModelSpec("lgbm_compact_mkt", "lgbm", "compact_market", market_aware=True),
    ModelSpec("lgbm_clf_compact_mkt", "lgbm_clf", "compact_market", targets=("home_win",), market_aware=True),
    # Residual-of-market models: learn (actual - closing line) from football features, add the line back.
    ModelSpec("ridge_resid_mkt", "ridge", "compact", market_aware=True, offsets={"margin": "mkt_spread", "total_points": "mkt_total"}),
    ModelSpec("lgbm_resid_mkt", "lgbm", "compact", market_aware=True, offsets={"margin": "mkt_spread", "total_points": "mkt_total"}, params={"params": {"num_leaves": 7, "min_child_samples": 100, "feature_fraction": 0.25, "lambda_l2": 50.0}}),
]

# Cheaper spec list for the early (T-6 days) horizon comparison.
EARLY_SPECS = [
    ModelSpec("ridge_compact", "ridge", "compact"),
    ModelSpec("lgbm_compact", "lgbm", "compact"),
    ModelSpec("xgb_compact", "xgb", "compact"),
    ModelSpec("logit_compact", "logit", "compact", targets=("home_win",)),
]


def run_experiment(name: str, specs: list[ModelSpec], seasons: list[int], horizon: str = "pregame", root: Path | None = None):
    p = paths(root)
    frame, meta = load_features(horizon, root)
    t0 = time.time()
    preds, diag = run_specs(frame, meta, specs, seasons, horizon=horizon)
    out = p.predictions / "historical" / f"raw_{name}_{horizon}.parquet"
    preds.to_parquet(out, index=False)
    diag["elapsed_seconds"] = round(time.time() - t0, 1)
    diag["specs"] = [s.__dict__ for s in specs]
    (p.reports / f"experiment_{name}_{horizon}.json").write_text(json.dumps(diag, indent=2, default=str))
    log.info("experiment %s: %s rows -> %s (%.0fs)", name, len(preds), out, time.time() - t0)
    return preds, diag


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    which = sys.argv[1] if len(sys.argv) > 1 else "dev"
    horizon = sys.argv[2] if len(sys.argv) > 2 else "pregame"
    specs = EARLY_SPECS if horizon == "early" else DEVELOPMENT_SPECS
    if which == "dev":
        run_experiment("dev", specs, list(DEVELOPMENT_TEST_SEASONS), horizon)
    elif which == "holdout":
        run_experiment("holdout", specs, list(HOLDOUT_SEASONS), horizon)
    elif which == "both":
        run_experiment("dev", specs, list(DEVELOPMENT_TEST_SEASONS), horizon)
        run_experiment("holdout", specs, list(HOLDOUT_SEASONS), horizon)
    else:
        raise SystemExit("usage: experiments.py dev|holdout|both [horizon]")
