"""Feature-family ablations on development seasons (walk-forward, one family removed at a time)."""
from __future__ import annotations
import json
import logging
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from app.nfl.config import paths, DEVELOPMENT_TEST_SEASONS
from app.nfl.features.builder import load_features
from app.nfl.models.backtest import ModelSpec, run_specs, system_table
from app.nfl.models.metrics import evaluate, bootstrap_difference
from app.nfl.models import prepare

log = logging.getLogger(__name__)

FAMILIES = ["availability", "qb", "elo", "ratings_margin", "ratings_efficiency", "team_form_offense", "team_form_defense", "matchup", "rest_travel", "weather", "coaching", "officials", "schedule"]


def run_ablations(seasons: list[int] | None = None, kind: str = "lgbm", base_set: str = "compact", root: Path | None = None) -> dict:
    p = paths(root)
    frame, meta = load_features("pregame", root)
    seasons = seasons or list(DEVELOPMENT_TEST_SEASONS)
    original = prepare.feature_sets
    results = {}
    t0 = time.time()
    full_preds, _ = run_specs(frame, meta, [ModelSpec("full", kind, base_set)], seasons, log_progress=False)
    full_table = system_table(full_preds, "full", "full__margin", "full__total_points")
    results["all_families"] = evaluate(full_table)
    fam_cols = {f: set(cols) for f, cols in meta["families"].items()}
    weeks = (full_table["season"].astype(str) + "_" + full_table["week"].astype(str)).to_numpy()
    base_err = np.abs((full_table["actual_home_points"] - full_table["actual_away_points"]) - full_table["pred_margin"]).to_numpy()
    for family in FAMILIES:
        drop = fam_cols.get(family, set())
        # Diff/matchup columns that are built from this family are removed too.
        derived = {c for c in meta["families"].get("matchup", []) + [c for cols in meta["families"].values() for c in cols if c.startswith("diff_")] if any(k in c for k in _keys(family))}

        def patched(meta_, columns, _drop=drop | derived):
            sets = original(meta_, columns)
            return {k: [c for c in v if c not in _drop] for k, v in sets.items()}

        prepare.feature_sets = patched
        try:
            import app.nfl.models.backtest as bt
            bt.feature_sets = patched
            preds, _ = run_specs(frame, meta, [ModelSpec("abl", kind, base_set)], seasons, log_progress=False)
        finally:
            prepare.feature_sets = original
            bt.feature_sets = original
        table = system_table(preds, "abl", "abl__margin", "abl__total_points")
        err = np.abs((table["actual_home_points"] - table["actual_away_points"]) - table["pred_margin"]).to_numpy()
        results[f"without_{family}"] = evaluate(table) | {"removed_columns": len(drop | derived), "bootstrap_margin_mae_vs_all": bootstrap_difference(err, base_err, weeks)}
        log.info("ablation without %s: margin MAE %.3f (all: %.3f) in %.0fs", family, results[f"without_{family}"]["margin_mae"], results["all_families"]["margin_mae"], time.time() - t0)
    results["elapsed_seconds"] = round(time.time() - t0, 1)
    (p.reports / "ablations.json").write_text(json.dumps(results, indent=2, default=str))
    return results


def _keys(family: str) -> list[str]:
    return {"availability": ["value_lost", "inj_lost", "starters_", "reserve_", "qb_starter_"], "qb": ["qb_"], "elo": ["elo"], "ratings_margin": ["srs"], "ratings_efficiency": ["off_adj", "def_adj", "mx_"], "team_form_offense": ["__ewm", "__std", "__prev", "__l8", "__l3"], "team_form_defense": ["_allowed__"],
            "rest_travel": ["rest", "travel", "tz_shift"], "weather": ["wx_"], "coaching": ["coach_"], "officials": ["ref_"], "schedule": ["ctx_"], "matchup": ["mx_"]}.get(family, [family])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    seasons = [int(s) for s in sys.argv[1:]] or None
    out = run_ablations(seasons)
    for k, v in out.items():
        if isinstance(v, dict) and "margin_mae" in v:
            print(f"{k:32s} margin MAE {v['margin_mae']:.3f}  total MAE {v['total_mae']:.3f}  log loss {v.get('log_loss', float('nan')):.4f}")
