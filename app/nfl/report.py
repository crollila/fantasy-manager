"""Render Markdown reports (BACKTEST_REPORT.md, MODEL_CARD.md) from saved JSON results."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
from app.nfl.config import paths, DEVELOPMENT_TEST_SEASONS, HOLDOUT_SEASONS, FEATURE_VERSION

SYSTEM_LABELS = {
    "baseline_hfa": "Naive: league average + home field", "baseline_rolling": "Rolling scoring (points for/against, EWM)", "baseline_elo": "Elo (margin-aware, preseason regression)", "market_closing": "Market: closing spread / total / no-vig moneyline",
    "ridge_compact": "Ridge, compact features", "ridge_full": "Ridge, full features", "lgbm_compact": "LightGBM, compact features", "lgbm_full": "LightGBM, full features", "xgb_compact": "XGBoost, compact features", "lgbm_core": "LightGBM, core (no availability family)",
    "lgbm_compact_hl8": "LightGBM compact, 8-season recency weighting", "ridge_compact_hl8": "Ridge compact, 8-season recency weighting", "lgbm_compact_direct": "LightGBM direct home/away score targets",
    "ridge_compact_mkt": "Ridge compact + market (market-aware)", "lgbm_compact_mkt": "LightGBM compact + market (market-aware)", "ensemble_market_free": "ENSEMBLE market-free (champion)", "ensemble_market_aware": "ENSEMBLE market-aware",
}


def _fmt(v, nd=3):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "—"
    return f"{v:.{nd}f}" if isinstance(v, (int, float)) else str(v)


def metrics_table(block: dict, keys=("score_mae", "margin_mae", "margin_rmse", "total_mae", "log_loss", "brier", "ece", "accuracy", "games")) -> str:
    header = "| System | " + " | ".join(k.replace("_", " ") for k in keys) + " |\n|---|" + "---|" * len(keys) + "\n"
    order = ["baseline_hfa", "baseline_rolling", "baseline_elo", "ridge_compact", "ridge_full", "lgbm_compact", "lgbm_full", "xgb_compact", "lgbm_core", "lgbm_compact_hl8", "ridge_compact_hl8", "lgbm_compact_direct", "ensemble_market_free", "market_closing", "ridge_compact_mkt", "lgbm_compact_mkt", "ensemble_market_aware"]
    rows = []
    for name in order + [n for n in block if n not in order]:
        r = block.get(name)
        if not isinstance(r, dict) or "games" not in r:
            continue
        rows.append(f"| {SYSTEM_LABELS.get(name, name)} | " + " | ".join(_fmt(r.get(k), 0 if k == "games" else 4 if k in ("log_loss", "brier", "ece") else 3) for k in keys) + " |")
    return header + "\n".join(rows) + "\n"


def season_table(rows: list[dict]) -> str:
    out = "| Season | games | score MAE | margin MAE | total MAE | log loss | Brier | accuracy |\n|---|---|---|---|---|---|---|---|\n"
    for r in rows:
        out += f"| {r['season']} | {r['games']} | {_fmt(r.get('score_mae'))} | {_fmt(r.get('margin_mae'))} | {_fmt(r.get('total_mae'))} | {_fmt(r.get('log_loss'), 4)} | {_fmt(r.get('brier'), 4)} | {_fmt(r.get('accuracy'))} |\n"
    return out


def segment_table(segments: dict, market: dict | None) -> str:
    out = "| Segment | games | model margin MAE | market margin MAE | model log loss | market log loss |\n|---|---|---|---|---|---|\n"
    for name, r in segments.items():
        m = (market or {}).get(name, {})
        out += f"| {name} | {r['games']} | {_fmt(r.get('margin_mae'))} | {_fmt(m.get('margin_mae'))} | {_fmt(r.get('log_loss'), 4)} | {_fmt(m.get('log_loss'), 4)} |\n"
    return out


def reliability_md(rows: list[dict]) -> str:
    out = "| probability bin | games | mean predicted | observed home win rate |\n|---|---|---|---|\n"
    for r in rows:
        out += f"| {r['bin']} | {r['n']} | {r['predicted']:.3f} | {r['observed']:.3f} |\n"
    return out


def ablation_md(ab: dict) -> str:
    base = ab.get("all_families", {})
    out = f"Reference (all families, LightGBM compact, development seasons): margin MAE {_fmt(base.get('margin_mae'))}, total MAE {_fmt(base.get('total_mae'))}, log loss {_fmt(base.get('log_loss'), 4)}.\n\n"
    out += "| Removed family | margin MAE | Δ vs all (+ = worse) | 95% CI of Δ (weekly bootstrap) | total MAE | log loss | verdict |\n|---|---|---|---|---|---|---|\n"
    for k, v in ab.items():
        if not k.startswith("without_"):
            continue
        d = v["margin_mae"] - base["margin_mae"]
        ci = v.get("bootstrap_margin_mae_vs_all", {}).get("ci95", [None, None])
        verdict = "adds value" if ci[0] is not None and ci[0] > 0 else "hurts (remove candidate)" if ci[1] is not None and ci[1] < 0 else "neutral / within noise"
        out += f"| {k.removeprefix('without_')} | {_fmt(v['margin_mae'])} | {d:+.3f} | [{_fmt(ci[0])}, {_fmt(ci[1])}] | {_fmt(v['total_mae'])} | {_fmt(v.get('log_loss'), 4)} | {verdict} |\n"
    return out


def horizon_comparison(root: Path | None = None) -> str:
    """Pregame (T-1h) versus early (T-6 days) accuracy from the persisted raw predictions."""
    import numpy as np
    from app.nfl.models.backtest import system_table
    from app.nfl.models.metrics import evaluate
    from app.nfl.models.ensemble import fit_weights, apply_weights
    p = paths(root)
    files = {h: [p.predictions / "historical" / f"raw_{stage}_{h}.parquet" for stage in ("dev", "holdout")] for h in ("pregame", "early")}
    if not all(f.exists() for fs in files.values() for f in fs):
        return ""
    out = "| Horizon | stage | ridge margin MAE | XGBoost margin MAE | Elo margin MAE | blend margin MAE | blend total MAE | blend log loss | market margin MAE |\n|---|---|---|---|---|---|---|---|---|\n"
    for horizon, (dev_path, hold_path) in files.items():
        dev, hold = pd.read_parquet(dev_path), pd.read_parquet(hold_path)
        cols = [c for c in ("ridge_compact__margin", "xgb_compact__margin", "lgbm_compact__margin", "elo__margin") if c in dev.columns]
        tcols = [c.replace("__margin", "__total_points") for c in cols]
        w = fit_weights(dev, cols, (dev["actual_home_points"] - dev["actual_away_points"]).to_numpy())
        wt = fit_weights(dev, tcols, (dev["actual_home_points"] + dev["actual_away_points"]).to_numpy())
        for stage, frame in (("development 2005–2021", dev), ("holdout 2022–2025", hold)):
            frame = frame.copy()
            frame["blend__margin"] = apply_weights(frame, w)
            frame["blend__total_points"] = apply_weights(frame, wt)
            m = {}
            for base in ("ridge_compact", "xgb_compact", "elo", "blend", "market"):
                if f"{base}__margin" in frame.columns:
                    m[base] = evaluate(system_table(frame, base, f"{base}__margin", f"{base}__total_points"))
            out += f"| {horizon} | {stage} | {_fmt(m.get('ridge_compact', {}).get('margin_mae'))} | {_fmt(m.get('xgb_compact', {}).get('margin_mae'))} | {_fmt(m.get('elo', {}).get('margin_mae'))} | {_fmt(m.get('blend', {}).get('margin_mae'))} | {_fmt(m.get('blend', {}).get('total_mae'))} | {_fmt(m.get('blend', {}).get('log_loss'), 4)} | {_fmt(m.get('market', {}).get('margin_mae'))} |\n"
    out += "\nThe early horizon uses no in-week injury reports, the previous game's starting quarterback and no market line; the pregame horizon uses reports timestamped before kickoff − 1 h and the scheduled starter. The blend here is a ridge/XGBoost/LightGBM/Elo margin blend fitted on the development seasons of the same horizon (the champion adds further members); log loss uses a Student-t margin distribution without Platt calibration.\n"
    return out


def build_backtest_report(root: Path | None = None) -> str:
    p = paths(root)
    rep = json.loads((p.reports / "champion_report.json").read_text())
    ab_path = p.reports / "ablations.json"
    ab = json.loads(ab_path.read_text()) if ab_path.exists() else {}
    dq = json.loads((p.reports / "data_quality.json").read_text()) if (p.reports / "data_quality.json").exists() else {}
    leak = json.loads((p.reports / "leakage_report.json").read_text()) if (p.reports / "leakage_report.json").exists() else {}
    hold = rep.get("holdout", {})
    dev = rep.get("development", {})
    free = hold.get("ensemble_market_free", {})
    aware = hold.get("ensemble_market_aware", {})
    market = hold.get("market_closing", {})
    md = [f"# Backtest report\n", f"Generated from `storage/nfl/reports/champion_report.json` (model `{rep.get('model_id')}`, feature version `{FEATURE_VERSION}`).\n",
          "## Protocol\n",
          f"- Data: nflverse play-by-play, schedules, injuries, snap counts, depth charts, rosters, {dq.get('completed_games', '—')} completed games {dq.get('seasons', '')}.",
          "- Walk-forward, expanding window: for every test season the models are trained on all completed games from earlier seasons only; features for each game use only information stamped before its prediction timestamp (kickoff − 1 h).",
          f"- Development seasons (model selection, ensemble weights, calibration): {DEVELOPMENT_TEST_SEASONS[0]}–{DEVELOPMENT_TEST_SEASONS[-1]}. Locked holdout seasons (scored once with the frozen configuration): {', '.join(map(str, HOLDOUT_SEASONS))}.",
          "- Hyper-parameters (ridge alpha, boosting rounds) are chosen inside each training window on its last two seasons; the test season is never touched.",
          "- Market lines are nflverse closing lines (untimestamped). They enter only the market-aware systems and the market baseline; the market-free ensemble never sees them.",
          f"- Leakage suite: {'PASSED' if leak.get('passed') else 'see leakage_report.json'} (perturbation of all post-cutoff outcomes leaves pre-cutoff features unchanged; no market-free feature correlates > 0.6 with the game's own result; rolling windows verified to exclude the current game).\n",
          "## Locked holdout results (2022–2025, all games of each season)\n", metrics_table(hold),
          ]
    if "bootstrap_free_vs_baseline_elo" in hold:
        md.append("Paired weekly-bootstrap differences in margin MAE (negative = ensemble better):\n")
        for k in ("bootstrap_free_vs_baseline_elo", "bootstrap_free_vs_ridge_compact", "bootstrap_free_vs_lgbm_compact", "bootstrap_free_vs_market_closing", "bootstrap_free_vs_ensemble_market_aware"):
            if k in hold:
                b = hold[k]
                md.append(f"- {k.removeprefix('bootstrap_free_vs_')}: {b['mean_difference']:+.3f} points, 95% CI [{b['ci95'][0]:+.3f}, {b['ci95'][1]:+.3f}] ({b['groups']} weeks)")
        md.append("")
    if free.get("by_season"):
        md += ["### Market-free ensemble by holdout season\n", season_table(free["by_season"])]
        if market.get("by_season"):
            md += ["### Market (closing line) by holdout season\n", season_table(market["by_season"])]
        if aware.get("by_season"):
            md += ["### Market-aware ensemble by holdout season\n", season_table(aware["by_season"])]
    if free.get("segments"):
        md += ["### Segments (holdout, market-free ensemble vs closing line)\n", segment_table(free["segments"], market.get("segments"))]
    if free.get("reliability"):
        md += ["### Calibration (holdout, market-free ensemble)\n", reliability_md(free["reliability"]), f"Expected calibration error {_fmt(free.get('ece'), 4)}; 80% margin interval coverage {_fmt(free.get('coverage_80_margin'))}; 50% coverage {_fmt(free.get('coverage_50_margin'))}; CRPS (margin) {_fmt(free.get('crps_margin'))}.\n"]
    md += ["## Development seasons (2005–2021, out-of-sample walk-forward, used for selection)\n", metrics_table(dev)]
    sysinfo = rep.get("systems", {})
    for name in ("ensemble_market_free", "ensemble_market_aware"):
        s = sysinfo.get(name, {})
        if s:
            md.append(f"### {name}\n")
            md.append(f"- Margin blend: {json.dumps({k: round(v, 3) for k, v in s['weights_margin']['weights'].items()})} ({s['weights_margin'].get('chosen')}; dev MAE {_fmt(s['weights_margin']['loss'])} vs simple average {_fmt(s['weights_margin']['simple_average_loss'])})")
            md.append(f"- Total blend: {json.dumps({k: round(v, 3) for k, v in s['weights_total']['weights'].items()})} ({s['weights_total'].get('chosen')})")
            md.append(f"- Win probability source: {s['probability_source']} (dev log loss by candidate: {json.dumps({k: round(v, 4) for k, v in s['probability_scores'].items()})}); calibrator: {s['calibrator']['method']} {json.dumps({k: round(v, 4) for k, v in s['calibrator_info'].get('oos_log_loss', {}).items()})}")
            r = s.get("residual_summary", {})
            md.append(f"- Residuals: margin sd {_fmt(r.get('margin_sd'))} (recent {_fmt(r.get('margin_sd_recent'))}), total sd {_fmt(r.get('total_sd'))}, margin/total residual correlation {_fmt(r.get('residual_corr'))}, Student-t df {_fmt(r.get('t_df'), 1)}, tie rate {_fmt(r.get('tie_rate'), 4)}\n")
    if ab:
        md += ["## Feature-family ablations (LightGBM compact, development seasons)\n", ablation_md(ab)]
    horizons = horizon_comparison(root)
    if horizons:
        md += ["## Information horizon: pregame (T−1h) versus early week (T−6 days)\n", horizons]
    md += ["## Limitations\n",
           "- Weather features are the observed game conditions from nflverse (temperature, wind), not archived pregame forecasts; a live forecast is used in production. Backtest weather is therefore slightly optimistic relative to a true T-24h forecast (the ablation quantifies how much weather matters at all).",
           "- Market lines are closing lines without timestamps, so the market-aware backtest is a 'closing-line horizon' evaluation; earlier-horizon lines are not available historically without a paid archive.",
           "- Injury reports before 2010 and in 2025+ carry no modification timestamp; they are assumed public 48 hours before kickoff (true for the pregame horizon, where all reports are public).",
           "- Play-by-play from 1999–2005 lacks air yards / CPOE / expected pass; those features are missing (not zero) for those seasons and the models handle missingness explicitly.",
           "- Player-level charting (FTN 2022+, participation 2016+, PFR 2018+, NGS 2016+) is ingested and normalized but not yet part of the champion feature set; see ML_SYSTEM.md for the ranked backlog.",
           ]
    text = "\n".join(md)
    return text


def build_model_card(root: Path | None = None) -> str:
    p = paths(root)
    rep = json.loads((p.reports / "champion_report.json").read_text())
    reg = json.loads((p.models / "registry.json").read_text())
    champ_id = reg.get("champion", {}).get("game_forecast")
    record = next((m for m in reg.get("models", []) if m["model_id"] == champ_id), {})
    hold = rep.get("holdout", {})
    free, aware, market, elo = hold.get("ensemble_market_free", {}), hold.get("ensemble_market_aware", {}), hold.get("market_closing", {}), hold.get("baseline_elo", {})
    sysfree = rep.get("systems", {}).get("ensemble_market_free", {})
    sysaware = rep.get("systems", {}).get("ensemble_market_aware", {})
    specs = rep.get("config", {})
    md = [f"# Model card: {champ_id}\n",
          f"- Created: {record.get('created_at')}; git commit {record.get('git_commit')}; feature version {record.get('feature_version')}; dataset fingerprint {record.get('dataset_fingerprint')}",
          f"- Kind: game-level NFL score / margin / total / win-probability forecaster (market-free champion plus a market-aware companion system)",
          f"- Training data: nflverse 1999–2025 completed games (regular season and playoffs); production learners refit on all completed games through the latest season; ensemble weights and calibrators fitted on out-of-fold walk-forward predictions for {DEVELOPMENT_TEST_SEASONS[0]}–{DEVELOPMENT_TEST_SEASONS[-1]}.\n",
          "## Architecture\n",
          f"- Base learners (market-free): {', '.join(specs.get('free_bases', []))}; classifiers for win probability: {', '.join(specs.get('free_classifiers', []))}",
          f"- Base learners (market-aware): {', '.join(specs.get('aware_bases', []))}; classifiers: {', '.join(specs.get('aware_classifiers', []))}",
          f"- Margin blend weights: {json.dumps({k: round(v, 3) for k, v in sysfree.get('weights_margin', {}).get('weights', {}).items()})}; total blend weights: {json.dumps({k: round(v, 3) for k, v in sysfree.get('weights_total', {}).get('weights', {}).items()})}",
          f"- Win probability: {sysfree.get('probability_source')} with {sysfree.get('calibrator', {}).get('method')} calibration; Student-t margin distribution (df {_fmt(sysfree.get('residual_summary', {}).get('t_df'), 1)}) with heteroscedastic sd; 50,000-draw Monte Carlo from the joint out-of-fold residual pool for score distributions, intervals and spread/total probabilities.",
          "- Hyper-parameters: LightGBM (learning rate 0.02, 15 leaves, min 50 samples per leaf, feature fraction 0.4, bagging 0.8, L2 20, early stopping on the last two training seasons then refit); ridge with alpha chosen on the same internal chronological split from {30 … 100000}; XGBoost is evaluated as a challenger only.\n",
          "## Holdout performance (2022–2025, never used for selection)\n",
          "| System | score MAE | margin MAE | total MAE | log loss | Brier | ECE | accuracy | games |\n|---|---|---|---|---|---|---|---|---|",
          ]
    for name, r in (("Market-free ensemble (champion)", free), ("Market-aware ensemble", aware), ("Closing line (market)", market), ("Elo baseline", elo)):
        if r:
            md.append(f"| {name} | {_fmt(r.get('score_mae'))} | {_fmt(r.get('margin_mae'))} | {_fmt(r.get('total_mae'))} | {_fmt(r.get('log_loss'), 4)} | {_fmt(r.get('brier'), 4)} | {_fmt(r.get('ece'), 4)} | {_fmt(r.get('accuracy'))} | {r.get('games')} |")
    md += ["\n## Intended use\n",
           "- Pregame forecasts of NFL games (expected score, margin, total, win probability, score distributions) for research, lineup context and forecast tracking inside Fantasy Manager.",
           "- The market-free system is the independent football model; the market-aware system is the best pure forecast when a line exists. Neither is a demonstrated betting edge: the market-free model trails the closing line on margin MAE and the market-aware model is roughly at market accuracy.\n",
           "## Known weaknesses\n",
           "- Week 1–4 forecasts lean on prior-season strength, coaching and QB priors; roster turnover beyond the QB is captured only through snap-continuity proxies.",
           "- Player-level availability uses official injury reports and reserve lists; late-breaking news, in-game injuries and undisclosed limitations are not modelled.",
           "- Weather in production is a city-level forecast; historical evaluation used observed conditions.",
           "- Totals are harder than margins for every system including the market; treat total forecasts as ±10 points at 1 sd.",
           "- Any single season can deviate materially from the long-run averages above; see BACKTEST_REPORT.md for season-level dispersion.",
           ]
    return "\n".join(md)


def write_reports(root: Path | None = None, docs: Path | None = None) -> None:
    docs = docs or (Path(__file__).resolve().parents[2] / "docs")
    (docs / "BACKTEST_REPORT.md").write_text(build_backtest_report(root), encoding="utf-8")
    (docs / "MODEL_CARD.md").write_text(build_model_card(root), encoding="utf-8")


if __name__ == "__main__":
    write_reports()
    print("wrote docs/BACKTEST_REPORT.md and docs/MODEL_CARD.md")
