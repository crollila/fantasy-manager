"""Command line interface: ``python -m app.nfl <command>``.

setup             check dependencies and create the data folders
ingest            download / refresh raw nflverse data (immutable raw store with manifests)
normalize         build the normalized point-in-time tables
build-features    build the game-level feature matrix (pregame horizon by default)
leakage           run the automated leakage test suite
backtest          walk-forward development + holdout experiments (persisted predictions)
ablate            feature-family ablations on development seasons
train             assemble the champion (ensemble, calibration, production refit, registry)
predict           forecast upcoming games with the champion
update            ingest + normalize + build-features
update-and-predict update, then predict (the weekly one-liner)
status            data / model status
"""
from __future__ import annotations
import argparse
import json
import logging
import sys
from app.nfl.config import paths, DEVELOPMENT_TEST_SEASONS, HOLDOUT_SEASONS


def cmd_setup(args):
    p = paths()
    missing = []
    for mod in ("polars", "duckdb", "lightgbm", "xgboost", "sklearn", "scipy", "pyarrow", "httpx"):
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    print(json.dumps({"data_root": str(p.root), "missing_packages": missing, "ok": not missing}, indent=2))
    if missing:
        print("Install with: python -m pip install -r requirements.txt", file=sys.stderr)
        sys.exit(1)


def cmd_ingest(args):
    from app.nfl.ingest import ingest
    rows = ingest(season=args.season, force=args.force, datasets=set(args.datasets) if args.datasets else None)
    counts = {}
    for r in rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    print(json.dumps({"assets": len(rows), "by_status": counts, "problems": [r for r in rows if r["status"] not in ("ok", "unavailable")]}, indent=2))


def cmd_normalize(args):
    from app.nfl.normalize import normalize_all
    print(json.dumps(normalize_all(), indent=2, default=str))


def cmd_features(args):
    from app.nfl.features.builder import build_features
    for horizon in args.horizons:
        frame = build_features(horizon)
        print(f"{horizon}: {frame.height} games")


def cmd_leakage(args):
    from app.nfl.leakage import run_all
    report = run_all()
    print(json.dumps({"passed": report["passed"], "perturbation": report["perturbation"]["passed"], "outcome_screen": report["outcome_screen"]["passed"], "suspicious": report["outcome_screen"]["suspicious"][:10], "timestamps": report["timestamps"]}, indent=2))
    if not report["passed"]:
        sys.exit(1)


def cmd_backtest(args):
    from app.nfl.models.experiments import run_experiment, DEVELOPMENT_SPECS
    if args.stage in ("dev", "all"):
        run_experiment("dev", DEVELOPMENT_SPECS, list(DEVELOPMENT_TEST_SEASONS), args.horizon)
    if args.stage in ("holdout", "all"):
        run_experiment("holdout", DEVELOPMENT_SPECS, list(HOLDOUT_SEASONS), args.horizon)
    print("backtest predictions saved under", paths().predictions / "historical")


def cmd_ablate(args):
    from app.nfl.models.ablation import run_ablations
    out = run_ablations(args.seasons or None)
    for k, v in out.items():
        if isinstance(v, dict) and "margin_mae" in v:
            print(f"{k:32s} margin MAE {v['margin_mae']:.3f}  total MAE {v['total_mae']:.3f}  log loss {v.get('log_loss', float('nan')):.4f}")


def cmd_train(args):
    import pandas as pd
    from app.nfl.models.train import build_champion, ChampionConfig
    from app.nfl.models.experiments import DEVELOPMENT_SPECS
    p = paths()
    dev = pd.read_parquet(p.predictions / "historical" / "raw_dev_pregame.parquet")
    hold_path = p.predictions / "historical" / "raw_holdout_pregame.parquet"
    hold = pd.read_parquet(hold_path) if hold_path.exists() else None
    config = ChampionConfig()
    if args.config:
        config = ChampionConfig(**json.loads(open(args.config).read()))
    specs = [s for s in DEVELOPMENT_SPECS if s.name in set(config.free_bases) | set(config.aware_bases) | set(config.free_classifiers) | set(config.aware_classifiers)]
    report = build_champion(dev, hold, config, specs, register=not args.no_register)
    print(json.dumps({"model_id": report.get("model_id"), "promotion": report.get("promotion"), "holdout_market_free": {k: v for k, v in report["holdout"].get("ensemble_market_free", {}).items() if not isinstance(v, (list, dict))}, "holdout_market_aware": {k: v for k, v in report["holdout"].get("ensemble_market_aware", {}).items() if not isinstance(v, (list, dict))}, "holdout_market": {k: v for k, v in report["holdout"].get("market_closing", {}).items() if not isinstance(v, (list, dict))}}, indent=2, default=str))


def cmd_refit(args):
    from app.nfl.models.train import refit_champion
    print(json.dumps(refit_champion(), indent=2, default=str))


def cmd_predict(args):
    from app.nfl.predict import run, render
    out = run(season=args.season, week=args.week, rebuild=not args.no_rebuild, n_sims=args.sims)
    print(render(out))
    if out.get("status") == "ok":
        print("saved:", paths().predictions / "current" / f"{out['season']}_week{out['week']:02d}.json")


def cmd_update(args):
    cmd_ingest(argparse.Namespace(season=None, force=False, datasets=None))
    cmd_normalize(args)
    cmd_features(argparse.Namespace(horizons=["pregame"]))


def cmd_update_and_predict(args):
    cmd_update(args)
    if getattr(args, "refit", False):
        cmd_refit(args)
    cmd_predict(argparse.Namespace(season=None, week=None, no_rebuild=False, sims=args.sims))


def cmd_status(args):
    from app.nfl.ingest import RawStore
    from app.nfl.models.registry import Registry
    p = paths()
    raw = RawStore(p.raw).status()
    by = {}
    for r in raw:
        by.setdefault(r["dataset"], {"ok": 0, "unavailable": 0, "other": 0})
        by[r["dataset"]]["ok" if r["status"] == "ok" else "unavailable" if r["status"] == "unavailable" else "other"] += 1
    reg = Registry().read()
    print(json.dumps({"data_root": str(p.root), "raw_assets": by, "champion": reg.get("champion"), "models": [m["model_id"] for m in reg.get("models", [])], "features": [str(f) for f in (p.features).glob("*/games_*.parquet")]}, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m app.nfl", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("setup").set_defaults(func=cmd_setup)
    s = sub.add_parser("ingest"); s.add_argument("--season", type=int); s.add_argument("--force", action="store_true"); s.add_argument("--datasets", nargs="*"); s.set_defaults(func=cmd_ingest)
    sub.add_parser("normalize").set_defaults(func=cmd_normalize)
    s = sub.add_parser("build-features"); s.add_argument("--horizons", nargs="*", default=["pregame"]); s.set_defaults(func=cmd_features)
    sub.add_parser("leakage").set_defaults(func=cmd_leakage)
    s = sub.add_parser("backtest"); s.add_argument("--stage", choices=["dev", "holdout", "all"], default="all"); s.add_argument("--horizon", default="pregame"); s.set_defaults(func=cmd_backtest)
    s = sub.add_parser("ablate"); s.add_argument("--seasons", nargs="*", type=int); s.set_defaults(func=cmd_ablate)
    s = sub.add_parser("train"); s.add_argument("--config"); s.add_argument("--no-register", action="store_true"); s.set_defaults(func=cmd_train)
    sub.add_parser("refit", help="refit the champion's learners on every completed game available now (same configuration and metrics)").set_defaults(func=cmd_refit)
    s = sub.add_parser("predict"); s.add_argument("--season", type=int); s.add_argument("--week", type=int); s.add_argument("--no-rebuild", action="store_true"); s.add_argument("--sims", type=int, default=50000); s.set_defaults(func=cmd_predict)
    sub.add_parser("update").set_defaults(func=cmd_update)
    s = sub.add_parser("update-and-predict"); s.add_argument("--sims", type=int, default=50000); s.add_argument("--refit", action="store_true", help="also refit the champion on newly completed games before predicting"); s.set_defaults(func=cmd_update_and_predict)
    sub.add_parser("status").set_defaults(func=cmd_status)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args.func(args)


if __name__ == "__main__":
    main()
