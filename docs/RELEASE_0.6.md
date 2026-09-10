# Fantasy Manager 0.6.0

NFL game predictions now come from a point-in-time forecasting engine (`app/nfl/`) built on 27 seasons of public nflverse data. Every game gets expected points for both teams, expected margin and total, a calibrated win probability, a 50,000-draw score distribution with intervals, spread and total probabilities, the strongest drivers, and a market-aware companion forecast when a line exists.

- Trained on 7,277 completed games from 1999 through the first game of 2026, with 2,100 candidate features per game (play-by-play efficiency, opponent-adjusted ratings, Elo, quarterback quality, injury value lost, rest/travel, weather, coaching, officials, matchups).
- Accuracy demonstrated out of sample with season-by-season walk-forward backtests. Locked 2022–2025 holdout (1,139 games): margin MAE 9.78, total MAE 10.42, log loss 0.623, winner accuracy 66.0%, versus Elo 9.98 / 0.635 / 64.1% and the closing line 9.54 / 0.607 / 67.6%. The market-aware system matches the market. No betting edge is claimed.
- Automated leakage tests (future outcomes perturbed, pre-cutoff features must not change), a champion/challenger registry with a bootstrap promotion rule, and a refit command that keeps the champion trained through the most recent completed game.
- The trained champion ships inside the installer and seeds the app's data folder on first start. The first refresh downloads the nflverse raw data (about 550 MB, once) and builds the point-in-time features; later refreshes are incremental.
- The NFL games screen shows engine win probabilities, score ranges, the market-aware forecast and drivers; picks are archived in the existing pregame ledger, so accuracy tracking, reviews and the correction layer are unchanged. The previous ridge model is stored alongside as `legacy_forecast` and remains the fallback.
- New API endpoints `/api/nfl/predictions` and `/api/nfl/model`; new documentation: `docs/ML_SYSTEM.md`, `docs/BACKTEST_REPORT.md`, `docs/MODEL_CARD.md`, `docs/DATA_SOURCES.md`, `docs/FEATURES.md`.

Install `Fantasy-Manager-Setup-0.6.0.exe` over the previous version. Saved teams, the ESPN profile and existing prediction history are retained. The Windows x64 installer includes the analysis engine (now with LightGBM, XGBoost, Polars and DuckDB); Python and Node are not required. It is unsigned, as prior releases were.
