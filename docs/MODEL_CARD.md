# Model card: nfl-forecast-fv1-20260910T085639Z

- Created: 2026-09-10T08:56:39.598898+00:00; git commit 15a31bd; feature version fv1; dataset fingerprint b9a80eec2794f9f9
- Kind: game-level NFL score / margin / total / win-probability forecaster (market-free champion plus a market-aware companion system)
- Training data: nflverse 1999–2025 completed games (regular season and playoffs); production learners refit on all completed games through the latest season; ensemble weights and calibrators fitted on out-of-fold walk-forward predictions for 2005–2021.

## Architecture

- Base learners (market-free): ridge_compact, lgbm_compact, xgb_compact, lgbm_compact_hl8, lgbm_small, elo; classifiers for win probability: logit_compact
- Base learners (market-aware): market, ridge_resid_mkt, lgbm_resid_mkt, ridge_compact_mkt, lgbm_compact_mkt; classifiers: market, lgbm_clf_compact_mkt
- Margin blend weights: {"ridge_compact__margin": 0.561, "lgbm_compact__margin": 0.141, "xgb_compact__margin": 0.0, "lgbm_compact_hl8__margin": 0.122, "lgbm_small__margin": 0.0, "elo__margin": 0.176}; total blend weights: {"ridge_compact__total_points": 0.483, "lgbm_compact__total_points": 0.319, "xgb_compact__total_points": 0.116, "lgbm_compact_hl8__total_points": 0.0, "lgbm_small__total_points": 0.0, "elo__total_points": 0.082}
- Win probability: margin_only with platt calibration; Student-t margin distribution (df 21.8) with heteroscedastic sd; 50,000-draw Monte Carlo from the joint out-of-fold residual pool for score distributions, intervals and spread/total probabilities.
- Hyper-parameters: LightGBM (learning rate 0.02, 15 leaves, min 50 samples per leaf, feature fraction 0.4, bagging 0.8, L2 20, early stopping on the last two training seasons then refit); ridge with alpha chosen on the same internal chronological split from {30 … 100000}; XGBoost is evaluated as a challenger only.

## Holdout performance (2022–2025, never used for selection)

| System | score MAE | margin MAE | total MAE | log loss | Brier | ECE | accuracy | games |
|---|---|---|---|---|---|---|---|---|
| Market-free ensemble (champion) | 7.276 | 9.783 | 10.418 | 0.6233 | 0.2172 | 0.0320 | 0.660 | 1139 |
| Market-aware ensemble | 7.152 | 9.538 | 10.203 | 0.6070 | 0.2100 | 0.0280 | 0.670 | 1139 |
| Closing line (market) | 7.155 | 9.536 | 10.189 | 0.6070 | 0.2100 | 0.0198 | 0.676 | 1139 |
| Elo baseline | 7.438 | 9.980 | 10.708 | 0.6350 | 0.2225 | 0.0355 | 0.641 | 1139 |

## Intended use

- Pregame forecasts of NFL games (expected score, margin, total, win probability, score distributions) for research, lineup context and forecast tracking inside Fantasy Manager.
- The market-free system is the independent football model; the market-aware system is the best pure forecast when a line exists. Neither is a demonstrated betting edge: the market-free model trails the closing line on margin MAE and the market-aware model is roughly at market accuracy.

## Known weaknesses

- Week 1–4 forecasts lean on prior-season strength, coaching and QB priors; roster turnover beyond the QB is captured only through snap-continuity proxies.
- Player-level availability uses official injury reports and reserve lists; late-breaking news, in-game injuries and undisclosed limitations are not modelled.
- Weather in production is a city-level forecast; historical evaluation used observed conditions.
- Totals are harder than margins for every system including the market; treat total forecasts as ±10 points at 1 sd.
- Any single season can deviate materially from the long-run averages above; see BACKTEST_REPORT.md for season-level dispersion.