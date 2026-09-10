# Backtest report

Generated from `storage/nfl/reports/champion_report.json` (model `nfl-forecast-fv1-20260910T040207Z`, feature version `fv1`).

## Protocol

- Data: nflverse play-by-play, schedules, injuries, snap counts, depth charts, rosters, 7277 completed games [1999, 2026].
- Walk-forward, expanding window: for every test season the models are trained on all completed games from earlier seasons only; features for each game use only information stamped before its prediction timestamp (kickoff − 1 h).
- Development seasons (model selection, ensemble weights, calibration): 2005–2021. Locked holdout seasons (scored once with the frozen configuration): 2022, 2023, 2024, 2025.
- Hyper-parameters (ridge alpha, boosting rounds) are chosen inside each training window on its last two seasons; the test season is never touched.
- Market lines are nflverse closing lines (untimestamped). They enter only the market-aware systems and the market baseline; the market-free ensemble never sees them.
- Leakage suite: PASSED (perturbation of all post-cutoff outcomes leaves pre-cutoff features unchanged; no market-free feature correlates > 0.6 with the game's own result; rolling windows verified to exclude the current game).

## Locked holdout results (2022–2025, all games of each season)

| System | score mae | margin mae | margin rmse | total mae | log loss | brier | ece | accuracy | games |
|---|---|---|---|---|---|---|---|---|---|
| Naive: league average + home field | 7.713 | 10.712 | 13.869 | 10.708 | 0.6881 | 0.2475 | 0.0134 | 0.550 | 1139 |
| Rolling scoring (points for/against, EWM) | 7.372 | 9.978 | 12.935 | 10.522 | 0.6406 | 0.2249 | 0.0601 | 0.626 | 1139 |
| Elo (margin-aware, preseason regression) | 7.438 | 9.980 | 12.899 | 10.708 | 0.6350 | 0.2225 | 0.0355 | 0.641 | 1139 |
| Ridge, compact features | 7.302 | 9.828 | 12.701 | 10.470 | 0.6253 | 0.2183 | 0.0318 | 0.651 | 1139 |
| Ridge, full features | 7.294 | 9.812 | 12.710 | 10.444 | 0.6274 | 0.2191 | 0.0282 | 0.643 | 1139 |
| LightGBM, compact features | 7.276 | 9.780 | 12.659 | 10.416 | 0.6199 | 0.2156 | 0.0275 | 0.666 | 1139 |
| LightGBM, full features | 7.294 | 9.809 | 12.712 | 10.498 | 0.6244 | 0.2176 | 0.0177 | 0.652 | 1139 |
| XGBoost, compact features | 7.275 | 9.808 | 12.708 | 10.412 | 0.6235 | 0.2173 | 0.0338 | 0.652 | 1139 |
| LightGBM, core (no availability family) | 7.279 | 9.819 | 12.738 | 10.419 | 0.6260 | 0.2184 | 0.0309 | 0.660 | 1139 |
| LightGBM compact, 8-season recency weighting | 7.270 | 9.763 | 12.692 | 10.423 | 0.6224 | 0.2166 | 0.0214 | 0.657 | 1139 |
| Ridge compact, 8-season recency weighting | 7.276 | 9.775 | 12.706 | 10.401 | 0.6258 | 0.2182 | 0.0344 | 0.649 | 1139 |
| LightGBM direct home/away score targets | 7.278 | 9.801 | 12.708 | 10.433 | 0.6238 | 0.2175 | 0.0188 | 0.654 | 1139 |
| ENSEMBLE market-free (champion) | 7.276 | 9.783 | 12.674 | 10.418 | 0.6233 | 0.2172 | 0.0320 | 0.660 | 1139 |
| Market: closing spread / total / no-vig moneyline | 7.155 | 9.536 | 12.423 | 10.189 | 0.6070 | 0.2100 | 0.0198 | 0.676 | 1139 |
| Ridge compact + market (market-aware) | 7.272 | 9.782 | 12.629 | 10.423 | 0.6215 | 0.2166 | 0.0322 | 0.658 | 1139 |
| LightGBM compact + market (market-aware) | 7.195 | 9.646 | 12.524 | 10.291 | 0.6132 | 0.2128 | 0.0296 | 0.666 | 1139 |
| ENSEMBLE market-aware | 7.152 | 9.538 | 12.419 | 10.203 | 0.6070 | 0.2100 | 0.0280 | 0.670 | 1139 |
| lgbm_small | 7.273 | 9.797 | 12.697 | 10.397 | 0.6241 | 0.2176 | 0.0293 | 0.657 | 1139 |
| ridge_resid_mkt | 7.167 | 9.557 | 12.413 | 10.260 | 0.6066 | 0.2099 | 0.0225 | 0.675 | 1139 |
| lgbm_resid_mkt | 7.159 | 9.562 | 12.425 | 10.221 | 0.6071 | 0.2101 | 0.0287 | 0.675 | 1139 |

Paired weekly-bootstrap differences in margin MAE (negative = ensemble better):

- baseline_elo: -0.197 points, 95% CI [-0.300, -0.096] (88 weeks)
- ridge_compact: -0.045 points, 95% CI [-0.086, -0.004] (88 weeks)
- lgbm_compact: +0.002 points, 95% CI [-0.068, +0.066] (88 weeks)
- market_closing: +0.247 points, 95% CI [+0.114, +0.388] (88 weeks)
- ensemble_market_aware: +0.244 points, 95% CI [+0.126, +0.374] (88 weeks)

### Market-free ensemble by holdout season

| Season | games | score MAE | margin MAE | total MAE | log loss | Brier | accuracy |
|---|---|---|---|---|---|---|---|
| 2022 | 284 | 7.158 | 9.007 | 10.866 | 0.6194 | 0.2158 | 0.667 |
| 2023 | 285 | 7.450 | 10.324 | 10.505 | 0.6483 | 0.2283 | 0.635 |
| 2024 | 285 | 7.122 | 9.940 | 9.757 | 0.6047 | 0.2081 | 0.688 |
| 2025 | 285 | 7.372 | 9.857 | 10.544 | 0.6208 | 0.2167 | 0.651 |

### Market (closing line) by holdout season

| Season | games | score MAE | margin MAE | total MAE | log loss | Brier | accuracy |
|---|---|---|---|---|---|---|---|
| 2022 | 284 | 6.948 | 8.783 | 10.398 | 0.6050 | 0.2093 | 0.660 |
| 2023 | 285 | 7.301 | 9.984 | 10.168 | 0.6270 | 0.2186 | 0.677 |
| 2024 | 285 | 7.079 | 9.704 | 9.768 | 0.5892 | 0.2010 | 0.705 |
| 2025 | 285 | 7.290 | 9.670 | 10.423 | 0.6070 | 0.2109 | 0.662 |

### Market-aware ensemble by holdout season

| Season | games | score MAE | margin MAE | total MAE | log loss | Brier | accuracy |
|---|---|---|---|---|---|---|---|
| 2022 | 284 | 6.956 | 8.780 | 10.462 | 0.6030 | 0.2085 | 0.663 |
| 2023 | 285 | 7.309 | 10.002 | 10.207 | 0.6295 | 0.2197 | 0.670 |
| 2024 | 285 | 7.063 | 9.702 | 9.739 | 0.5870 | 0.2000 | 0.698 |
| 2025 | 285 | 7.280 | 9.666 | 10.404 | 0.6085 | 0.2117 | 0.648 |

### Segments (holdout, market-free ensemble vs closing line)

| Segment | games | model margin MAE | market margin MAE | model log loss | market log loss |
|---|---|---|---|---|---|
| weeks_1_4 | 256 | 9.846 | 9.754 | 0.6496 | 0.6415 |
| weeks_5_plus | 831 | 9.726 | 9.415 | 0.6177 | 0.5980 |
| playoffs | 52 | 10.369 | 10.404 | 0.5832 | 0.5831 |
| home_favorite | 690 | 9.812 | 9.601 | 0.6046 | 0.5927 |
| road_favorite | 449 | 9.738 | 9.435 | 0.6522 | 0.6293 |
| close_games_spread_lt_3 | 293 | 9.461 | 9.312 | 0.6992 | 0.6838 |
| large_spread_ge_7 | 292 | 10.325 | 10.116 | 0.5020 | 0.4886 |

### Calibration (holdout, market-free ensemble)

| probability bin | games | mean predicted | observed home win rate |
|---|---|---|---|
| 0.0-0.1 | 1 | 0.086 | 0.000 |
| 0.1-0.2 | 23 | 0.168 | 0.217 |
| 0.2-0.3 | 86 | 0.251 | 0.326 |
| 0.3-0.4 | 164 | 0.357 | 0.354 |
| 0.4-0.5 | 209 | 0.455 | 0.431 |
| 0.5-0.6 | 208 | 0.555 | 0.577 |
| 0.6-0.7 | 202 | 0.649 | 0.688 |
| 0.7-0.8 | 154 | 0.747 | 0.708 |
| 0.8-0.9 | 79 | 0.839 | 0.886 |
| 0.9-1.0 | 10 | 0.917 | 1.000 |

Expected calibration error 0.0320; 80% margin interval coverage 0.817; 50% coverage 0.563; CRPS (margin) 7.093.

## Development seasons (2005–2021, out-of-sample walk-forward, used for selection)

| System | score mae | margin mae | margin rmse | total mae | log loss | brier | ece | accuracy | games |
|---|---|---|---|---|---|---|---|---|---|
| Naive: league average + home field | 8.169 | 11.483 | 14.815 | 11.118 | 0.6853 | 0.2461 | 0.0097 | 0.564 | 4559 |
| Rolling scoring (points for/against, EWM) | 7.746 | 10.738 | 13.790 | 10.832 | 0.6351 | 0.2223 | 0.0396 | 0.644 | 4559 |
| Elo (margin-aware, preseason regression) | 7.835 | 10.683 | 13.668 | 11.118 | 0.6277 | 0.2192 | 0.0163 | 0.646 | 4559 |
| Ridge, compact features | 7.608 | 10.571 | 13.555 | 10.711 | 0.6227 | 0.2168 | 0.0276 | 0.653 | 4559 |
| Ridge, full features | 7.639 | 10.608 | 13.605 | 10.744 | 0.6249 | 0.2177 | 0.0275 | 0.651 | 4559 |
| LightGBM, compact features | 7.645 | 10.612 | 13.620 | 10.725 | 0.6245 | 0.2176 | 0.0315 | 0.654 | 4559 |
| LightGBM, full features | 7.662 | 10.632 | 13.616 | 10.757 | 0.6268 | 0.2186 | 0.0216 | 0.646 | 4559 |
| XGBoost, compact features | 7.641 | 10.605 | 13.599 | 10.735 | 0.6239 | 0.2173 | 0.0308 | 0.653 | 4559 |
| LightGBM, core (no availability family) | 7.646 | 10.607 | 13.604 | 10.758 | 0.6251 | 0.2178 | 0.0190 | 0.651 | 4559 |
| LightGBM compact, 8-season recency weighting | 7.651 | 10.626 | 13.633 | 10.747 | 0.6254 | 0.2180 | 0.0241 | 0.648 | 4559 |
| Ridge compact, 8-season recency weighting | 7.613 | 10.582 | 13.573 | 10.711 | 0.6228 | 0.2168 | 0.0244 | 0.655 | 4559 |
| LightGBM direct home/away score targets | 7.655 | 10.613 | 13.633 | 10.745 | 0.6257 | 0.2181 | 0.0182 | 0.649 | 4559 |
| ENSEMBLE market-free (champion) | 7.604 | 10.554 | 13.536 | 10.691 | 0.6196 | 0.2155 | 0.0158 | 0.656 | 4559 |
| Market: closing spread / total / no-vig moneyline | 7.502 | 10.395 | 13.359 | 10.598 | 0.6101 | 0.2114 | 0.0208 | 0.660 | 4559 |
| Ridge compact + market (market-aware) | 7.589 | 10.542 | 13.514 | 10.691 | 0.6204 | 0.2157 | 0.0285 | 0.658 | 4559 |
| LightGBM compact + market (market-aware) | 7.585 | 10.530 | 13.503 | 10.669 | 0.6182 | 0.2147 | 0.0260 | 0.658 | 4559 |
| ENSEMBLE market-aware | 7.498 | 10.395 | 13.354 | 10.578 | 0.6096 | 0.2110 | 0.0232 | 0.663 | 4559 |
| lgbm_small | 7.637 | 10.601 | 13.595 | 10.735 | 0.6241 | 0.2174 | 0.0233 | 0.653 | 4559 |
| ridge_resid_mkt | 7.512 | 10.424 | 13.367 | 10.639 | 0.6098 | 0.2111 | 0.0225 | 0.663 | 4559 |
| lgbm_resid_mkt | 7.519 | 10.436 | 13.386 | 10.628 | 0.6103 | 0.2113 | 0.0227 | 0.663 | 4559 |

### ensemble_market_free

- Margin blend: {"ridge_compact__margin": 0.561, "lgbm_compact__margin": 0.141, "xgb_compact__margin": 0.0, "lgbm_compact_hl8__margin": 0.122, "lgbm_small__margin": 0.0, "elo__margin": 0.176} (constrained_blend; dev MAE 10.554 vs simple average 10.568)
- Total blend: {"ridge_compact__total_points": 0.483, "lgbm_compact__total_points": 0.319, "xgb_compact__total_points": 0.116, "lgbm_compact_hl8__total_points": 0.0, "lgbm_small__total_points": 0.0, "elo__total_points": 0.082} (constrained_blend)
- Win probability source: margin_only (dev log loss by candidate: {"margin_only": 0.6189, "classifier_only": 0.6221, "blend": 0.6197}); calibrator: platt {"identity": 0.6189, "platt": 0.6169, "isotonic": 0.6215}
- Residuals: margin sd 13.531 (recent 13.174), total sd 13.497, margin/total residual correlation 0.024, Student-t df 21.8, tie rate 0.0024

### ensemble_market_aware

- Margin blend: {"market__margin": 0.863, "ridge_resid_mkt__margin": 0.0, "lgbm_resid_mkt__margin": 0.0, "ridge_compact_mkt__margin": 0.039, "lgbm_compact_mkt__margin": 0.098} (constrained_blend; dev MAE 10.395 vs simple average 10.419)
- Total blend: {"market__total_points": 0.665, "ridge_resid_mkt__total_points": 0.0, "lgbm_resid_mkt__total_points": 0.0, "ridge_compact_mkt__total_points": 0.149, "lgbm_compact_mkt__total_points": 0.186} (constrained_blend)
- Win probability source: blend (dev log loss by candidate: {"margin_only": 0.6117, "classifier_only": 0.6126, "blend": 0.6116}); calibrator: platt {"identity": 0.6116, "platt": 0.6104, "isotonic": 0.6131}
- Residuals: margin sd 13.353 (recent 13.074), total sd 13.348, margin/total residual correlation 0.028, Student-t df 19.7, tie rate 0.0024

## Feature-family ablations (LightGBM compact, development seasons)

Reference (all families, LightGBM compact, development seasons): margin MAE 10.612, total MAE 10.725, log loss 0.6245.

| Removed family | margin MAE | Δ vs all (+ = worse) | 95% CI of Δ (weekly bootstrap) | total MAE | log loss | verdict |
|---|---|---|---|---|---|---|
| availability | 10.633 | +0.021 | [-0.015, 0.055] | 10.746 | 0.6257 | neutral / within noise |
| qb | 10.647 | +0.035 | [-0.005, 0.075] | 10.776 | 0.6271 | neutral / within noise |
| elo | 10.604 | -0.008 | [-0.042, 0.025] | 10.729 | 0.6250 | neutral / within noise |
| ratings_margin | 10.605 | -0.007 | [-0.042, 0.025] | 10.739 | 0.6243 | neutral / within noise |
| ratings_efficiency | 10.612 | +0.000 | [-0.035, 0.036] | 10.744 | 0.6244 | neutral / within noise |
| team_form_offense | 10.615 | +0.003 | [-0.038, 0.045] | 10.735 | 0.6262 | neutral / within noise |
| team_form_defense | 10.596 | -0.016 | [-0.052, 0.020] | 10.747 | 0.6246 | neutral / within noise |
| matchup | 10.615 | +0.003 | [-0.032, 0.035] | 10.731 | 0.6250 | neutral / within noise |
| rest_travel | 10.616 | +0.004 | [-0.027, 0.036] | 10.738 | 0.6250 | neutral / within noise |
| weather | 10.619 | +0.006 | [-0.024, 0.038] | 10.755 | 0.6250 | neutral / within noise |
| coaching | 10.610 | -0.002 | [-0.035, 0.029] | 10.736 | 0.6243 | neutral / within noise |
| officials | 10.621 | +0.009 | [-0.023, 0.040] | 10.765 | 0.6241 | neutral / within noise |
| schedule | 10.598 | -0.015 | [-0.045, 0.017] | 10.744 | 0.6243 | neutral / within noise |

## Information horizon: pregame (T−1h) versus early week (T−6 days)

| Horizon | stage | ridge margin MAE | XGBoost margin MAE | Elo margin MAE | blend margin MAE | blend total MAE | blend log loss | market margin MAE |
|---|---|---|---|---|---|---|---|---|
| pregame | development 2005–2021 | 10.571 | 10.605 | 10.683 | 10.555 | 10.691 | 0.6217 | 10.395 |
| pregame | holdout 2022–2025 | 9.828 | 9.808 | 9.980 | 9.786 | 10.418 | 0.6234 | 9.536 |
| early | development 2005–2021 | 10.616 | 10.632 | 10.683 | 10.595 | 10.696 | 0.6249 | 10.395 |
| early | holdout 2022–2025 | 9.913 | 9.899 | 9.980 | 9.886 | 10.481 | 0.6312 | 9.536 |

The early horizon uses no in-week injury reports, the previous game's starting quarterback and no market line; the pregame horizon uses reports timestamped before kickoff − 1 h and the scheduled starter. The blend here is a ridge/XGBoost/LightGBM/Elo margin blend fitted on the development seasons of the same horizon (the champion adds further members); log loss uses a Student-t margin distribution without Platt calibration.

## Limitations

- Weather features are the observed game conditions from nflverse (temperature, wind), not archived pregame forecasts; a live forecast is used in production. Backtest weather is therefore slightly optimistic relative to a true T-24h forecast (the ablation quantifies how much weather matters at all).
- Market lines are closing lines without timestamps, so the market-aware backtest is a 'closing-line horizon' evaluation; earlier-horizon lines are not available historically without a paid archive.
- Injury reports before 2010 and in 2025+ carry no modification timestamp; they are assumed public 48 hours before kickoff (true for the pregame horizon, where all reports are public).
- Play-by-play from 1999–2005 lacks air yards / CPOE / expected pass; those features are missing (not zero) for those seasons and the models handle missingness explicitly.
- Player-level charting (FTN 2022+, participation 2016+, PFR 2018+, NGS 2016+) is ingested and normalized but not yet part of the champion feature set; see ML_SYSTEM.md for the ranked backlog.