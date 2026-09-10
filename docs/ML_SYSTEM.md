# NFL game forecasting engine

`app/nfl/` turns public nflverse data into calibrated pregame forecasts for every NFL game:
expected points for each team, expected margin and total, win probabilities, score
distributions with intervals, and probabilities for common spreads and totals. Two systems
are maintained side by side:

- **Market-free** (the champion): football, player, schedule and environment information only. This is the independent model; it never sees a sportsbook number.
- **Market-aware**: the same features plus the closing spread, total and no-vig moneyline probability. It is the strongest pure forecast when a line exists and the yardstick for whether the football model adds anything to the market.

Results: [BACKTEST_REPORT.md](BACKTEST_REPORT.md) (walk-forward out-of-sample results, baselines,
ablations, calibration) and [MODEL_CARD.md](MODEL_CARD.md) (the current champion). Inputs:
[DATA_SOURCES.md](DATA_SOURCES.md). Feature definitions: [FEATURES.md](FEATURES.md).

## Pipeline

```
raw (immutable nflverse downloads + manifests)          app/nfl/ingest.py
  -> normalized point-in-time tables                     app/nfl/normalize.py
       games, plays/<season>, team_games, qb_games, kicker_games,
       injuries, snaps, depth_charts, rosters, players
  -> feature factory (per game, per horizon)             app/nfl/features/
       rolling.py     windows over prior games (offense + opponent-allowed)
       ratings.py     Elo, ridge margin ratings, opponent-adjusted efficiency (weekly snapshots)
       qb.py          expected starter, shrunk QB quality, change flags
       availability.py injuries/reserve lists -> expected value lost by position group
       context.py     rest, travel, body clock, stadium, weather, coaching, officials, market
       builder.py     home/away blocks, differences, matchup interactions, family metadata
  -> models                                              app/nfl/models/
       learners.py    ridge / logistic, LightGBM, XGBoost wrappers (internal chronological tuning)
       backtest.py    expanding-window walk-forward, persisted per-game predictions
       ensemble.py    constrained blend on out-of-fold predictions (must beat simple average)
       calibration.py residual model (heteroscedastic sd, Student-t, joint residual pool), Platt/isotonic
       simulate.py    Monte Carlo from the joint residual pool -> distributions, intervals, spreads/totals
       train.py       champion assembly, holdout scoring, production refit, registry promotion
       registry.py    champion/challenger registry with artifacts, metrics, git commit, fingerprints
       ablation.py    feature-family ablations
  -> predictions                                         app/nfl/predict.py, app/nfl/integration.py
       predictions/current/<season>_week<NN>.{json,parquet,csv}, API /api/nfl/predictions,
       forecasts archived in the existing ledger (app/tracking.py) before kickoff
```

Storage (all under `storage/nfl/`, gitignored; `NFL_DATA_DIR` overrides):
`raw/`, `normalized/`, `features/<feature_version>/`, `models/registry.json` + `models/artifacts/<model_id>/`,
`predictions/historical/` (persisted backtest tables), `predictions/current/`, `reports/`.
`app/nfl/query.py` exposes every table as a DuckDB view.

## Point-in-time discipline

- Every feature row carries `prediction_timestamp = kickoff − horizon`. Rolling windows use games with an earlier kickoff (`shift(1)` inside each team's chronological sequence). Rating snapshots use games that kicked off before the first kickoff of the week (deliberately conservative). Injury reports must have `date_modified <= prediction_timestamp`; player values come from an as-of join on the kickoff of the last game played. The expected starting quarterback at the pregame horizon is the scheduled starter (public with inactives ~90 minutes before kickoff).
- Horizons: `pregame` (T−1h, the reference) and `early` (T−6 days: no in-week injury reports, previous starter, no market). Both matrices are built by `python -m app.nfl build-features --horizons pregame early`.
- Automated leakage tests (`python -m app.nfl leakage`): perturbing every outcome after a cutoff leaves every pre-cutoff feature bit-identical across all eight feature blocks; no market-free feature correlates > 0.6 with the game's own result; rolling windows verified against the previous game; prediction timestamps precede kickoff.
- Evaluation: expanding-window walk-forward by season. Development seasons 2005–2021 for every selection decision (feature sets, learners, ensemble weights, probability source, calibrator); holdout seasons 2022–2025 scored once with the frozen configuration. Hyper-parameters are chosen inside each training window on its last two seasons. No random shuffling anywhere.

## Learned components: audit

| Component | Problem / target | Inputs (exist at prediction time) | Baseline it must beat | Validation | Value | Conf. | Difficulty | Data | Decision |
|---|---|---|---|---|---|---|---|---|---|
| Opponent-adjusted efficiency ratings (ridge) | Team offense/defense strength on EPA, success, points/drive | Prior team-games, weekly snapshots | Raw rolling EPA | Ablation of the family; walk-forward | 8 | 8 | 3 | 9 | IMPLEMENT |
| Elo / margin ridge ratings | Overall strength with offseason regression | Prior results | Home-field constant | Baseline rows in the backtest | 7 | 9 | 2 | 10 | IMPLEMENT (Elo is also a baseline) |
| Team form windows (l1…l16, std, prev, ewm) | Recency of ~120 PBP metrics | Prior team-games | Season-to-date only | Ablation; window mix left to the learners | 7 | 8 | 3 | 9 | IMPLEMENT |
| QB quality with shrinkage | Starter contribution beyond team ratings | Prior QB dropbacks, schedule starter | Team ratings alone | Ablation (qb family) | 8 | 8 | 4 | 9 | IMPLEMENT |
| Availability = expected value lost | Injury / reserve impact | Timestamped reports, snap shares, depth charts | Model without the family (core tier) | Ablation; `lgbm_core` challenger | 6 | 6 | 6 | 7 (2009+/2012+) | IMPLEMENT, coverage-flagged |
| Rest / travel / body clock / stadium | Schedule context | Schedule, static venue table | Model without the family | Ablation | 4 | 7 | 2 | 10 | IMPLEMENT |
| Weather (observed) | Totals / passing environment | nflverse temp/wind (observed) | Model without the family | Ablation; production uses a forecast | 4 | 5 | 2 | 7 | IMPLEMENT with horizon caveat |
| Coaching tenure / change | Continuity | Schedule coach names | Model without the family | Ablation | 3 | 6 | 2 | 10 | IMPLEMENT (cheap) |
| Officials tendencies | Penalty / pace environment | Referee, prior games | Model without the family | Ablation | 2 | 4 | 2 | 10 | RETAIN AS EXPERIMENTAL (must not gain material weight without OOS evidence) |
| Margin / total regressors (ridge, LightGBM, XGBoost) | Expected margin and total | Feature sets | Elo, rolling scoring, home-field constant | Walk-forward, bootstrap vs baselines | 9 | 9 | 4 | 9 | IMPLEMENT (ensemble members chosen on dev OOF) |
| Direct home/away score regressors | Alternative decomposition | Same | Margin/total decomposition | Walk-forward comparison | 4 | 5 | 3 | 9 | RETAIN AS EXPERIMENTAL unless it wins |
| Win-probability classifiers (logistic, LightGBM) | P(home win) | Same | Probability derived from the margin distribution | Log loss / Brier / ECE, chronological calibrator selection | 6 | 7 | 3 | 9 | IMPLEMENT if it lowers OOF log loss, else margin-derived |
| Ensemble weights | Combine base models | OOF predictions | Simple average, best single | Dev OOF MAE; holdout confirmation | 6 | 8 | 2 | 9 | IMPLEMENT (falls back to simple average automatically) |
| Residual / uncertainty model | Score distributions, intervals | OOF residuals | Constant normal sd | CRPS, interval coverage, calibration table | 7 | 8 | 3 | 9 | IMPLEMENT |
| Market-aware model | Best forecast given a line | + closing line | The closing line itself | Walk-forward vs market baseline | 7 | 9 | 2 | 8 (closing only) | IMPLEMENT as a separate system |
| Player-level charting (NGS, FTN, participation, PFR) | Scheme/matchup detail | 2016+/2018+/2022+ only | Current champion | Enriched-tier challenger | 5 | 4 | 7 | 4 | NEEDS MORE DATA / next step (ingested, not in champion) |
| Deep / sequence models | — | — | Gradient boosting | — | 2 | 2 | 8 | 5 | NOT WORTH ML at ~7k games |
| NLP on news | Late availability news | Timestamped articles (not archived) | Official reports | — | 4 | 3 | 8 | 2 | NEEDS MORE DATA (architecture hook: convert to availability probabilities with provenance) |

What the first full run showed (details in BACKTEST_REPORT.md): every learned system beats
Elo, rolling scoring and the naive baseline out of sample; ridge on the compact set is the
strongest single market-free member on the development seasons and LightGBM on the holdout,
so the blend (ridge 0.56, LightGBM 0.14 + 0.12, Elo 0.18) is kept for robustness; win
probabilities derived from the margin distribution beat dedicated classifiers; the closing
line remains ~0.25 points of margin MAE better than the football-only model; "features plus
market line" learners were worse than the line itself until modelled as residual-of-market,
after which the market-aware ensemble matches the market. Removing the QB or availability
families made LightGBM worse (+0.035 / +0.021 margin MAE, directionally consistent but inside
the weekly-bootstrap noise band); no family was harmful enough to remove; in-week information
(pregame vs T−6 days) is worth ~0.05–0.10 points of margin MAE.

## Update cycle

`python -m app.nfl update-and-predict` = ingest (TTL-aware, manifests) → normalize (per-season
aggregates cached; only changed seasons re-aggregate) → build features → forecast the next
week with the champion → save JSON/Parquet/CSV and print the report. The application's
15-minute intelligence refresh calls the same path (`app/nfl/integration.py`) and archives the
engine forecast in the existing pregame ledger; the legacy ridge model remains as a fallback
and is stored alongside for comparison.

Retraining: `python -m app.nfl backtest` (dev + holdout walk-forward) then `python -m app.nfl train`.
A retrained candidate is registered as a challenger and promoted only if its holdout margin
MAE beats the champion by at least 0.05 points with a paired weekly bootstrap interval
entirely below zero and no material log-loss regression (`Registry.challenger_beats_champion`).

## Reproducibility

Fixed seeds (`SEED = 20260909`), pinned dependencies (`requirements.txt`), deterministic
LightGBM/XGBoost settings, a feature version stamp (`fv1`), git commit and dataset fingerprint in
every registry card, and persisted raw predictions for every experiment.
