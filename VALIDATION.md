# Verification — 2026-09-06

The local application was launched and exercised. This report distinguishes executable checks from real-league validation.

| Check | Result |
|---|---|
| Python engine/API/model tests | 29 passed |
| Chrome extension DOM parser tests | 6 passed |
| React/TypeScript production build | Passed |
| Headless Chrome UI smoke | Passed; no page errors |
| Complete synthetic ESPN-format replay | 180 picks, all verified |
| Replay ingest + recommendation p95 | 94 ms |
| Fast candidate ranking, 360-player fixture | 15 ms |
| Deep comparison, six candidates × 1,000 draft paths | 25.0 s |
| Live warm API, 981-player 2026 catalog | 110 ms |
| Public NFL news evidence feed | 26 items cached at verification |

Two dependency deprecation warnings occurred in Starlette's test-client/httpx compatibility layer. There were no test failures after fixes. The tests do not validate a real ESPN browser session.

## Projection evidence

The original opportunity model failed the baseline comparison and had poorly calibrated intervals. It was retained as a benchmark, rather than silently treated as successful.

The selected workload tree achieved common-fold reference-PPR MAE **40.106**, compared with **42.049** for last-season persistence and **50.954** for the original opportunity model. Its yearly MAE was 39.797 (2023), 42.296 (2024), and 38.289 (2025). It predicts workload and games, then applies regressed historical efficiency. Model choice for historical residuals uses only earlier folds.

Prospective local residual P10–P90 coverage:

| Held-out season | Players | Coverage |
|---|---:|---:|
| 2023 | 575 | 76.17% |
| 2024 | 545 | 77.61% |
| 2025 | 559 | 84.08% |

These figures exclude rookies, current team/role adjustments, market ensembles, bonuses and alternative league settings. nflverse datasets may contain later revisions, so this is not a fully vintage-correct backtest. The 2026 performance is unknown.

## Strategy evidence

Six policies each completed 1,000 synthetic drafts. The experimental adaptive policy underperformed ADP. Candidate comparisons now use matched constrained-ADP continuation for baseline and candidate paths. A regression test verifies that taking the baseline player produces exactly zero estimated effect under common random numbers.

The best first-pick candidate in this deliberately synthetic fixture matched the ADP baseline. **No real-world championship advantage has been demonstrated.** Synthetic ESPN/ECR/ADP ranks are identical by construction, so those baseline results coincide. Licensed historical market snapshots and real league inputs are necessary to evaluate genuine strategy improvement.

## Remaining acceptance gates

- Verify both actual ESPN league configurations and team mappings.
- Exercise the extension against the logged-in draft DOM, including virtualized history, reconnects and ESPN IDs for every roster position.
- Import current licensed ADP/market data and verify active player roles, particularly rookies, injuries and depth-chart changes.
- Implement and validate the additional requirements listed in the README, including advanced feeds, full scoring/rule variants and historical strategy validation.

Detailed reproducible outputs are saved locally in `storage/validation.json`, `storage/model_2026.json`, `storage/calibration.json`, and the UI screenshots. Re-run the documented commands after material changes.

## Weekly lineup update — 2026-09-07

The league-snapshot adapter, atomic imports, correct-week forecast validation, injury ID matching, OUT/IR exclusions, byes, locked starters and exact-provider-scoring fallback now have automated coverage. The full Python suite passes, as do six extension parser tests and Chrome checks for lineup/bench rendering and player details. The live injury adapter returned 800 ESPN-ID-linked records. Private ESPN league access remains an acceptance test requiring the user’s league IDs/session. See docs/WEEKLY_LINEUPS.md for provisional probability assumptions and coverage limits.
