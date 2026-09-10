# Weekly intelligence and game picks — version 0.5

The home screen has **My teams**, **NFL games**, **Accuracy**, **Learning**, and **Data & evidence**. Predictions are estimates. The live record starts with forecasts actually saved before kickoff; historical diagnostics never become live wins.

For the new statistical reviews, prospective correction tests, data imports and audit exports, read [Game reviews and learning](GAME_LEARNING.md).

## How player projections work

1. Start with the independent historical opportunity/efficiency catalog and shrink recent targets, carries and pass attempts toward that baseline.
2. Apply a league-scored, position-specific opponent adjustment. A regularized model separates offensive team/season strength from the defense faced, home field and weather. A defense that faced elite offenses is not automatically a weak defense. Estimates shrink toward neutral and have bounded effects; this does not perfectly isolate every individual matchup.
3. Incorporate verified depth-chart absences and historical backup workload. Offensive and defensive units have separate learned, bounded team-scoring effects. QB identity is included when historical data support it. Prior-season snap shares provide a conservative proxy for line/defensive personnel turnover; they are not player talent grades. Established transfers are not automatically treated as backups.
4. Apply learned wind, cold, heat and precipitation effects, with regularized team interactions. Indoor venues receive neutral weather. Neutral-situation pass rate, red-zone/goal-line opportunities and play-by-play efficiency are visible evidence. Play-calling effects are not multiplied into fantasy points a second time when the fantasy-weather model already captures them.
5. Blend the independent projection with ESPN's exact weekly league projection: initially 65% / 35%. After at least 50 completed, archived comparisons for a position, past absolute error selects a weight, shrunk toward the initial weight. This is adaptive fitting, not proof of superiority. Unsupported independent scoring or missing historical production uses the provider forecast with an explicit warning.
6. Simulate workload and efficiency variability with shared game and passing-team factors. Weekly ranges and correlation magnitudes remain provisional. The app reports range coverage and boom/bust Brier errors as actual results accumulate.

### Injury treatment

The displayed points are **conditional on playing**, with a binary eligibility rule. Estimated play probability at least 50% retains full points; below 50%, official unavailable status, or a bye gives zero. Questionable status alone does not multiply points by 70%.

Participation probabilities are separate. Recent dated report phrases such as “expected to play,” “game-time decision,” or “not expected to play” modify transparent designation priors. Official absence wins over optimistic prose. Undated/stale text does not override the prior. This is a rule-based report-language classifier, not calibrated sentiment AI, an injury diagnosis, or a recovery prognosis. ESPN's own projection may already contain its injury assumptions; we cannot reconstruct or remove them.

Matchup win simulations include participation uncertainty separately. The **Favor matchup win probability** objective searches legal lineup swaps against the opponent's modeled balanced lineup. It respects eligibility and known locks, but local search is not guaranteed globally optimal. Current live points plus clock-scaled remaining production are available when the live clock is known; otherwise only banked points are shown. This is not a full live play-by-play forecast.

## NFL scores and winners

Version 0.6 publishes the picks of the point-in-time forecasting engine in `app/nfl/` (see [ML_SYSTEM.md](ML_SYSTEM.md), [BACKTEST_REPORT.md](BACKTEST_REPORT.md) and [MODEL_CARD.md](MODEL_CARD.md)). Each refresh updates the engine's nflverse raw store, rebuilds the point-in-time features and asks the registered champion for a market-free forecast (expected score, calibrated win probability, Monte Carlo score distribution and intervals, spread/total probabilities, drivers) plus a market-aware companion forecast. The forecast is archived in the same pregame ledger, so accuracy tracking, reviews and the correction layer keep working unchanged; the legacy ridge forecast described below is stored alongside as `legacy_forecast` and remains the fallback when no champion is registered. `/api/nfl/predictions` and `/api/nfl/model` expose the raw engine output and the holdout metrics.

The legacy model: a regularized model learns team offense, opposing defense, home/neutral field, rest, QB and team-specific weather relationships from up to four years of completed regular-season games. Recency weighting favors recent games. Historical backup cohorts add bounded personnel adjustments. Residual spread estimates produce winner probabilities and score-margin intervals.

Market lines are **benchmarks**, not features in this independent score model. The default benchmark is the available nflverse schedule snapshot, which is not guaranteed live or closing. An optional local `THE_ODDS_API_KEY` enables current spreads/totals and quoted prices from The Odds API. Without a key, the app remains functional; there is no bundled premium subscription. Quotes require exact team and kickoff matching and are rejected when older than six hours. Keys are never written to the repository or source metadata.

Winner, spread and total picks are recorded before kickoff. Where actual quoted prices exist, the app also computes one-unit hypothetical returns. It does not place bets. Better winner accuracy is different from profitable spread betting; neither an edge over Vegas nor positive returns has been demonstrated.

## Automatic updates and audit history

Opening the home screen starts an asynchronous refresh. The API repeats public intelligence refreshes every 15 minutes while running; the UI reads updated status every 30 seconds. Connected desktop leagues are refreshed after the opening/manual refresh when the ESPN session permits, and before lineup analysis when its refresh checkbox is enabled. The older season catalog still refreshes every six hours. Nothing runs while the app is closed.

Source caching limits redundant downloads:

| Input | Requested cache interval |
|---|---:|
| Schedule | 15 minutes |
| ESPN scoreboard | 5 minutes (requested during intelligence runs) |
| Injury feed | Forced on intelligence refresh and requested lineup refresh |
| Live depth charts / kickoff weather | 30 minutes |
| Current-season player/team weekly statistics | 30 minutes |
| Current-season snaps / play-by-play | 1 hour |
| Historical statistics | 24 hours |
| Next Gen Stats passing/rushing/receiving | 6 hours |
| Historical depth charts / current rosters | 6 hours |

These are polling intervals, not promises about provider publication latency. Statistics may arrive after games. Missing current-season files before Week 1 are normal. The evidence screen displays retrieval times and availability; stale forecasts remain accessible when refresh fails. Forecast weather uses approximate city coordinates and a finite forecast horizon. Missing weather has neutral inputs, not invented conditions.

The SQLite archive stores prediction versions, inputs, model version, creation time and kickoff. The last pregame version is graded once. Repeated result refreshes are idempotent; ties and pushes are separate from decisive W/L. Final scores update automatically on subsequent refreshes. Corrections can update derived grades without rewriting the forecast. Player comparisons require identifiable actual statistics and supported saved scoring; missing rows are not assumed to mean zero. This conservatively excludes some DNPs and is a coverage limitation.

Postgame reviews show score errors, available turnover/sack observations and what the model assumed before kickoff. These are measured discrepancies, not causal proof of why a pick missed. Automatic training uses completed data and archived errors on future refreshes; it does not invent a narrative and turn it into a new rule after each loss.

## Evidence and limits

Historical game diagnostics hold out a substantially complete season and use observed historical weather/starting QBs. They exclude the new personnel adjustments and are not a fully vintage-correct, end-to-end backtest. The actual held-out comparison is shown even if market favorites perform better. Live common-game comparisons are the fairer prospective test.

Backup effects are observational, shrunk estimates from historical depth charts and snap participation. In-game injuries, bad depth information and other confounding factors can affect those estimates. Sparse groups stay neutral. The system does not assign an invented individual talent grade to every backup, lineman or defender.

Live routes, reliable coverage assignments, proprietary tracking data, player props, complete coaching/scheme histories and validated injury participation labels are not bundled. Their absence is explicit. More data are useful only when identities, timestamps, availability and out-of-sample benefit can be verified. No feed claims universal coverage and no model claim guarantees superior results.

## Source documentation

- [nflverse data schedule and availability](https://nflreadr.nflverse.com/articles/nflverse_data_schedule.html)
- [Historical snap counts](https://nflreadr.nflverse.com/reference/load_snap_counts.html)
- [Participation-data publication limits](https://nflreadr.nflverse.com/reference/load_participation.html)
- [ESPN scoring formats](https://support.espn.com/hc/en-us/articles/360003914032-Scoring-Formats)
- [ESPN stat identifiers used by espn-api](https://github.com/cwendt94/espn-api/blob/master/espn_api/football/constant.py)
- [Open-Meteo forecast API](https://open-meteo.com/en/docs)
- [The Odds API](https://the-odds-api.com/liveapi/guides/v4/)

Downloaded data and account-specific league snapshots stay in the local data directory. Public repository and installer contain application code, not private league data or credentials.
