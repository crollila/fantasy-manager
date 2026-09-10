# Game reviews and prospective learning

Version 0.5 adds **Accuracy → Review** and a **Learning** tab. No subscription is required for the bundled public feeds. The first run downloads additional Next Gen Stats data. Upgrades preserve the existing local forecast ledger and ESPN connection.

## What happens automatically

On app open, manual refresh, and every 15 minutes while the app runs:

1. Refresh the public schedule, scoreboard, injury/depth reports and statistical feeds.
2. Match final games to predictions actually saved before kickoff. Review final scores and available statistical evidence; preserve earlier observations during a source outage.
3. Update derived grades when a publisher corrects a score or statistic. Keep the original predictions and a versioned review history.
4. Update the foundation model from completed game history and fit statistical expectations. An eight-hour kickoff buffer keeps partially populated live schedule scores out of training.
5. Evaluate any frozen correction that has accumulated its required later-game test. Train a new candidate when enough eligible archived games exist.
6. Save future picks, statistical expectations, injury/participation assumptions, weather, market snapshots, exact model coefficients and input-source hashes. If a candidate is being tested, save its shadow score alongside the published score.

Nothing runs while the app is closed. Cached picks, reviews and learning history remain available offline. A missing source can delay a detailed review; it never creates an invented zero.

## What the review measures

The score table shows the original projection, final result and **actual minus predicted** difference for each team. Score MAE is the average absolute error for the two teams. Margin and total errors are also signed actual-minus-predicted quantities. A team score within three points is labeled a close match, a descriptive review threshold rather than a statistical significance test.

New forecasts archive expectations for up to 14 metrics:

| Input | Metric in the review |
|---|---|
| Team statistics | Net passing yards, rushing yards, offensive plays, turnovers lost, sacks allowed |
| Play-by-play | EPA per play, success rate, pass rate in close situations, explosive-play rate, red-zone drive TD rate |
| Next Gen Stats passing | Completion percentage above expectation |
| Next Gen Stats rushing | Rushing yards over expectation per carry |
| Next Gen Stats receiving | Receiver separation, YAC above expectation |

Statistical expectations use a joint, regularized offense/opponent-defense model with home, rest, quarterback and weather context. It uses a 180-day recency half-life and up to three years of prior games. This adjusts for schedule strength rather than interpreting raw yards allowed as defensive quality. At least 160 team-game observations are required per metric. Reference bands are the fitted mean ±1.28155 times historical residual scatter, subject to metric bounds. **These bands are descriptive, not validated 80% prediction intervals.** Their real coverage is observable in later reviews.

EPA, success and explosive rates exclude kneels/spikes and require at least 20 usable offensive plays. Close-situation pass rate uses a score difference of at most seven points with more than two minutes remaining and at least ten qualifying plays. Explosive means 20+ yards. Red-zone TD rate uses offensive drives that reached the opponent's 20 and ended in an offensive TD; defensive return touchdowns do not count.

NGS team aggregates are opportunity-weighted averages of qualifying players: attempts for passing, rush attempts for rushing, targets for separation, receptions for YAC. Week-zero season summaries are excluded. NGS imposes minimum opportunity thresholds, so these are **not** complete all-player/all-snap measurements.

The review also compares saved expected starter participation with available actual player statistics and snap counts, using ESPN/GSIS/PFR identity mappings. A missing row is unknown, not a DNP. A player who participated may still have been limited or injured during the game; the review does not infer full health from one snap. Weather comparisons use saved kickoff forecasts versus subsequently available reported conditions, not reconstructed forecasts.

“What matched” and “What missed” describe observed evidence. Turnovers, efficiency and personnel surprises are useful diagnostic clues, but correlated differences do not establish causation or allocate exact points of the score error to an injury. The app does not generate an unverified story and treat it as a training label.

Older forecasts retain their original accuracy record. If they did not save statistical or participation expectations, those expectations cannot be recreated using hindsight; the new correction layer does not train on them.

## How the correction learns

The foundation already refits historical results on refresh. The new layer separately learns **residual errors of saved foundation predictions**. Its inputs are the archived foundation score, team/opponent indicators, home/rest/weather values, personnel point adjustments, and both teams' saved statistical expectations, with explicit indicators for missing metrics. Ridge regularization (alpha 40) shrinks noisy coefficients. Corrections are limited to ±3 points per team and final scores to 0–60; win probabilities, spread picks and total picks are recomputed consistently.

The promotion process is deliberately prospective:

- **Train:** require at least 80 new-format archived games. All archived games in an NFL week must be final and at least four days beyond the latest kickoff in that week. This allows routine stat corrections and prevents splitting teams from the same game or a partially completed week across training and evaluation.
- **Freeze:** save the candidate coefficients, creation time, training forecast IDs and outcome fingerprint. Do not refit it while its test is running.
- **Shadow:** predict future games from the same inputs and at the same time as the published model. Training IDs and forecasts created before the candidate existed cannot count as its test.
- **Evaluate once:** use the first full set of mature weeks with at least 32 shadow games across at least three NFL weeks. Compare candidate and published model on those exact games. No repeated testing of the same candidate after a failed result.
- **Promote:** require mean score-MAE improvement of at least 0.15 points per team; the lower bound of a 95% paired week-bootstrap interval must exceed zero; three-outcome Brier error must not worsen by more than 0.01. The bootstrap uses 2,000 deterministic samples and resamples whole NFL weeks.
- **Monitor:** after promotion, compare successive fixed batches of at least 32 later games and three weeks against the saved unadjusted foundation. If reverting passes the same improvement rule, roll back. Record every training, rejection, promotion, monitoring decision and rollback.
- **Try again:** a rejected or completed candidate is retired. At least 32 additional eligible games are needed before training another candidate. The active model, if any, continues while the next candidate shadows.

A later stat correction updates the review and future training data. It does not rewrite a past model decision or silently retrain an already frozen artifact. The correction layer starts with no approved changes. Several weeks of prospective data are required; the UI shows actual counts. A finite successful test does not guarantee improvement, and repeated experimentation can still overfit. No advantage over Vegas is claimed.

## Comparisons and additional sources

The **Accuracy** tab reports score, margin, total and probability errors with explicit denominators. Market margin/total errors use exactly the same games as the corresponding model errors. When both moneylines exist, implied probabilities are normalized by their sum to remove the two-sided overround; that binary Brier comparison excludes ties. The main three-outcome Brier metric includes home/away/tie. Market schedule values and live API quotes are saved snapshots, **not certified closing lines**. No closing-line-value claim is made.

The **Learning** tab shows probability reliability bins and error breakdowns by archived pregame weather, expected starter absences and rest disadvantage. These are descriptive, overlapping subsets and are not causal estimates of win probability under a condition. Historical model diagnostics remain separate from the prospective record.

**Data & evidence → Add a forecast source** accepts a local JSON export of licensed game-score forecasts. Download the format to get an actual upcoming game ID, fill in real provider predictions and their issue time, then import before kickoff. Rows require the correct home/away team identifiers. The entire import is rejected on a duplicate, mismatched game, future issue time, issue time older than 14 days or already-started game. No past benchmark is backfilled into the live record. Refresh archives the comparison with subsequent pregame predictions. Imported scores are benchmarks, not automatic extra model weights.

```json
{
  "source": "Your licensed provider",
  "issued_at": "2026-09-10T12:00:00Z",
  "rows": [{
    "game_id": "COPY_FROM_THE_DOWNLOADED_FORMAT",
    "home_team": "DEN",
    "away_team": "KC",
    "home_score": 24.5,
    "away_score": 22.0,
    "home_win_probability": 0.57
  }]
}
```

The optional probability is a conditional home-win probability excluding ties. Current external-provider comparisons report score MAE; they do not yet grade that optional probability. Leave it null if unavailable. Player/fantasy projection imports are available separately under **Advanced tools → Market / ADP**; ESPN comparisons retain each league's saved scoring rules.

Useful optional additions, when licensed access is available:

| Source | What it would add | Current integration |
|---|---|---|
| The Odds API | Timed multi-book moneylines, spreads and totals | Existing live adapter via local `THE_ODDS_API_KEY`; no key or subscription is bundled |
| Historical odds | Pregame/closing price vintages and more rigorous market evaluation | Requires paid historical access and a separate historical replay adapter; not fetched automatically |
| Provider score forecasts | Independent same-game benchmark | JSON import described above |
| Player props / tracking / protection and coverage charting | Additional role, efficiency and individual matchup features | Requires licensed timestamped exports, IDs, definitions and a dedicated adapter; not claimed as connected |
| Archived weather forecasts | Weather forecast error at fixed lead times | Current forecasts are archived locally from now on; no retrospective forecast reconstruction |

More data must have known timestamps, identities and definitions. Adding a provider does not by itself prove better predictions. Do not put API keys in Git, an import file or a chat message.

## Provenance and exports

**Export full prediction record** downloads the game's pregame versions, all review revisions, frozen foundation/process/correction artifacts and source metadata. The SQLite tables `game_forecasts`, `game_result_versions`, `game_model_artifacts` and `game_learning_events` preserve the audit trail locally. `game_results` is the latest derived grade. No league login data are included in a game audit export.

The JSON endpoints are `/api/intelligence`, `POST /api/intelligence/benchmarks` and `/api/intelligence/games/{game_id}/audit`. They require the existing local token/origin controls.

## Provider publication limits

- [nflverse's update schedule](https://nflreadr.nflverse.com/articles/nflverse_data_schedule.html) describes post-game-day statistics and later corrections. App polling frequency cannot make an unpublished result available.
- [Next Gen Stats documentation](https://nflreadr.nflverse.com/reference/load_nextgen_stats.html) documents weekly qualifying-player coverage. The app requests the three aggregate files at most every six hours; upstream publication is generally nightly.
- [The Odds API historical data](https://the-odds-api.com/historical-odds-data/) documents paid timestamped snapshots. Those are distinct from the app's currently available schedule snapshots.
- [Open-Meteo previous runs](https://open-meteo.com/en/docs/previous-runs-api) distinguishes archived forecasts from observed/reanalysis weather. Current observed weather must not be passed off as a historical pregame forecast.
