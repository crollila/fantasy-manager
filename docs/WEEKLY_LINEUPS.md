# League sync and injury-aware weekly lineups

The weekly assistant is inspired by the league-specific start/sit and range-of-outcomes experience described by [WalterPicks](https://www.walterpicks.com/). This project is independent; it does not use WalterPicks' code, models, branding or paid data.

## Connect a league

Open **My lineup** in the app. The ESPN URL contains `leagueId` and usually `teamId`; enter both IDs, the season, and the scoring week. Click **Sync public league**. Select the resulting league in the sidebar. Repeat for another league.

For private leagues:

1. Load/reload `extension` as an unpacked Chrome extension. Version 0.2 adds permission for ESPN's read API.
2. Stay logged into ESPN in that Chrome profile.
3. Copy the local pairing token from **Data & models** into the popup.
4. Fill league ID, your ESPN team ID, season and week, then click **Sync league**.
5. Refresh the app to list the imported league, select it, and open **My lineup**.

The extension sends the read-only league snapshot to loopback. ESPN authentication cookies are sent only to ESPN by Chrome; they are never read into the code, copied to the app, or uploaded to GitHub. No ESPN lineup is edited automatically. Draft DOM monitoring continues on draft pages.

The ESPN read API is undocumented. It is independently used by the maintained [espn-api client](https://github.com/cwendt94/espn-api). Requests validate complete team rosters and settings and fail visibly on authentication or schema errors. Actual private-league access still needs testing with your league/profile; fixture tests are not a substitute for that acceptance test.

## Analyze your lineup

Select the same week as the imported snapshot and click **Analyze my lineup & injuries**. Choose highest expected points, floor, or upside. The report includes:

- Recommended starting slots and bench alternatives, respecting ESPN eligibility and available game-lock information.
- Projected total and P10–P90 range, plus the change from your currently synced starters.
- Per-player expected points, floor, ceiling, boom/bust estimates, injury designation and modeled availability.
- Suggested starts/sits, source timestamps, missing projections, and a detail view with recent usage and injury evidence.

Each sync replaces ownership and weekly metadata atomically. A wrong-week snapshot is rejected. Re-sync weekly and close to kickoff; league snapshots do not refresh while the extension is paused or the app is closed.

## Inputs and limits

The model uses the independent historical opportunity/efficiency catalog, available prior current-season weekly usage (never the target week's actual results), ESPN's current weekly forecast, exact provider scoring, injury designations, byes, roster eligibility and scheduled kickoff times. Confirmed structured events continue to affect independent projections.

The public [ESPN injury feed](https://www.espn.com/nfl/injuries) is cached with fetch time and source report time. Matching uses ESPN IDs, never unconfirmed names. Missing/stale reports are flagged. Out, IR, inactive, suspended and bye players have zero modeled availability. The questionable 70%, doubtful 15% and unknown 90% probabilities are explicit policy assumptions, not calibrated medical estimates. Descriptions are evidence; no diagnosis or recovery time is inferred from prose.

A provisional 65% independent / 35% ESPN weekly blend is used when both are available. If scoring includes components the independent model cannot represent, the engine uses ESPN's already league-scored forecast and labels that fallback. Without either valid forecast, the player is reported missing rather than assigned invented points. Weekly forecast-source weights and intervals still need historical weekly calibration.

Boom means exceeding 150% of the healthy weekly forecast; bust means below 60%. Availability, workload, efficiency and weekly variability affect simulated outcomes. A started player's known current points are frozen, rather than pretending that a complete live-game remainder model exists. Locked starters remain in their current slots; locked bench players cannot be promoted. Re-sync whenever ESPN status changes.

**This does not claim to ingest all football statistics.** Snap/route participation, weather, betting lines, offensive-line changes and detailed practice participation are not available in every forecast and are not silently treated as observed. Weekly probabilities are estimates, not proven championship or game-win probabilities. See each player's coverage warnings.

## API

- `POST /api/espn/sync`: `{league_id, my_team_id, season, week}` for public leagues.
- `POST /api/espn/import`: same fields plus `snapshot`, for the authenticated browser bridge or a local export.
- `POST /api/leagues/ID/lineup`: `{week, risk: "balanced"|"floor"|"upside", refresh_injuries: true}`.

All calls require `X-Local-Token`. Runtime database, private league snapshots, caches and tokens are ignored by Git.
