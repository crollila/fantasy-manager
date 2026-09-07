# League sync and injury-aware weekly lineups

The weekly assistant is inspired by the league-specific start/sit and range-of-outcomes experience described by [WalterPicks](https://www.walterpicks.com/). This project is independent; it does not use WalterPicks' code, models, branding or paid data.

## Connect a league

In the Windows app use **Connect ESPN** and sign in in its separate window. The following ID/extension flow is also available under Advanced tools.

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

Select the same week as the imported snapshot and click **Find my best lineup**. Choose highest expected points, floor, upside, or matchup win probability. The report includes:

- Recommended starting slots and bench alternatives, respecting ESPN eligibility and available game-lock information.
- Projected total and P10–P90 range, plus the change from your currently synced starters.
- Per-player expected points, floor, ceiling, boom/bust estimates, injury designation and modeled availability.
- Suggested starts/sits, source timestamps, missing projections, and a detail view with recent usage and injury evidence.

Each sync replaces ownership and weekly metadata atomically. A wrong-week snapshot is rejected. Re-sync weekly and close to kickoff; league snapshots do not refresh while the extension is paused or the app is closed.

## Inputs and limits

Version 0.4 uses opponent-adjusted positional defense, league scoring, recent usage, depth-chart and historical backup evidence, home/away and team-specific weather effects. Full points-if-active are retained when play probability is at least 50%; unlikely/out/bye is zero. Injury probability is separate, based on designation priors and recent dated report-language rules.

Weekly blending can learn from archived completed forecasts. Shared game/team simulation factors and a matchup win objective are available. Known current points plus clock-scaled remaining production are used when live clock data exist. These intervals and participation estimates remain provisional.

See [the full model, source, update and accuracy documentation](INTELLIGENCE.md) for current behavior and limitations. That document supersedes the earlier version 0.2 modeling assumptions.

## API

- `POST /api/espn/sync`: `{league_id, my_team_id, season, week}` for public leagues.
- `POST /api/espn/import`: same fields plus `snapshot`, for the authenticated browser bridge or a local export.
- `POST /api/leagues/ID/lineup`: `{week, risk: "balanced"|"floor"|"upside"|"win", refresh_injuries: true}`.

All calls require `X-Local-Token`. Runtime database, private league snapshots, caches and tokens are ignored by Git.
