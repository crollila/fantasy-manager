# Fantasy Manager 0.6.5

Fixes refreshes that stopped updating your teams, fixes a lineup error and makes the data refresh
about twice as fast.

## Fixed: refreshing stopped updating your teams

One kicker had been saved twice in the player list, once from the ESPN roster and once from
the NFL statistics, under the same ids. Every ESPN import checks that each id belongs to one
player, so every **Refresh teams** after that point failed. Rosters, ESPN projections and
lineups stayed on an old week. The same check blocked the weekly player list rebuild.

- The player list now repairs itself. Entries that share an id are merged into the one your
  rosters use, keeping its stat history.
- The scheduled player-list refresh, which created the duplicate, now merges new players into
  the saved list and no longer adds a second copy of anyone.
- Teams now re-sync from ESPN after every data refresh, including a refresh that failed.

## Fixed: "Find my best lineup" error

A player with two feature rows for the same week crashed the lineup analysis with an
`isfinite` error. The latest row is now used. A roster player missing from the player list is
reported instead of stopping the analysis.

## Faster refresh

**Refresh data & results** took about three minutes on a typical data set and now takes about
one and a half. The results are identical.

- Player feature building is 8× faster. Its rolling averages now use pandas' grouped windows
  instead of Python code run separately for every player.
- Play-by-play game metrics are computed for all teams at once, down from 50s to 15s.
- The replacement-player study reads only completed past seasons. It is now saved and reused
  until those files change, saving about 25s per refresh.
- The player point model is refit only when the downloaded statistics actually change, not
  every time an unchanged file is downloaded again.
- Lineup requests reuse the week's player features instead of rebuilding them, saving about
  25s per team.

## Smaller fixes

- A refresh cut short by closing the app no longer shows as "running" on the next start.
- `desktop.log` entries now carry timestamps. The log is rotated once it passes 5 MB.

Install `Fantasy-Manager-Setup-0.6.5.exe` over the previous version. Saved teams, the ESPN
profile and existing prediction history are retained. The duplicate is repaired on the first
refresh after the update.
