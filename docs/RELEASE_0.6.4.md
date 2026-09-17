# Fantasy Manager 0.6.4

Fixes the error shown when the app opens and makes player projections complete, persistent and graded.

## Fixed: error on opening the app

The NFL games tab showed `SyntaxError: Unexpected token 'I', "Internal S"... is not valid JSON`
and no games. Forecasts from the NFL engine store a short text note beside the home and away
scoring components, and the accuracy diagnostics crashed on it. They now skip it.

## Every player is projected

The weekly refresh was meant to project every player with an upcoming game, but the player list
contained only players from synced ESPN rosters, none with stat history, so no one was projected
and nothing could be graded. The refresh now builds the full player list from NFL statistics —
about 960 players a week — keeping the ids of players already on your rosters.

A new **Player projections** tab lists them all with position filter, search, likely range,
boom/bust chances, ESPN's figure where available and the actual score once the game is final.
They use default PPR scoring so every player is comparable; your team view keeps your league's rules.

## Lineup projections stay put

The last lineup analysis for each team is saved and shown immediately when you switch teams or
reopen the app, until a new analysis replaces it. The week now defaults to the current NFL week,
and projections are refreshed quietly after a data refresh re-syncs your teams.

## Player projection accuracy

The Accuracy tab adds a player section once games are graded:

- **Average miss (MAE)** — mean absolute error in fantasy points, lower is better
- Share of projections within 3 and 5 points, typical (median) miss, RMSE and lean (high or low)
- **ESPN on the same players** — ESPN's projected stat line rescored under the same rules, with
  how often ours was closer
- Breakdowns by position and by week, biggest misses and closest calls

Each player-game counts once, from the last projection saved before kickoff.

Install `Fantasy-Manager-Setup-0.6.4.exe` over the previous version. Saved teams, the ESPN
profile and existing prediction history are retained.
