# Fantasy Manager 0.7.3

Fixes the app running out of memory after a few weeks of use.

## Fixed: the saved forecast history kept growing

The app refreshes about every 15 minutes while it is open. Each refresh stored a new copy of
every finished game's review, every upcoming forecast and most player projections, even when
nothing but the download time had changed. After two months the database could pass 3 GB.

A new version is now saved only when something changed:

- A change to a pick, score, win probability, betting line or injury report is saved at once.
- Small model drift is saved at most once every six hours.
- In the last two hours before kickoff every change is saved, so the graded forecast is the
  latest one.

The first refresh after updating removes the duplicate copies already stored and shrinks the
file (about a minute, one time). Every graded forecast, game result and player result is kept
unchanged, along with each game's first and final pregame forecast.

## Fixed: high memory use on computers with many CPU cores

The forecasting engine reserved memory for every CPU core. On a 32-core PC it held about 6 GB
and could fail with an out-of-memory error, which left the dashboard empty. It now uses about
2 GB at rest and a refresh is slightly faster.

## Fixed: occasional dashboard error right after opening

A request made while the engine was still starting could fail once. It no longer does.

Saved teams, the ESPN profile and prediction history are kept.
