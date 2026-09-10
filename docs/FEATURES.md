# Features

The feature factory (`app/nfl/features/`) produces 2,100 candidate features per game for the
`pregame` horizon (kickoff − 1 h). Every family is listed with its provenance in
`storage/nfl/features/fv1/meta_pregame.json`. Two model feature sets are used:

- **compact** (699 features): the curated set used by the champion base learners.
- **full** (2,095 features): every non-market feature, evaluated as a challenger.
- **compact_market** / **full_market**: the same plus the 5 market features (market-aware systems only).
- **core**: compact without the availability family (long-history tier, 1999+).

Naming: `home_*` / `away_*` for the two teams, `diff_*` = home − away, `mx_*` = matchup interactions.
Windows: `__l1/__l3/__l5/__l8/__l16` last-n games (across seasons), `__std` season-to-date,
`__prev` previous full season, `__ewm` exponentially weighted (half-life 4 games), `__sd8`
8-game standard deviation, `__trend` = l3 − l8. `*_allowed` = the opponent's offensive production
in the same games (the team's defense).

## Team play-by-play form (`team_form_offense`, `team_form_defense`)
Per team-game aggregates from nflverse play-by-play, then windowed:

- Efficiency: EPA/play, EPA/dropback, EPA/rush, early-down, late-down, third-down, red-zone, goal-line, neutral-situation (win probability 10–90%), leading/trailing/tied, two-minute, short/long yardage; totals of EPA.
- Success: overall, pass, rush, early, late, neutral, red zone, third down; third/fourth-down conversion rates; first-down rate.
- Explosiveness: 10+/20+/40+ yard rates, explosive pass (20+) and rush (10+) rates, yards/play, yards/attempt, yards/carry, air yards per attempt, YAC per completion, deep-target rate and deep EPA (2006+).
- Play calling: pass rate, neutral pass rate, close-game pass rate, early-down pass rate, PROE (pass rate over expected, 2006+), shotgun and no-huddle rates, scramble rate.
- Passing quality: CPOE, completion %, sack rate, QB-hit rate, interception rate.
- Turnovers: interceptions, fumbles, fumbles lost, giveaways, fumble rates (unstable outcomes are shrunk by the windowing and regularisation rather than used raw).
- Drives: drives, points/drive, EPA/drive, yards/drive, plays/drive, TD/FG/punt/turnover/turnover-on-downs per drive, three-and-out rate, red-zone trips per drive, red-zone TD rate.
- Pace: seconds per play (all and neutral drives), plays per game, no-huddle rate.
- Special teams: FG attempts/makes by distance bucket, FG make rate, mean attempt distance, XP, punts and gross punt distance, special-teams EPA, FG EPA, punt EPA.
- Penalties: committed count/yards, false starts, offensive holding; drawn count/yards, DPI drawn, defensive holding drawn.
- Garbage time: share of plays outside 10–90% win probability, mean win probability.
- Record: games played (season / total), wins last 5, wins season-to-date, previous season win %.

## Ratings (`elo`, `ratings_margin`, `ratings_efficiency`)
- Elo: margin-aware K=20, 55-point home advantage, 1/3 regression to 1505 each offseason; pre-game rating, difference, implied probability and margin.
- SRS-style ridge margin ratings, refit at every (season, week) snapshot on games before that week's first kickoff, exponential recency (400-day half-life) inside a 1,100-day window; home-field estimate.
- Opponent-adjusted efficiency ratings (offense and defense effects) for EPA/play, EPA/pass, EPA/rush, success rate, points/drive, neutral EPA and explosive rate, refit weekly (300-day half-life); league means at the snapshot.

## Quarterback (`qb`)
Expected starter (schedule starter for the pregame horizon, previous starter for the early horizon) with shrunk career, previous-season, season-to-date, last-4 and EWM EPA/dropback; CPOE; sack, interception, hit and scramble rates; success rate; ANY/A; neutral-situation EPA; dropbacks, starts, experience; rookie/no-history, changed-starter and new-to-team flags; draft round/pick and age.

## Availability (`availability`)
Expected value lost per position group (QB, RB, WR, TE, OL, DL, LB, DB, ST): Σ P(miss | report status, practice status) × player value (recent snap share, depth-chart fallback), from injury reports known before the prediction timestamp; starters out / questionable counts; QB starter listed out or questionable; reserve-list value lost (IR/PUP/suspension); listing and DNP counts; source-availability flags.

## Context (`schedule`, `rest_travel`, `weather`, `coaching`, `officials`)
Season, week, playoff, neutral site, division/conference game, indoor/dome/retractable, grass/turf, kickoff local hour, primetime, Thursday/Monday/Saturday, international; rest days, short week, bye, rest difference; travel miles, time-zone shift and body-clock hour (stadium coordinates in `app/nfl/reference/teams.py`); observed temperature and wind with thresholds (wind > 10/15 mph, cold, freezing, heat) and missing indicators; head-coach tenure with the team, career games, first season, change since last game; referee penalty / DPI / holding / total-points tendencies from prior games (shrunk).

## Matchup interactions (`matchup`)
Offense-vs-defense sums of adjusted ratings (pass, rush, total, points/drive), explosive-play, pressure (sack rate taken vs generated), QB vs pass defense, red-zone and third-down matchups, combined pace and pass rate, wind × pass rate, Elo × QB change.

## Market (`market`, market-aware systems only)
Closing spread (home margin), total, implied team scores, no-vig moneyline probability.

## Not yet in the champion (ingested, normalized, ranked in ML_SYSTEM.md)
Next Gen Stats aggregates (2016+), FTN charting (2022+), participation/coverage (2016+), PFR advanced stats (2018+), timestamped depth-chart snapshots (2025+), transactions/trades.
