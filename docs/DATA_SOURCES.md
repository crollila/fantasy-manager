# Data sources

All inputs are public nflverse releases downloaded into an immutable raw store
(`storage/nfl/raw/<dataset>/<file>.parquet`) with a manifest per file (URL, retrieval time,
SHA-256, byte size, row count, schema, season/week coverage, publication cadence).
`python -m app.nfl status` summarises the store; `storage/nfl/reports/ingest_status.json` and
`storage/nfl/reports/data_quality.json` are written by every ingest/normalize run.

| Dataset | nflverse asset | Seasons | Rows (Sep 2026) | Used for | Point-in-time handling |
|---|---|---|---|---|---|
| Schedules / results | `schedules/games.parquet` | 1999–2026 | 7,548 games | Game ids, kickoff (ET → UTC), scores, roof/surface, rest days, coaches, referee, starting QB ids, closing lines | Kickoff from `gameday`+`gametime` (259 games in 1999 lack a time; 13:00 ET assumed and flagged). Scores only used for games that kicked off before the prediction timestamp. |
| Play-by-play | `pbp/play_by_play_<season>.parquet` | 1999–2025 (2026 published in season) | ~47k plays/season | All team efficiency, drive, pace, special-teams, penalty and QB features | Aggregated per game; rolling windows use strictly earlier kickoffs. `wp` (non-Vegas) defines competitive situations so market information never enters market-free features. |
| Injury reports | `injuries/injuries_<season>.parquet` | 2009–2026 | ~5k rows/season | Expected value lost by position group, QB availability | `date_modified` (2010–2024) must precede the prediction timestamp; 2009 and 2025+ rows have no timestamp and are assumed public 48 h before kickoff. |
| Snap counts | `snap_counts/snap_counts_<season>.parquet` | 2012–2025 (2012 nearly empty) | ~23–25k rows/season | Player value (rolling snap share) for availability features | As-of join on kickoff of the last game played before the prediction timestamp. PFR ids mapped to GSIS via `players`. |
| Depth charts | `depth_charts/depth_charts_<season>.parquet` | 2001–2026 (weekly to 2024, timestamped snapshots from 2025) | 27–37k rows/season; 500k+ snapshots/season from 2025 | Fallback player value when no snap history | Weekly file for the game week (2001–2024); timestamped snapshots not yet used for features. |
| Weekly rosters | `weekly_rosters/roster_weekly_<season>.parquet` | 2002–2026 | ~31–52k rows/season | Reserve/PUP/suspension value lost | Roster status for the game week. |
| Players | `players/players.parquet` | all | 24,826 | GSIS ↔ PFR ↔ ESPN ids, birth dates, draft round/pick, positions | Static reference. |
| ESPN QBR | `espn_data/qbr_week_level.parquet` | 2006–2025 | 11,249 | Ingested; not in champion features | — |
| Officials | `officials/officials.parquet` | 2015–2025 | 22,012 | Ingested; referee tendencies use the schedule's `referee` column for all seasons | Referee penalty rates computed from prior games only. |
| Next Gen Stats | `nextgen_stats/ngs_{passing,rushing,receiving}.parquet` | 2016–2025 | 5.9k / 3.6k / 10k | Ingested and normalized; reserved for the enriched tier | — |
| FTN charting | `ftn_charting/ftn_charting_<season>.parquet` | 2022–2025 | ~48k plays/season | Ingested (play action, motion, blitzers, interception-worthy throws); reserved for the enriched tier | — |
| Participation | `pbp_participation/pbp_participation_<season>.parquet` | 2016–2025 | ~45k plays/season | Ingested (personnel, coverage, pressure, time to throw); reserved for the enriched tier | — |
| PFR advanced weekly | `pfr_advstats/advstats_week_{pass,rush,rec,def}_<season>.parquet` | 2018–2025 | 0.7–8k rows/season | Ingested; reserved for the enriched tier | — |
| Draft picks / combine | `draft_picks/draft_picks.parquet`, `combine/combine.parquet` | 1980+ / 2000+ | 12,927 / 8,968 | QB draft capital prior (via `players`) | Static. |

Not available / not bundled:

- **Timestamped historical odds** (opening lines, line movement, book-level prices). nflverse lines are closing lines without timestamps; the market-aware backtest is therefore a closing-line evaluation. The live `THE_ODDS_API_KEY` adapter in `app/market_data.py` supplies current multi-book quotes when configured; a paid historical archive would be needed for earlier-horizon market features (`app/nfl/features/context.py::market_features` is the single integration point).
- **Archived pregame weather forecasts.** Historical features use the observed conditions recorded by nflverse (`temp`, `wind`; ~72% coverage outdoors, indoor games set to 70°F / 0 mph). Production uses the Open-Meteo forecast already wired into `app/research_sources.py`. Feature names carry the `wx_` prefix and the sidecar `meta_pregame.json` records the caveat.
- **Raw tracking data.** Only aggregate NGS metrics are public.
- **Coordinator / play-caller histories.** Head coaches come from the schedule; coordinator continuity is not modelled.

Licensing: nflverse data is released under CC-BY 4.0; FTN charting is shared through nflverse for non-commercial use. No credentials are required; the optional Odds API key is read from the environment only.
