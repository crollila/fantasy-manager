# Fantasy Manager


**[Download the Windows app](https://github.com/crollila/fantasy-manager/releases/latest)** · [Installation and ESPN setup](docs/DESKTOP.md)

Version 0.3 adds a standalone Windows application and installer. The simpler home screen puts your teams first, with one **Find my best lineup** action and draft/research features under **Advanced tools**. Connect public or private leagues through an isolated ESPN sign-in window; no copying league IDs or cookies is required. Sign-in is performed by you, and ESPN account access is not bundled with the app. The installer includes Python, the analysis engine, and the desktop UI.

Your league links, database and ESPN session remain on your computer. The public repository and downloads contain application code only. The current installer is unsigned. See the desktop guide for data locations, offline operation, authentication limitations and reproducible builds.


Local-first fantasy football research and decision support: Python/FastAPI, SQLite, cached Parquet, React/TypeScript, and a read-only Chrome Manifest V3 ESPN monitor.

**This is a working, tested local application, not a validated production championship oracle.** The core engines and replay pipeline work. Live ESPN compatibility, your two exact league configurations, current market feeds and several advanced modeling requirements still need the inputs and validation listed below. Never interpret simulated championship probabilities as established real-world odds.

## New: injury-aware weekly lineups

Open **My lineup** to sync an ESPN league and compare recommended starters with your current lineup. Public leagues sync directly; private leagues use the Chrome extension while you are logged in. Per-player cards show weekly projections, floor/ceiling, boom/bust estimates, injury status, availability assumptions and source freshness. Byes, IR/out players, ESPN slot eligibility and known game locks are respected.

See [weekly setup and modeling details](docs/WEEKLY_LINEUPS.md). Actual private-league access requires your league IDs/profile; it has not been verified against your personal leagues yet. The model reports missing data and does not claim to ingest every football statistic.

## Start on Windows

Requires Python 3.12+ and Node.js 20.19+ / npm. Open a terminal in this directory:

```bat
start.bat
```

Then open **http://127.0.0.1:8000**. Ctrl+C stops the server. The command creates a virtual environment if necessary, installs dependencies on first run, builds the frontend, and starts the API and frontend on one loopback port. First installation requires internet; cached operation works offline. Linux/macOS: `sh start.sh`.

For explicit installation:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
npm --prefix frontend ci
npm --prefix frontend run build
.\.venv\Scripts\python.exe -m uvicorn app.api:app --host 127.0.0.1 --port 8000
```

Do not bind this service to a public interface. It uses a local pairing token, origin checks, and host validation; it is not a multi-user internet service.

## First real league

1. The **Demo** league contains deliberately fictional players, isolated in season 2099. Demo outputs are not 2026 advice.
2. In **League settings**, choose **New league from these settings**. Set `id` to your ESPN league ID, `name`, `season: 2026`, team count, `my_team`, scoring, slots, bench/IR, draft order, waiver rules, and playoffs. Repeat for the second league.
3. Team indexes are zero-based. `draft_order` lists the team indexes in first-round order. Snake reversal is automatic. Ordinary snake and linear drafts work. Auction optimization is explicitly unsupported.
4. Check every field against ESPN before setting `settings_verified: true`. The default is a sample PPR rule set, **not an assertion of ESPN's exact defaults**. In particular, default DST scoring includes defensive event components but omits points/yards-allowed bands. Add supported component projections and corresponding rules if your league scores these.
5. In **Data & models**, refresh 2026 data. A real 2026 projection database was generated during development; rerun refresh to obtain newer source snapshots.
6. Import ADP/market data you have a license to use. No FantasyPros subscription/API access is assumed. Missing ADP is shown as unavailable; simulations use model rank as an explicitly weaker fallback.
7. Use **Mock laboratory** and replay a draft before your timed draft. Keep ESPN visible and make all selections yourself.

`scoring` maps statistical component names to points per unit. `bonuses` are weekly thresholds, e.g. `{"stat":"passing_yards","threshold":300,"points":3}`. Multiple thresholds stack. Advanced unsupported stat categories must be provided explicitly; the app cannot infer unavailable inputs. IR slots are not drafted. Supported slots: QB, RB, WR, TE, K, DST, FLEX (RB/WR/TE), SUPERFLEX (QB/RB/WR/TE), RB/WR, WR/TE. Supported playoff format: reseeded single elimination, top seeds receive byes, one or two weeks per round. Divisions, custom brackets, keepers, traded picks and third-round reversal are not implemented.

## ESPN monitor

1. Open Chrome's extensions page, enable Developer mode, and **Load unpacked** the `extension` folder.
2. In the local app's **Data & models**, display the pairing token.
3. Open the extension popup. Paste the token and the ESPN league ID, which must equal the local league ID.
4. Enter an explicit mapping from ESPN team names or DOM team IDs to local zero-based indexes, for example `{"My Team":7,"Other Manager":0}`. Include every team.
5. Open your logged-in ESPN football draft page. Enable the monitor and click **Inspect current page**. Confirm the parsed pick numbers, teams and ESPN player IDs against the visible history.
6. If ESPN uses different markup, supply a draft-row CSS selector. The parser needs an overall pick number, a player link containing an ESPN ID, and a mapped team. `parser.js` supports configurable pick/team/timer selectors as well.

The monitor uses a MutationObserver and retry polling, then sends full snapshots to loopback through its service worker. Authentication cookies never leave ESPN. It has no code to click a Draft button. Backend validation checks contiguous overall pick numbers, snake ownership, canonical identities and duplicate players. Snapshots are transactional, idempotent and append-only; conflicting or truncated history is rejected visibly.

**Live limitation:** the supplied adapters are verified against synthetic DOM fixtures, not your logged-in ESPN draft room. Virtualized history may omit earlier picks and will be rejected. The extension captures settings-table evidence but does **not** automatically claim exact scoring detection. Use manual settings verification. Unmapped ESPN IDs require explicit player mappings; names are suggestions only. DST IDs particularly require mapping. Paused/offline sync retries; look at the popup's last successful timestamp. Timer capture is diagnostic and is not used by the optimizer.

## Decision engine

- Raw public downloads have URL, retrieval timestamp, SHA-256 and retry/stale-cache metadata. A failed optional source does not erase the last good catalog.
- Historical regular-season stats use three years with recency weighting and efficiency shrinkage. Player GSIS IDs anchor joins; ESPN IDs come from nflverse identity/roster data. Current rosters add previously unseen players as conspicuously uncertain priors.
- Expanding-window model comparison evaluates the opportunity/efficiency baseline, persistence and an ExtraTrees workload model. The tree predicts attempts, carries, targets and expected games, then regressed historical efficiency converts opportunity into scoring components. Model selection uses common out-of-sample folds on reference PPR scoring. The production catalog uses the winning model, not automatically the most complex model. Team target/carry/pass budgets are reconciled to prior team volume.
- The independent model and imported market projection remain separate. Market weight can be fitted to aligned historical forecast errors, accounting for their covariance. Without those errors, weight defaults to zero; a user-set provisional weight is clearly labeled.
- Seeded workload/efficiency/game-availability draws produce distribution summaries and position top-3/5/12 probabilities. Rolling-origin residuals provide additional bias/interval calibration for modeled veteran projections. Bonuses, rookies, role adjustments and market blends do not have independently demonstrated coverage. Bust means below 65% of modeled mean; breakout means above 135%. These are explicit definitions, not universal labels.
- Starter replacement uses a global assignment of league starting slots, including FLEX competition. Bench replacement is a documented positional-depth approximation. Quick rankings combine marginal lineup value, depth, upside, scarcity and value over next pick.
- ADP survival uses a conditional probabilistic distribution rather than deterministic ADP order, with shrunk adjustments for observed reaches and recent position runs. No survival probability is invented when ADP is absent.
- Batched Monte Carlo drafts sample imperfect opponents, rank affinity and position preferences. A Hall-constraint check guarantees enough remaining picks to fill all starting slots. Every player can be drafted only once.
- The **top six fast-ranked candidates** receive deeper simulations (API supports up to 12). These comparisons hold continuation policy and random seed fixed. The current rollout uses constrained ADP continuation; the experimental adaptive policy remains a separate benchmark because it underperformed on the synthetic fixture.
- Weekly seasons model shared NFL-team shocks, persistent season-quality uncertainty, correlated missed-game blocks, byes, legal starting lineups, round-robin matchups, playoffs and a title. Lineups are chosen using projected available-player value, not hindsight realized scores. Schedule, health and covariance assumptions remain simplified.
- The UI shows an immediate heuristic and can refine on your turn. Job results carry draft/catalog/settings/event fingerprints; stale simulations are not applied. Championship delta is in **percentage points vs the displayed baseline**, with a paired Monte Carlo interval. This interval excludes model error and candidate-selection bias.

Historical validation during implementation selected the workload tree: common-fold MAE **40.1**, versus **42.0** persistence and **51.0** initial opportunity model. Prospective empirical P10–P90 intervals covered **76.2%, 77.6%, 84.1%** in 2023–2025. These are broad veteran-population reference-PPR results; they do not establish draft-strategy profit or championship advantage. Full records are in local `storage/model_2026.json` and `storage/calibration.json`.

## Imports and season management

Interactive API schemas: **http://127.0.0.1:8000/docs**. Authenticated calls need `X-Local-Token`; the local frontend pairs itself through `/api/bootstrap`. To use API docs, provide the header through an HTTP client or use the frontend JSON import forms.

Market import (`POST /api/market/2026`, or **Market / ADP** in the UI):

```json
{
  "source": "Your licensed projection export",
  "as_of": "2026-09-06T12:00:00Z",
  "provisional_weight": 0,
  "rows": [{
    "id": "COPY_A_CANONICAL_ID_FROM_THE_PLAYER_TABLE",
    "adp": 24.5,
    "adp_sd": 9,
    "market_stats": {"receptions": 80, "receiving_yards": 1050, "receiving_tds": 7}
  }]
}
```

Use complete stat projections when ensembling; omitted market stat components are treated as zero, not as borrowed independent stats. For fitted weights supply `independent_errors` and `market_errors` with at least 30 aligned pre-season out-of-sample errors in the same units and league scoring. Do not use in-sample residuals or post-draft information.

Player import (`POST /api/players/2026`): `{"players":[...]}` with full Player objects. Require `id`, `name`, `position`, `stats`, and an explicit `source`. `ids` can contain `gsis`, `espn`, `fantasypros`. Conflicting IDs reject the whole import. Export a player first to see a complete editable object. Stable canonical IDs are mandatory.

Structured event (`POST /api/events`, or **Structured event**):

```json
{
  "id": "source-event-unique-id",
  "player_id": "COPY_CANONICAL_ID",
  "kind": "injured",
  "occurred_at": "2026-09-06T12:00:00Z",
  "known_at": "2026-09-06T12:30:00Z",
  "source_url": "https://your-authoritative-source.example/report",
  "note": "Describe verified evidence and why this adjustment is justified",
  "workload_multiplier": 1,
  "efficiency_multiplier": 1,
  "games_delta": -2,
  "confirmed": false
}
```

Only confirmed events known by the forecast cutoff affect projections. Different event IDs compound; update the same ID to correct an existing event. This is a transparent quantitative override framework, not an automated injury prognosis. No LLM produces numeric rankings. RSS headlines can be cached as evidence; they do not automatically change player values.

After drafting, set the league's `mode` to `SEASON`. Import a current complete ownership snapshot in **Data & models → Current team rosters**, or `PUT /api/leagues/ID/rosters`:

```json
{"0":["canonical-player-id"],"1":[],"2":[],"3":[]}
```

Include every team index. This updates season ownership without rewriting draft history. Until imported, season tools use draft ownership and can therefore become stale after transactions.

- **Start/sit:** the new My lineup flow combines injury evidence, weekly projections, recent usage, eligibility and known game locks. The older Season management view remains a simpler baseline. See docs/WEEKLY_LINEUPS.md.
- **Waivers:** available-player ROS replacement value, immediate lineup gain and breakout probability. FAAB is a bounded budget heuristic, not a validated optimal bidding model. Current roster data are essential.
- **Trades:** equal-count, two-team swaps, lineup effects and distributional championship comparison when full league rosters exist before week 1. Unequal trades need explicit accompanying drops. Midseason title odds require standings/remaining schedule not yet imported, so the API returns no title estimate for those cases.

## Refresh, replay and tests

Public refresh runs every six hours while the application is running, for configured real seasons. There is no machine-level scheduled task; if the app is closed, it does not refresh. Manual refresh is available in the UI or CLI. Failed refresh details are saved locally.

```powershell
.\.venv\Scripts\python.exe -m app.cli refresh --season 2026
.\.venv\Scripts\python.exe -m app.cli backtest
.\.venv\Scripts\python.exe -m app.cli calibration
.\.venv\Scripts\python.exe -m pytest -q
npm --prefix frontend test
npm --prefix frontend run build
.\.venv\Scripts\python.exe scripts/replay.py
.\.venv\Scripts\python.exe scripts/validate.py 1000
```

`replay.py` builds a complete synthetic 12-team ESPN-format draft and sends every snapshot through the actual API in an isolated temporary database. It tests authentication, identity resolution, ingestion, recommendations, idempotency and truncation rejection. This is distinct from testing the extension against a real ESPN page.

`validate.py` runs 1,000 simulated drafts for each of ADP, ESPN rank, ECR, VORP, BPA and adaptive policies, plus six candidate comparisons. Its synthetic market ranks are identical, so ADP/ESPN/ECR equivalence is expected. It is a reproducibility/performance benchmark, **not historical strategy validation**. Results go to `storage/validation.json`.

Browser smoke check, with the server running and Google Chrome installed:

```powershell
cd frontend
node e2e.mjs
```

The app saves desktop/mobile screenshots in `storage`. No browser account is accessed by these headless tests.

## Project structure and persistence

```text
app/domain.py       Typed league, player, pick and event contracts
app/storage.py      SQLite/WAL transactions and catalog versions
app/data.py         Public source cache and historical normalization
app/identity.py     Explicit cross-provider ID reconciliation
app/projections.py  Opportunity/efficiency distributions and scoring
app/learning.py     Workload trees and rolling model selection
app/calibration.py  Prospective residual interval audit
app/special.py      K/DST components and team volume constraints
app/draft.py        Survival and fast candidate recommendations
app/simulation.py   Batched drafts and season/playoff simulations
app/season.py       Lineups, waivers and trades
app/ownership.py    Season roster snapshots
app/espn.py         Strict ESPN-to-canonical state normalization
app/news.py         Public headline evidence cache
app/api.py          Loopback API, background jobs and refresh loop
frontend/           React/TypeScript application
extension/          Read-only Chrome extension and DOM parser tests
scripts/            Full API replay and benchmark commands
tests/              Core, API, modeling and roster constraint tests
storage/            Generated SQLite, Parquet, reports and screenshots
```

Back up the whole `storage` directory with the app stopped (SQLite WAL may otherwise contain recent writes). Export each league through the Draft board as an additional portable snapshot. Draft and catalog state persist across restarts; in-flight jobs do not.

## Remaining work before production use

The following requested capabilities are incomplete and should not be mistaken for implemented/validated features:

- Your exact two ESPN configurations and a real logged-in draft-room acceptance test. The new snapshot bridge imports league settings, current rosters and weekly forecasts. Reliable virtualized draft-history extraction and continuous free-agent sync remain incomplete.
- Licensed FantasyPros/ECR/ADP history and current feeds; historical draft-policy backtests with vintage-correct news, rosters and market snapshots. No measurable championship advantage is established.
- Automated routes/participation, snap trends, practice/injury/suspension feeds, coaching/QB/line changes, rookie draft-capital models, sportsbook inputs, strength-of-schedule and playoff matchup models. nflverse source breadth does not mean all those features are modeled.
- Full correlated distribution calibration, manager-specific learned draft behavior, exhaustive multi-step optimization over all legal candidates, adaptive contingency trees, calibrated confidence and real-league season simulations with standings/schedules.
- Auction/keeper/traded-pick rules, independently modeled full ESPN DST scoring bands, automatic transactions, multi-player add/drop trades and optimized FAAB bidding. Weekly lineups preserve known locks and use ESPN-scored projections for unsupported independent scoring.

These gaps are material for a production-quality system. The included application provides executable engines, diagnostics and explicit input boundaries to continue that work; it must not be represented as satisfying every requirement in the original brief.

## Sources

- [nflverse public data releases](https://github.com/nflverse/nflverse-data/releases): local cached player/team weekly stats, rosters, schedules and player identity data.
- [nflreadr player identity documentation](https://nflreadr.nflverse.com/reference/load_players.html): GSIS primary keys and ESPN identity fields.
- [nflreadr roster documentation](https://nflreadr.nflverse.com/reference/load_rosters.html).
- [ESPN NFL RSS](https://www.espn.com/espn/rss/nfl/news): optional public headline evidence only.

Source rights and availability remain those of the respective providers. No premium feed is scraped or assumed accessible.
