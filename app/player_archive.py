"""Archive a projection for every relevant player on every refresh, not just lineup players.

Projections used to be saved only when someone asked for a lineup, and only for the players
on that roster, so most players were never predicted and never graded. This module projects
every player with a game in the current week under a fixed, league-independent scoring rule
and archives each one before kickoff, so the learning loop has a complete record to grade.

Nothing here reads a result: projections are written only for games that have not started,
and :func:`app.tracking.settle_players` grades the last version saved before kickoff.
"""
from __future__ import annotations
import logging
from datetime import datetime, timezone
from app.domain import League, DEFAULT_SCORING
from app.storage import DATA
from app.weekly import project_week

log = logging.getLogger(__name__)
ARCHIVE_LEAGUE = '__all_players__'
POSITIONS = ('QB', 'RB', 'WR', 'TE', 'K', 'DST')
DRAWS = 1024  # Smaller than a lineup request: this runs for every player on every refresh.


def archive_league(season):
    """A fixed scoring rule so archived projections stay comparable across weeks and leagues."""
    return League(id=ARCHIVE_LEAGUE, name='All players', season=season, scoring=dict(DEFAULT_SCORING), bonuses=[])


def relevant_players(players, board, week):
    """Players with a recognised position whose team plays a game that has not kicked off."""
    playing = set()
    for g in board:
        if g.get('state') != 'pre' or not g.get('kickoff'): continue
        for side in ('home_team', 'away_team'):
            if g.get(side): playing.add(g[side])
    return [p for p in players if p.position in POSITIONS and p.team in playing]


def ensure_catalog(store, season, week, cache_path=None):
    """Make sure the catalog holds every NFL player with a stat history, not only synced ESPN rosters.

    An ESPN sync adds bare roster entries with no history, which cannot be projected
    independently. The nflverse history is folded into those entries (keeping their ids, so
    rosters and archived forecasts still resolve) and everyone else is added. Rebuilt once per
    week, or whenever most of the catalog has no history. Never raises.
    """
    from app.intelligence import get_meta, set_meta
    cache_path = cache_path or DATA/'cache'
    try:
        existing = store.players(season)
        skill = [p for p in existing if p.position in ('QB', 'RB', 'WR', 'TE')]
        covered = sum(bool(p.stats) for p in skill)
        built_for = get_meta(store, f'catalog-built:{season}')
        from app.identity import Identity, merge_catalog, referenced_ids
        try:
            Identity(existing); consistent = True
        except ValueError:
            consistent = False   # a duplicated player: rebuild now so the catalog is repaired
        if consistent and built_for == week and len(skill) >= 300 and covered >= .8*len(skill):
            return {'status': 'current', 'players': len(existing)}
        from app.projections import build_catalog
        built = build_catalog(season, cache_path)
        players = merge_catalog(existing, built, referenced_ids(store))
        Identity(players)
        store.save_players(season, players)
        set_meta(store, f'catalog-built:{season}', week)
        return {'status': 'rebuilt', 'players': len(players), 'with_history': sum(bool(p.stats) for p in players)}
    except Exception as exc:  # noqa: BLE001 - projections fall back to whatever catalog exists
        log.warning('catalog could not be completed: %s', exc)
        return {'status': 'error', 'error': str(exc)}


def projection_context(store, league, week):
    """Everything a league projection uses: week context, blend weights, learned correction and point model."""
    from app.intelligence import weekly_context
    from app.tracking import blend_weight
    from app.player_learning import fit_correction
    from app.player_model import cached_fit, current_features, upcoming_players
    context = weekly_context(store, league, week)
    context['blend_weights'] = {pos: blend_weight(store, pos)[0] for pos in POSITIONS}
    context['player_correction'] = fit_correction(store)
    try:
        context['player_point_model'] = cached_fit()
        context['player_model_features'] = current_features(league.season, week, None, upcoming_players(store, league.season, context.get('games') or [])) if context['player_point_model'] else None
    except Exception:  # noqa: BLE001
        context['player_point_model'] = None; context['player_model_features'] = None
    return context


def archive_leagues(store, season, week, health, context=None, cache_path=None, as_of=None, draws=DRAWS, only=None):
    """Archive every rostered player of every synced ESPN league under that league's scoring.

    These rows carry ESPN's own projection, so they are the ones that allow a like-for-like
    accuracy comparison with ESPN once games finish. Never raises.
    """
    from app.tracking import save_player_rows, blend_weight
    from app.league_sync import sync_metadata
    from app.scoring_support import player_scoring
    cache_path = cache_path or DATA/'cache'
    as_of = as_of or datetime.now(timezone.utc)
    out = {}
    try:
        by_id = {p.id: p for p in store.players(season)}
        for league in store.leagues():
            if league.source != 'ESPN league snapshot' or league.season != season or (only and league.id != only): continue
            meta = sync_metadata(store, league.id)
            if meta.get('week') != week: continue
            league_context = projection_context(store, league, week) if context is None else dict(context) | {'blend_weights': {pos: blend_weight(store, pos)[0] for pos in POSITIONS}}
            rows = []
            for pid, provider in meta.get('weekly', {}).items():
                player = by_id.get(pid)
                if player is None: continue
                # The snapshot may be hours old: never reuse its lock state or banked points.
                provider = {k: v for k, v in provider.items() if k not in ('locked', 'actual_points')}
                try:
                    scoring_league, incomplete = player_scoring(league, player, meta, provider)
                    provider['independent_scoring_incomplete'] = incomplete
                    injury = health.get('players', {}).get(player.ids.get('espn', ''), {})
                    row, _ = project_week(player, scoring_league, week, provider, injury, health.get('status', 'unknown'), cache_path, draws=draws, context=league_context)
                except Exception:  # noqa: BLE001
                    continue
                if row is not None and not row.get('game', {}).get('started'): rows.append(row)
            out[league.id] = {'projected': len(rows), 'saved': save_player_rows(store, league, week, rows, stamp=as_of)}
    except Exception as exc:  # noqa: BLE001
        log.warning('league archive failed: %s', exc)
        out['error'] = str(exc)
    return out


def espn_default_projections(store, season, week):
    """ESPN's projected stat lines from synced leagues, rescored under the archive's default rules.

    ESPN's headline number uses each league's own scoring, so it cannot be compared with the
    all-player archive directly. Its projected stat line can. Kickers and defenses are left out
    because their ESPN stat lines do not map onto the default rules completely.
    """
    from app.league_sync import sync_metadata
    out = {}
    try:
        positions = {p.id: p.position for p in store.players(season)}
        for league in store.leagues():
            if league.source != 'ESPN league snapshot' or league.season != season: continue
            meta = sync_metadata(store, league.id)
            if meta.get('week') != week: continue
            for pid, provider in meta.get('weekly', {}).items():
                stats = provider.get('projected_stats') or {}
                if positions.get(pid) not in ('QB', 'RB', 'WR', 'TE') or not stats: continue
                out[pid] = float(sum(DEFAULT_SCORING.get(k, 0)*v for k, v in stats.items()))
    except Exception as exc:  # noqa: BLE001
        log.warning('ESPN comparison projections unavailable: %s', exc)
    return out


def attach_espn(store, season, week, as_of=None):
    """After an ESPN sync, record ESPN's comparison number on already archived projections.

    Writes a new pregame version (same projection, updated ESPN value); games that have kicked
    off are never touched.
    """
    import json
    from app.tracking import initialize, date
    as_of = as_of or datetime.now(timezone.utc)
    espn = espn_default_projections(store, season, week)
    initialize(store); saved = 0
    with store.connect() as c:
        latest = {}
        for r in c.execute('SELECT id,player_id,game_id,kickoff,body FROM player_forecasts WHERE league_id=? ORDER BY id', (ARCHIVE_LEAGUE,)):
            latest[(r['player_id'], r['game_id'])] = r
        for (pid, gid), r in latest.items():
            body = json.loads(r['body'])
            value = espn.get(pid)
            if value is None or body.get('season') != season or body.get('week') != week or as_of >= date(r['kickoff']): continue
            old = body.get('espn_projection')
            if old is not None and abs(old-value) < 1e-6: continue
            body['espn_projection'] = value
            c.execute('INSERT INTO player_forecasts(league_id,player_id,game_id,created_at,kickoff,body) VALUES(?,?,?,?,?,?)',
                      (ARCHIVE_LEAGUE, pid, gid, as_of.isoformat(), r['kickoff'], json.dumps(body, allow_nan=False)))
            saved += 1
    return saved


def archive_projections(store, season, week, board, health, context, cache_path=None, as_of=None, draws=DRAWS):
    """Project and archive every relevant player. Never raises: a failure must not stop a refresh."""
    from app.tracking import save_player_rows
    cache_path = cache_path or DATA/'cache'
    as_of = as_of or datetime.now(timezone.utc)
    league = archive_league(season)
    try:
        players = relevant_players(store.players(season), board, week)
    except Exception as exc:  # noqa: BLE001
        log.warning('player archive could not read the catalog: %s', exc)
        return {'status': 'error', 'error': str(exc), 'players': 0, 'saved': 0}
    rows, failed = [], 0
    espn = espn_default_projections(store, season, week)
    for player in players:
        injury = health.get('players', {}).get(player.ids.get('espn', ''), {}) if health.get('season', season) == season else {}
        try:
            row, _ = project_week(player, league, week, {}, injury, health.get('status', 'unknown'), cache_path, draws=draws, context=context)
        except Exception:  # noqa: BLE001 - one unprojectable player must not lose the rest
            failed += 1
            continue
        if row is not None:
            # Comparison only: ESPN's number never enters this independent projection.
            row['espn_projection'] = espn.get(player.id)
            rows.append(row)
    saved = save_player_rows(store, league, week, rows, stamp=as_of)
    return {'status': 'ok', 'league_id': ARCHIVE_LEAGUE, 'players': len(players), 'projected': len(rows),
            'saved': saved, 'unprojectable': failed, 'draws': draws, 'as_of': as_of.isoformat(),
            'note': 'Every player with an upcoming game is projected under default scoring and archived before kickoff.'}
