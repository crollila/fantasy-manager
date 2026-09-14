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
    for player in players:
        injury = health.get('players', {}).get(player.ids.get('espn', ''), {}) if health.get('season', season) == season else {}
        try:
            row, _ = project_week(player, league, week, {}, injury, health.get('status', 'unknown'), cache_path, draws=draws, context=context)
        except Exception:  # noqa: BLE001 - one unprojectable player must not lose the rest
            failed += 1
            continue
        if row is not None: rows.append(row)
    saved = save_player_rows(store, league, week, rows, stamp=as_of)
    return {'status': 'ok', 'league_id': ARCHIVE_LEAGUE, 'players': len(players), 'projected': len(rows),
            'saved': saved, 'unprojectable': failed, 'draws': draws, 'as_of': as_of.isoformat(),
            'note': 'Every player with an upcoming game is projected under default scoring and archived before kickoff.'}
