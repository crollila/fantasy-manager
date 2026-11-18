"""One-time cleanup of archived versions that never should have been stored.

Before 0.7.3 every refresh (one per quarter hour while the app is open) wrote a new copy of
each finished game's review, each upcoming game's forecast and most player projections, because
a "fetched at" stamp or a third-decimal refit wobble made the row look new. This applies the
current save rules to the rows already on disk and returns the space to the system.

Nothing that is graded or cited is removed: the first version, the final pregame version, every
version a result points at and every version that differs from its predecessor are kept as
they are.
"""
import json
import logging
import sqlite3
from app.storage import now
from app.tracking import REVIEW_STAMPS, content_fingerprint, date, unchanged, version_due

VERSION = 1
KEY = 'ledger-compaction'
log = logging.getLogger(__name__)


def _pregame(created_at, kickoff):
    try: return date(created_at) < date(kickoff)
    except (ValueError, TypeError): return False


def _result_versions(c):
    """Keep the first copy of each distinct review per (game, forecast)."""
    seen, drop, keep = set(), [], []
    for r in c.execute('SELECT id,game_id,forecast_id,body FROM game_result_versions ORDER BY id'):
        try: fingerprint = content_fingerprint(json.loads(r['body']), REVIEW_STAMPS)
        except ValueError: continue
        key = (r['game_id'], r['forecast_id'], fingerprint)
        if key in seen: drop.append((r['id'],))
        else: seen.add(key); keep.append((fingerprint, r['id']))
    return drop, keep


def _game_forecasts(c):
    protected = {r[0] for r in c.execute('SELECT forecast_id FROM game_result_versions')}
    protected |= {r[0] for r in c.execute("SELECT json_extract(body,'$.forecast_id') FROM game_results")}
    final = {}
    for r in c.execute('SELECT id,game_id,created_at,kickoff FROM game_forecasts ORDER BY id'):
        final[(r['game_id'], 'last')] = r['id']
        if _pregame(r['created_at'], r['kickoff']): final[(r['game_id'], 'pregame')] = r['id']
    protected |= set(final.values())
    drop, keep, cited, game, kept = [], [], set(), None, None
    for r in c.execute('SELECT id,game_id,created_at,body FROM game_forecasts ORDER BY game_id,id'):
        if r['game_id'] != game: game, kept = r['game_id'], None
        try:
            body = json.loads(r['body']); fingerprint = content_fingerprint(body); stamp = date(r['created_at'])
            due = kept is None or r['id'] in protected or (fingerprint != kept[0] and version_due(kept[1], kept[2], body, stamp))
        except (ValueError, TypeError, KeyError):
            kept = None; continue  # An unreadable row is left alone.
        if due:
            kept = (fingerprint, body, stamp); keep.append((fingerprint, r['id']))
            cited.update(filter(None, [body.get('base_artifact'), (body.get('learning') or {}).get('active_artifact'), ((body.get('learning') or {}).get('candidate') or {}).get('artifact')]))
        else: drop.append((r['id'],))
    return drop, keep, cited


def _player_forecasts(c):
    protected = {r[0] for r in c.execute('SELECT forecast_id FROM player_results')}
    final = {}
    for r in c.execute('SELECT id,league_id,player_id,game_id,created_at,kickoff FROM player_forecasts ORDER BY id'):
        key = (r['league_id'], r['player_id'], r['game_id'])
        final[key + ('last',)] = r['id']
        if _pregame(r['created_at'], r['kickoff']): final[key + ('pregame',)] = r['id']
    protected |= set(final.values())
    drop, key, kept = [], None, None
    for r in c.execute('SELECT id,league_id,player_id,game_id,body FROM player_forecasts ORDER BY league_id,player_id,game_id,id'):
        current = (r['league_id'], r['player_id'], r['game_id'])
        if current != key: key, kept = current, None
        try: body = json.loads(r['body'])
        except ValueError:
            kept = None; continue
        if kept is None or r['id'] in protected or not unchanged(kept, body): kept = body
        else: drop.append((r['id'],))
    return drop


def _artifacts(c, cited):
    cited = set(cited)
    for r in c.execute("SELECT json_extract(body,'$.base_artifact'),json_extract(body,'$.learning_artifact') FROM game_results"): cited.update(filter(None, r))
    row = c.execute("SELECT value FROM meta WHERE key='game-learning'").fetchone()
    if row:
        state = json.loads(row[0]); cited.update(filter(None, [state.get('active'), state.get('pending')]))
    return [(r[0],) for r in c.execute("SELECT id FROM game_model_artifacts WHERE kind='foundation'") if r[0] not in cited]


def compact(store):
    """Remove redundant archived versions. Safe to run more than once."""
    from app.game_learning import initialize
    initialize(store)
    with store.connect() as c:
        result_drop, result_keep = _result_versions(c)
        game_drop, game_keep, cited = _game_forecasts(c)
        player_drop = _player_forecasts(c)
        artifact_drop = _artifacts(c, cited)
    removed = {'game_result_versions': len(result_drop), 'game_forecasts': len(game_drop),
               'player_forecasts': len(player_drop), 'game_model_artifacts': len(artifact_drop)}
    with store.connect() as c:
        c.execute('BEGIN IMMEDIATE')
        c.executemany('DELETE FROM game_result_versions WHERE id=?', result_drop)
        c.executemany('UPDATE game_result_versions SET fingerprint=? WHERE id=?', result_keep)
        c.executemany('DELETE FROM game_forecasts WHERE id=?', game_drop)
        c.executemany('UPDATE game_forecasts SET fingerprint=? WHERE id=?', game_keep)
        c.executemany('DELETE FROM player_forecasts WHERE id=?', player_drop)
        c.executemany('DELETE FROM game_model_artifacts WHERE id=?', artifact_drop)
        c.execute('INSERT OR REPLACE INTO meta VALUES(?,?)', (KEY, json.dumps({'version': VERSION, 'at': now(), 'removed': removed, 'vacuumed': False})))
    return removed


def vacuum(store):
    """Hand the freed pages back to the disk; deleting rows alone leaves the file its old size."""
    c = sqlite3.connect(store.db, timeout=120, isolation_level=None)
    try:
        c.execute('VACUUM'); c.execute('PRAGMA wal_checkpoint(TRUNCATE)')
    finally: c.close()
    with store.connect() as c:
        row = c.execute('SELECT value FROM meta WHERE key=?', (KEY,)).fetchone()
        c.execute('INSERT OR REPLACE INTO meta VALUES(?,?)', (KEY, json.dumps((json.loads(row[0]) if row else {'version': VERSION}) | {'vacuumed': True, 'vacuumed_at': now()})))


def status(store):
    with store.connect() as c:
        row = c.execute('SELECT value FROM meta WHERE key=?', (KEY,)).fetchone()
    return json.loads(row[0]) if row else {}


def maybe_compact(store, progress=None):
    """Run the cleanup once per data directory. Never raises: a refresh must still go ahead."""
    try:
        state = status(store)
        if state.get('version', 0) < VERSION:
            if progress: progress('Removing duplicate copies from the saved forecast history (one time)')
            removed = compact(store); state = status(store)
            log.info('ledger compaction removed %s', removed)
        if not state.get('vacuumed'):
            if progress: progress('Reclaiming disk space from the saved forecast history (one time)')
            vacuum(store)
        return status(store)
    except Exception as exc:  # noqa: BLE001
        log.warning('ledger compaction did not finish; it is retried on the next refresh: %s', exc)
        return {}
