"""Automatic weekly refit of the NFL forecasting engine champion.

The engine's learners used to be refreshed only by running ``python -m app.nfl refit``
by hand at release time, so a shipped model stayed frozen for the rest of the season.
This module runs that refit from inside the ordinary refresh: once every regular-season
game in a week has a final score, the champion is refit on every completed game
available at that moment and promoted in the registry, and the very next forecast in
the same refresh loads the new champion.

State lives on disk (the registry card's ``trained_through`` plus a row in ``meta``), so a
restart resumes exactly where it left off and never repeats or loses a refit.
"""
from __future__ import annotations
import logging
from datetime import datetime, timezone

log = logging.getLogger(__name__)
STATE_KEY = 'nfl-engine-autorefit'


def last_completed_week(schedule, season):
    """Latest (season, week) whose every regular-season game has a final score."""
    import pandas as pd
    rows = [r for r in schedule.to_dict('records') if r.get('season') == season and r.get('game_type') == 'REG']
    weeks = {}
    for r in rows:
        done = not (pd.isna(r.get('home_score')) or pd.isna(r.get('away_score')))
        week = int(r['week'])
        weeks[week] = weeks.get(week, True) and done
    complete = [w for w, done in weeks.items() if done]
    return (int(season), max(complete)) if complete else None


def champion_trained_through(store):
    """What the live champion has already been trained on, from persisted state only."""
    try:
        from app.nfl.models.registry import Registry
        champion = Registry().champion('game_forecast')
    except Exception:  # noqa: BLE001 - a missing or unreadable registry simply means "unknown"
        champion = None
    if champion:
        through = champion.get('trained_through') or {}
        if through.get('season') is not None and through.get('week') is not None:
            return champion.get('model_id'), (int(through['season']), int(through['week']))
    from app.intelligence import get_meta
    state = get_meta(store, STATE_KEY) or {}
    if state.get('model_id') and champion and state['model_id'] == champion.get('model_id') and state.get('week') is not None:
        return champion.get('model_id'), (int(state['season']), int(state['week']))
    return (champion or {}).get('model_id'), None


def maybe_refit(store, schedule, season, as_of=None, progress=None):
    """Refit and promote the champion when a new NFL week has completed. Never raises."""
    from app.intelligence import get_meta, set_meta
    as_of = as_of or datetime.now(timezone.utc)
    completed = last_completed_week(schedule, season)
    model_id, trained_through = champion_trained_through(store)
    status = {'checked_at': as_of.isoformat(), 'completed_week': completed, 'model_id': model_id,
              'trained_through': trained_through}
    if completed is None:
        return status | {'action': 'no completed week yet'}
    if trained_through is not None and trained_through >= completed:
        return status | {'action': 'up to date'}
    previous = get_meta(store, STATE_KEY) or {}
    # A refit that already failed for this exact week is not retried every 15 minutes.
    if previous.get('failed_week') == list(completed) and previous.get('model_id') == model_id:
        return status | {'action': 'skipped after an earlier failure for this week', 'error': previous.get('error')}
    if progress: progress(f'Refitting the forecasting engine on completed games through {season} week {completed[1]}')
    try:
        from app.nfl.models.train import refit_champion
        result = refit_champion()
    except Exception as exc:  # noqa: BLE001 - a failed refit must never block forecasting
        log.warning('automatic engine refit failed: %s', exc)
        # Only a successful refit records a trained-through week: a failure must never be
        # mistaken for training the champion actually received.
        set_meta(store, STATE_KEY, {'model_id': model_id, 'failed_week': list(completed),
                                    'error': str(exc), 'at': as_of.isoformat()})
        return status | {'action': 'failed', 'error': str(exc)}
    set_meta(store, STATE_KEY, {'model_id': result['model_id'], 'season': completed[0], 'week': completed[1],
                                'refit_of': result.get('refit_of'), 'training_games': result.get('training_games'),
                                'at': as_of.isoformat()})
    log.info('automatic engine refit promoted %s', result['model_id'])
    return status | {'action': 'refit', 'model_id': result['model_id'], 'refit_of': result.get('refit_of'),
                     'training_games': result.get('training_games'), 'refit_at': as_of.isoformat()}
