"""The promoted player projection model: snaps + red zone + expected passing volume.

Chosen by walk-forward ablation over 2023-2025 as the *smallest* family set clearing the
promotion standard (overall MAE gain >= 0.15, bootstrap lower bound > 0, no position
degraded). It beat the full 267-feature candidate on parsimony, not accuracy: 118 features
for +0.174 MAE against 267 for +0.216.

Rejected families: ``market``, ``usage`` and ``defense`` are excluded because the same
search showed the set is substitutable and these three are not needed to clear the bar.
Adding them costs 149 features to buy a further 0.042.

Applied as a **ratio**, not a replacement. The model is trained on default scoring, so
multiplying a league's own baseline by ``prediction / control`` carries the learned matchup
and role information across to custom scoring rules without re-training per league, and
leaves production's simulation, p10/p90 intervals, boom/bust thresholds and participation
logic untouched. The new quantile intervals are deliberately NOT shipped: they held only
71% coverage against an 80% target.
"""
from __future__ import annotations
import logging
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from app.player_features import build, family_columns
from app.storage import DATA

log = logging.getLogger(__name__)

FAMILIES = ('snaps', 'redzone', 'passrate')
POSITIONS = ('QB', 'RB', 'WR', 'TE')
MIN_TRAIN_ROWS = 2000
# A learned ratio never moves a projection by more than this, so a thin or odd feature row
# cannot produce an absurd number in the lineup.
MIN_RATIO, MAX_RATIO = .6, 1.6
MODEL_ID = 'player-point-model-1'


def feature_columns(frame):
    fam = family_columns(frame)
    columns = ['control']
    for name in FAMILIES:
        columns += fam.get(name, [])
    return [c for c in columns if c in frame.columns]


def fit(seasons=None, cache_path=None, as_of=None):
    """Train one model per position on every completed player-week available now."""
    from lightgbm import LGBMRegressor
    from app.player_backtest import prepare
    as_of = as_of or datetime.now(timezone.utc)
    season = as_of.year if as_of.month >= 3 else as_of.year - 1
    seasons = seasons or range(season - 7, season + 1)
    frame = prepare(seasons, cache_path)
    if frame.empty:
        return None
    columns = feature_columns(frame)
    models, counts = {}, {}
    for position in POSITIONS:
        part = frame[frame.position == position]
        if len(part) < MIN_TRAIN_ROWS:
            continue
        model = LGBMRegressor(objective='l2', n_estimators=300, num_leaves=15, learning_rate=.05,
                              min_child_samples=40, subsample=.8, subsample_freq=1,
                              colsample_bytree=.7, reg_lambda=5., random_state=11, n_jobs=4, verbose=-1)
        model.fit(part[columns].astype(float), part['actual'].astype(float))
        models[position] = model
        counts[position] = int(len(part))
    if not models:
        return None
    return {'model_id': MODEL_ID, 'families': list(FAMILIES), 'columns': columns, 'models': models,
            'training_rows': counts, 'trained_at': as_of.isoformat(),
            'seasons': [int(min(seasons)), int(max(seasons))]}


_CACHE = {}


def data_fingerprint(cache_path=None):
    """Cheap stamp of the inputs, so a refit happens exactly when new results arrive."""
    from pathlib import Path
    root = Path(cache_path or DATA/'cache')
    parts = []
    for path in sorted(root.glob('stats_*.parquet')) + sorted(root.glob('snaps_*.parquet')):
        try:
            parts.append((path.name, path.stat().st_mtime_ns, path.stat().st_size))
        except OSError:
            continue
    return hash(tuple(parts))


def cached_fit(cache_path=None, as_of=None, force=False):
    """Fit once per data version. Training takes ~20s, far too slow for a lineup request."""
    import os
    if os.environ.get('FANTASY_DISABLE_AUTO_REFRESH') == '1' and not force:
        return None   # keeps tests and short-lived processes hermetic and fast
    key = data_fingerprint(cache_path)
    if not force and _CACHE.get('key') == key:
        return _CACHE.get('bundle')
    bundle = fit(cache_path=cache_path, as_of=as_of)
    _CACHE.update(key=key, bundle=bundle)
    return bundle


def current_features(season, week, cache_path=None, players=None):
    """Pregame-safe feature rows for the upcoming week, indexed by nflverse player id.

    The upcoming week has not been played, so it has no box score and would otherwise have no
    row at all. ``players`` names who is due to play and those rows are created empty, then
    filled from completed weeks only.
    """
    upcoming = None
    if players is not None and len(players):
        upcoming = {'season': season, 'week': week, 'players': players}
    frame = build(range(season - 7, season + 1), cache_path, upcoming=upcoming)
    if frame.empty:
        return pd.DataFrame()
    rows = frame[(frame.season == season) & (frame.week == week)]
    return rows.set_index('player_id') if not rows.empty else pd.DataFrame()


def upcoming_players(store, season, board):
    """Who is due to play in a game that has not kicked off, as the feature builder needs them."""
    playing = {}
    for g in board:
        if g.get('state') != 'pre':
            continue
        for side, other in (('home_team', 'away_team'), ('away_team', 'home_team')):
            if g.get(side):
                playing[g[side]] = g.get(other)
    rows = [{'player_id': p.ids.get('gsis', p.id), 'position': p.position, 'team': p.team,
             'opponent': playing[p.team]}
            for p in store.players(season) if p.position in POSITIONS and p.team in playing]
    return pd.DataFrame(rows).drop_duplicates('player_id') if rows else pd.DataFrame()


def ratio(bundle, features, position, control):
    """How much the learned model moves this player's projection, as a multiplier."""
    if not bundle or position not in bundle.get('models', {}) or features is None:
        return 1., None
    if control is None or not np.isfinite(control) or control <= .5:
        return 1., None   # a near-zero baseline makes the ratio meaningless
    row = {c: features.get(c, np.nan) for c in bundle['columns']}
    row['control'] = float(control)
    frame = pd.DataFrame([row], columns=bundle['columns']).astype(float)
    try:
        predicted = float(bundle['models'][position].predict(frame)[0])
    except Exception:  # noqa: BLE001 - a bad feature row must never break a projection
        return 1., None
    if not np.isfinite(predicted):
        return 1., None
    value = float(np.clip(predicted / control, MIN_RATIO, MAX_RATIO))
    return value, {'model_id': bundle['model_id'], 'predicted_default_points': round(predicted, 2),
                   'control_default_points': round(float(control), 2), 'ratio': round(value, 4)}


def control_points(features):
    """The current method's default-scoring projection for this player-week."""
    from app.domain import DEFAULT_SCORING
    if features is None:
        return None
    total = 0.
    for stat, weight in DEFAULT_SCORING.items():
        value = features.get(f'use_{stat}_ewm')
        if value is not None and np.isfinite(value):
            total += float(value) * weight
    return total


def summary(bundle):
    if not bundle:
        return {'available': False, 'note': 'The player point model trains once graded weeks exist.'}
    return {'available': True, 'model_id': bundle['model_id'], 'families': bundle['families'],
            'features': len(bundle['columns']), 'training_rows': bundle['training_rows'],
            'trained_at': bundle['trained_at'], 'seasons': bundle['seasons'],
            'method': 'Walk-forward validated point projection (snaps, red-zone opportunity, expected '
                      'passing volume). Applied as a ratio to the league baseline; production intervals '
                      'and participation logic are unchanged.',
            'limits': 'Validated on 2023-2025 holdout weeks: overall MAE 4.592 -> 4.418. The candidate '
                      'prediction intervals were not promoted (71% coverage against an 80% target).'}
