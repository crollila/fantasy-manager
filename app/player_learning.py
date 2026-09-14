"""Weekly correction of player point projections, refit from graded forecasts.

Unlike the game correction in :mod:`app.game_learning`, this layer has no promotion
gate: it is refit on every weekly run and applied immediately. It stays honest
because it can only ever train on games that have already finished, while it is only
ever applied to games that have not started, so no recorded forecast is influenced by
its own result. Its size is shrunk toward zero by sample count, so early weeks move
projections barely at all and later weeks move them more.
"""
from __future__ import annotations
import json
import numpy as np
from app.game_model import ridge_fit, predict

FAMILY = 'player-correction-1'
POSITIONS = ('QB', 'RB', 'WR', 'TE', 'K', 'DST')
PRIOR_STRENGTH = 150  # Graded forecasts needed before a position's correction reaches half its fitted size.
MAX_FRACTION = .25    # A correction never moves a projection by more than a quarter of itself,
MAX_POINTS = 4.       # nor by more than this many fantasy points.
MIN_OBSERVATIONS = 20


def features(position, center, play_probability, independent_weight):
    """Inputs known strictly before kickoff, so a correction never depends on the outcome."""
    x = {'level': float(np.clip((center-10)/10, -2, 6)),
         'play_probability': float(np.clip(play_probability, 0, 1)),
         'independent_weight': float(np.clip(independent_weight, 0, 1))}
    for p in POSITIONS:
        if position == p: x['position:'+p] = 1.
    return x


def fit_correction(store, as_of=None):
    """Refit on every graded player forecast available now. Returns a model or None."""
    from app.tracking import initialize
    initialize(store)
    with store.connect() as c:
        rows = [json.loads(r[0]) for r in c.execute('SELECT body FROM player_results')]
    xs, ys, counts = [], [], {}
    for r in rows:
        position = r.get('position')
        if position not in POSITIONS: continue
        forecast, actual = r.get('forecast'), r.get('actual')
        if forecast is None or actual is None: continue
        if not (np.isfinite(forecast) and np.isfinite(actual)): continue
        play = r.get('play_probability'); weight = r.get('independent_weight')
        xs.append(features(position, forecast, 1. if play is None else play, .65 if weight is None else weight))
        ys.append(float(actual)-float(forecast))
        counts[position] = counts.get(position, 0)+1
    if len(ys) < MIN_OBSERVATIONS: return None
    model = ridge_fit(xs, ys, alpha=25)
    model.update(family=FAMILY, observations=len(ys), by_position=counts,
                 fitted_at=(as_of.isoformat() if as_of is not None else None),
                 mean_residual=float(np.mean(ys)), residual_sd=float(np.std(ys)),
                 method='Ridge on (actual - projected) fantasy points from graded pregame forecasts; '
                        'alpha=25; shrunk toward no correction by graded count per position; '
                        f'bounded to {MAX_FRACTION:.0%} of the projection and {MAX_POINTS:.0f} points.')
    return model


def adjustment(model, position, center, play_probability=1., independent_weight=.65):
    """Points to add to a projection, shrunk by how much graded evidence that position has."""
    if not model or model.get('family') != FAMILY: return 0.
    graded = model.get('by_position', {}).get(position, 0)
    if graded <= 0: return 0.
    shrink = graded/(graded+PRIOR_STRENGTH)
    raw = predict(model, features(position, center, play_probability, independent_weight))
    if not np.isfinite(raw): return 0.
    limit = min(MAX_POINTS, MAX_FRACTION*abs(float(center)))
    return float(np.clip(raw*shrink, -limit, limit))


def summary(model):
    if not model: return {'available': False, 'observations': 0,
                          'note': f'A correction begins once {MIN_OBSERVATIONS} player forecasts have been graded.'}
    return {'available': True, 'observations': model['observations'], 'by_position': model.get('by_position', {}),
            'mean_residual': model.get('mean_residual'), 'residual_sd': model.get('residual_sd'),
            'prior_strength': PRIOR_STRENGTH, 'max_fraction': MAX_FRACTION, 'max_points': MAX_POINTS,
            'method': model.get('method'),
            'limits': 'Refit every week and applied immediately, with no held-out promotion test. '
                      'It can chase noise; the graded record in Accuracy is the only check on whether it helps.'}
