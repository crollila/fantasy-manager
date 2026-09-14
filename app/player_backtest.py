"""Walk-forward backtest of the candidate player projection model against the current method.

The control reproduces what production computes today for the independent projection: a
prior-season weighted per-game baseline, blended with recent opportunity, scaled by the
fitted opponent/home/weather multiplier. The candidate is a small per-position model fitted
on that control plus the pregame-safe features in :mod:`app.player_features`.

Every fit is trained only on player-weeks whose games finished before the week being
predicted, so the comparison reflects what the system could actually have known.
"""
from __future__ import annotations
import logging
import numpy as np
import pandas as pd
from app.domain import DEFAULT_SCORING
from app.player_features import build, family_columns, read
from app.storage import DATA

log = logging.getLogger(__name__)
MIN_TRAIN_ROWS = 2000
POSITIONS = ('QB', 'RB', 'WR', 'TE')


def actual_points(stats, scoring=None):
    """Fantasy points actually scored, from the weekly box score."""
    scoring = scoring or DEFAULT_SCORING
    frame = stats.copy()
    frame['fumbles_lost'] = sum(frame.get(c, 0).fillna(0) for c in ('rushing_fumbles_lost', 'receiving_fumbles_lost', 'sack_fumbles_lost'))
    frame['two_point_conversions'] = sum(frame.get(c, 0).fillna(0) for c in ('passing_2pt_conversions', 'rushing_2pt_conversions', 'receiving_2pt_conversions'))
    points = np.zeros(len(frame))
    for key, value in scoring.items():
        if key in frame:
            points = points + frame[key].fillna(0).to_numpy(dtype=float) * value
    return points


def control_projection(frame):
    """The current production method's independent projection, reconstructed per player-week.

    Production scores a per-game stat line built from the player's weighted history and
    recent opportunity. Both are already present in the lagged frame as the exponentially
    weighted and season-to-date columns, so the control is that same stat line scored with
    the same rule — without the ESPN blend and injury multiplier, which have no historical
    record and are identical for both arms of the comparison.
    """
    out = np.zeros(len(frame))
    # Production blends the recent opportunity window into the season baseline at n/(n+4);
    # the exponentially weighted column is that same recency-weighted quantity.
    for stat, weight in DEFAULT_SCORING.items():
        col = f'use_{stat}_ewm'
        if col in frame:
            out = out + frame[col].fillna(0).to_numpy(dtype=float) * weight
    return out


def prepare(seasons=range(2019, 2027), cache_path=None):
    """Feature frame joined to the realised fantasy points for each player-week."""
    frame = build(seasons, cache_path)
    stats = pd.concat([read((cache_path or DATA/'cache')/f'stats_{y}.parquet') for y in seasons], ignore_index=True)
    stats = stats[(stats.season_type == 'REG') & (stats.position.isin(POSITIONS))].copy()
    stats['actual'] = actual_points(stats)
    keys = ['season', 'week', 'player_id']
    frame = frame.merge(stats[keys + ['actual']], on=keys, how='inner')
    frame['control'] = control_projection(frame)
    # Mirror what actually gets projected: a player with no prior workload is not forecast.
    prior = frame[[c for c in ('use_targets_std', 'use_carries_std', 'use_attempts_std') if c in frame]].fillna(0).sum(axis=1)
    frame = frame[prior > 0].copy()
    return frame


def fit_predict(train, test, features, position, seed=11, objective='l1'):
    """One small gradient-boosted model per position; falls back to ridge when data is thin."""
    from lightgbm import LGBMRegressor
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    x_train = train[features].astype(float)
    y_train = train['actual'].astype(float).to_numpy()
    x_test = test[features].astype(float)
    if len(train) < MIN_TRAIN_ROWS:
        scaler = StandardScaler()
        model = Ridge(alpha=10.)
        model.fit(scaler.fit_transform(x_train.fillna(0)), y_train)
        return model.predict(scaler.transform(x_test.fillna(0)))
    # The primary metric is MAE, and a squared-error fit shrinks toward the mean in a way
    # that measurably costs MAE on this target, so the default objective is L1.
    model = LGBMRegressor(objective=objective, n_estimators=300, num_leaves=15, learning_rate=.05,
                          min_child_samples=40, subsample=.8, subsample_freq=1, colsample_bytree=.7,
                          reg_lambda=5., random_state=seed, n_jobs=2, verbose=-1)
    model.fit(x_train, y_train)
    return model.predict(x_test)


def walk_forward(frame, features, eval_seasons, retrain_weeks=1, objective='l1'):
    """Predict each evaluation week using only player-weeks completed before it."""
    frame = frame.sort_values(['season', 'week']).copy()
    predictions = pd.Series(np.nan, index=frame.index)
    periods = sorted({(s, w) for s, w in zip(frame.season, frame.week) if s in eval_seasons})
    cache, counter = {}, {}
    for season, week in periods:
        mask = (frame.season == season) & (frame.week == week)
        history = frame[(frame.season < season) | ((frame.season == season) & (frame.week < week))]
        if history.empty:
            continue
        for position in POSITIONS:
            test = frame[mask & (frame.position == position)]
            if test.empty:
                continue
            train = history[history.position == position]
            if train.empty:
                continue
            counter[position] = counter.get(position, 0) + 1
            key = position
            if key not in cache or counter[position] % retrain_weeks == 1 or retrain_weeks == 1:
                cache[key] = (train, week, season)
            fit_train = cache[key][0] if retrain_weeks > 1 else train
            predictions.loc[test.index] = fit_predict(fit_train, test, features, position, objective=objective)
    frame['candidate'] = predictions
    return frame[frame.candidate.notna()].copy()


def metrics(frame, column):
    error = frame[column] - frame['actual']
    return {'games': int(len(frame)), 'mae': float(np.mean(np.abs(error))), 'rmse': float(np.sqrt(np.mean(error ** 2))),
            'bias': float(np.mean(error))}


def bootstrap_gain(frame, control='control', candidate='candidate', draws=2000, seed=74051):
    """Paired week-block bootstrap of the MAE improvement (control minus candidate)."""
    frame = frame.copy()
    frame['week_block'] = list(zip(frame.season, frame.week))
    control_error = np.abs(frame[control] - frame['actual'])
    candidate_error = np.abs(frame[candidate] - frame['actual'])
    frame['delta'] = control_error - candidate_error
    blocks = [g['delta'].to_numpy() for _, g in frame.groupby('week_block')]
    rng = np.random.default_rng(seed)
    samples = np.array([float(np.mean(np.concatenate([blocks[i] for i in rng.integers(0, len(blocks), len(blocks))]))) for _ in range(draws)])
    lower, upper = np.quantile(samples, [.025, .975])
    return {'gain': float(np.mean(frame.delta)), 'ci95': [float(lower), float(upper)],
            'weeks': len(blocks), 'significant': bool(lower > 0)}
