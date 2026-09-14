"""Team-level features derived from player production expectations and role changes.

Deliberately **not** a sum of fantasy points. Each column is an expectation of an underlying
football quantity (yards, touchdowns, targets, carries) aggregated over the players a team
is expected to field, plus measures of how much those roles have recently moved.

Two exclusions are load-bearing:

* **No market information.** Spread, total and implied points are supplied to the game model
  through its own ``market`` family. Routing them in again through the player channel would
  measure Vegas against itself rather than the value of player information.
* **No duplication of the engine's QB or availability families.** ``qb_*`` (95 features) and
  ``inj_lost_* / value_lost_* / starters_out_* / reserve_lost_*`` (54 features) already exist
  and are stronger than a fantasy projection; this module adds only what is missing.

Everything is lagged: a team's week-N row is built from player games completed before week N.
"""
from __future__ import annotations
import logging
import numpy as np
import pandas as pd
from app.player_features import player_usage, snap_features, read
from app.storage import DATA

log = logging.getLogger(__name__)

SKILL = ('RB', 'WR', 'TE')
# Underlying per-game expectations aggregated to the team, taken from the exponentially
# weighted lagged column so recent usage dominates without any window seeing its own week.
EXPECTATIONS = {
    'pass_yards': 'use_passing_yards_ewm',
    'rush_yards': 'use_rushing_yards_ewm',
    'rec_yards': 'use_receiving_yards_ewm',
    'pass_td': 'use_passing_tds_ewm',
    'rush_td': 'use_rushing_tds_ewm',
    'rec_td': 'use_receiving_tds_ewm',
    'targets': 'use_targets_ewm',
    'carries': 'use_carries_ewm',
    'attempts': 'use_attempts_ewm',
}


def player_frame(seasons, cache_path=None):
    """Lagged per-player expectations joined to snap share, one row per player-week."""
    cache_path = cache_path or DATA/'cache'
    stats = pd.concat([read(cache_path/f'stats_{y}.parquet') for y in seasons], ignore_index=True)
    if stats.empty:
        return pd.DataFrame()
    usage = player_usage(stats)
    base = stats[(stats.season_type == 'REG') & (stats.position.isin(('QB',) + SKILL))]
    frame = base[['season', 'week', 'player_id', 'position', 'team']].copy()
    frame = frame.merge(usage.drop(columns=['position', 'team'], errors='ignore'),
                        on=['player_id', 'season', 'week'], how='left')
    snaps = pd.concat([read(cache_path/f'snaps_{y}.parquet') for y in seasons], ignore_index=True)
    ids = read(cache_path/'players.parquet')
    if not snaps.empty and not ids.empty and {'gsis_id', 'pfr_id'} <= set(ids.columns):
        mapping = ids[['gsis_id', 'pfr_id']].dropna().drop_duplicates('gsis_id').rename(columns={'gsis_id': 'player_id'})
        frame = frame.merge(mapping, on='player_id', how='left')
        snapf = snap_features(snaps, None).rename(columns={'pfr_player_id': 'pfr_id'})
        frame = frame.merge(snapf, on=['pfr_id', 'season', 'week'], how='left')
    return frame


def team_features(seasons, cache_path=None):
    """Aggregate the lagged player expectations to one row per team-week."""
    frame = player_frame(seasons, cache_path)
    if frame.empty:
        return pd.DataFrame()
    for name, column in EXPECTATIONS.items():
        frame[f'x_{name}'] = frame[column].fillna(0.) if column in frame else 0.
    rows = []
    for (season, week, team), part in frame.groupby(['season', 'week', 'team'], sort=False):
        row = {'season': season, 'week': week, 'team': team}
        skill = part[part.position.isin(SKILL)]
        # Projected team production by phase, from the players expected to be on the field.
        row['pl_proj_pass_yards'] = float(part.x_pass_yards.sum())
        row['pl_proj_rush_yards'] = float(part.x_rush_yards.sum())
        row['pl_proj_rec_yards'] = float(skill.x_rec_yards.sum())
        row['pl_proj_pass_td'] = float(part.x_pass_td.sum())
        row['pl_proj_rush_td'] = float(part.x_rush_td.sum())
        row['pl_proj_rec_td'] = float(skill.x_rec_td.sum())
        row['pl_proj_targets'] = float(skill.x_targets.sum())
        row['pl_proj_carries'] = float(part.x_carries.sum())
        row['pl_proj_attempts'] = float(part.x_attempts.sum())
        # Aggregate strength by position group.
        for position in SKILL:
            group = part[part.position == position]
            row[f'pl_{position.lower()}_rec_yards'] = float(group.x_rec_yards.sum())
            row[f'pl_{position.lower()}_rush_yards'] = float(group.x_rush_yards.sum())
            row[f'pl_{position.lower()}_targets'] = float(group.x_targets.sum())
            row[f'pl_{position.lower()}_count'] = float(len(group))
        # Concentration: the single most productive skill player and the top three.
        produced = (skill.x_rec_yards + skill.x_rush_yards).sort_values(ascending=False)
        row['pl_top_skill_yards'] = float(produced.iloc[0]) if len(produced) else 0.
        row['pl_top3_skill_yards'] = float(produced.head(3).sum()) if len(produced) else 0.
        total_skill = float(produced.sum())
        row['pl_top_skill_share'] = float(produced.iloc[0] / total_skill) if total_skill > 0 and len(produced) else 0.
        # Dispersion across contributors, as a crude projection-uncertainty proxy.
        row['pl_skill_yards_sd'] = float(produced.std()) if len(produced) > 1 else 0.
        row['pl_contributors'] = float((produced > 20).sum())
        # Role movement: how much the team's usage pattern has recently shifted.
        for column, label in (('snap_share_change', 'snap_change'), ('snap_share_jump', 'snap_jump'),
                              ('trend_targets', 'target_trend'), ('trend_carries', 'carry_trend')):
            if column in part:
                values = part[column].dropna()
                row[f'pl_{label}_mean'] = float(values.mean()) if len(values) else 0.
                row[f'pl_{label}_absmax'] = float(values.abs().max()) if len(values) else 0.
            else:
                row[f'pl_{label}_mean'] = 0.
                row[f'pl_{label}_absmax'] = 0.
        if 'snap_share_change' in part:
            moved = part.snap_share_change.dropna().abs()
            row['pl_roles_shifted'] = float((moved > .2).sum())
        else:
            row['pl_roles_shifted'] = 0.
        rows.append(row)
    return pd.DataFrame(rows)


FEATURE_PREFIX = 'pl_'


def game_features(seasons, schedule, cache_path=None):
    """Home/away/diff player-derived columns, one row per game, ready to join to the engine frame."""
    team = team_features(seasons, cache_path)
    if team.empty:
        return pd.DataFrame()
    games = schedule[(schedule.game_type == 'REG')][['game_id', 'season', 'week', 'home_team', 'away_team']].copy()
    columns = [c for c in team.columns if c.startswith(FEATURE_PREFIX)]
    out = games.copy()
    for side in ('home', 'away'):
        merged = team.rename(columns={'team': f'{side}_team', **{c: f'{side}_{c}' for c in columns}})
        out = out.merge(merged, on=[f'{side}_team', 'season', 'week'], how='left')
    for column in columns:
        out[f'diff_{column}'] = out[f'home_{column}'] - out[f'away_{column}']
    keep = ['game_id', 'season', 'week'] + [f'{s}_{c}' for s in ('home', 'away') for c in columns] + [f'diff_{c}' for c in columns]
    return out[keep]


def assert_no_market_columns(frame):
    """Guard the exclusion that makes this test meaningful."""
    banned = [c for c in frame.columns
              if any(k in c.lower() for k in ('spread', 'total_line', 'implied', 'moneyline', 'odds', 'mkt_', 'vegas'))]
    if banned:
        raise ValueError(f'market information must not reach the player feature vector: {banned}')
    return True
