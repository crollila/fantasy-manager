"""Pregame-safe player-week features for the fitted projection model.

Every feature here is built from games that finished **strictly before** the target game.
The rule is enforced in one place: :func:`lagged_frames` shifts each player's and each
team's history by one game-week before any aggregation, so a target week's own box score
can never reach its own features. Market lines come from the schedule, which is settled
before kickoff by definition.

Nothing in this module reads a result for the game being predicted.
"""
from __future__ import annotations
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from app.storage import DATA

log = logging.getLogger(__name__)

POSITIONS = ('QB', 'RB', 'WR', 'TE')
# Horizons in games, plus an exponentially weighted value and season to date.
HORIZONS = (1, 3, 5)
DECAY = .7

USAGE_COLUMNS = ('targets', 'carries', 'attempts', 'receptions', 'receiving_yards', 'rushing_yards',
                 'passing_yards', 'receiving_tds', 'rushing_tds', 'passing_tds', 'target_share',
                 'air_yards_share', 'receiving_air_yards', 'receiving_yards_after_catch')


def add_upcoming(frame, keys, season, week, values, extra=None):
    """Append empty rows for a week that has not been played, so it gets lagged features.

    Every feature here comes from ``shift(1)`` over prior rows, so a target week needs a row
    to exist before it can receive one. A completed week gets that row from the box score; an
    upcoming week has no box score yet, so one is appended with no statistics in it. The
    appended row contributes nothing to anyone's history precisely because it is empty, and
    it cannot see itself because the shift still excludes it.
    """
    if frame is None or frame.empty or not values:
        return frame
    present = frame[(frame.season == season) & (frame.week == week)]
    have = set(map(tuple, present[keys].to_numpy())) if not present.empty else set()
    rows = []
    for value in values:
        value = value if isinstance(value, tuple) else (value,)
        if value in have:
            continue
        row = {k: v for k, v in zip(keys, value)}
        row.update(season=season, week=week)
        if extra:
            row.update(extra)
        rows.append(row)
    if not rows:
        return frame
    return pd.concat([frame, pd.DataFrame(rows)], ignore_index=True)


def read(path):
    path = Path(path)
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


# --------------------------------------------------------------------------- market

def market_features(schedule):
    """Team implied points, spread and total from the settled pregame line (one row per team-game)."""
    games = schedule[(schedule.game_type == 'REG')].copy()
    rows = []
    for side, other in (('home', 'away'), ('away', 'home')):
        part = games[['game_id', 'season', 'week', f'{side}_team', f'{other}_team', 'spread_line', 'total_line', 'roof', 'surface', 'location']].copy()
        part.columns = ['game_id', 'season', 'week', 'team', 'opponent', 'spread_line', 'total_line', 'roof', 'surface', 'location']
        # nflverse spread_line is the home margin; a positive value favours the home team.
        part['team_spread'] = part.spread_line * (1 if side == 'home' else -1)
        part['is_home'] = 1. if side == 'home' else 0.
        rows.append(part)
    out = pd.concat(rows, ignore_index=True)
    out['game_total'] = out.total_line
    out['team_implied_points'] = out.total_line / 2 + out.team_spread / 2
    out['opponent_implied_points'] = out.total_line / 2 - out.team_spread / 2
    out['is_favorite'] = (out.team_spread > 0).astype(float)
    out['abs_spread'] = out.team_spread.abs()
    out['is_dome'] = out.roof.astype(str).str.lower().isin(['dome', 'closed']).astype(float)
    out['is_turf'] = (~out.surface.astype(str).str.lower().isin(['grass', ''])).astype(float)
    out['is_neutral'] = (out.location.astype(str).str.lower() == 'neutral').astype(float)
    keep = ['game_id', 'season', 'week', 'team', 'opponent', 'is_home', 'team_spread', 'abs_spread',
            'game_total', 'team_implied_points', 'opponent_implied_points', 'is_favorite',
            'is_dome', 'is_turf', 'is_neutral']
    return out[keep]


# --------------------------------------------------------------------------- lagging

def expanding_lagged(frame, keys, order, columns, prefix):
    """Per-key lagged aggregates: last game, last 3/5, season to date, exponentially weighted.

    The frame is shifted by one row within each key *before* any window is taken, which is
    the single guarantee that a row never sees its own outcome.
    """
    frame = frame.sort_values(list(keys) + list(order)).copy()
    grouped = frame.groupby(keys, sort=False)
    out = {}
    for col in columns:
        if col not in frame:
            continue
        shifted = grouped[col].shift(1)
        by = shifted.groupby([frame[k] for k in keys], sort=False)
        out[f'{prefix}{col}_last1'] = shifted
        for h in HORIZONS[1:]:
            out[f'{prefix}{col}_last{h}'] = by.transform(lambda s, h=h: s.rolling(h, min_periods=1).mean())
        out[f'{prefix}{col}_std'] = by.transform(lambda s: s.expanding().mean())
        out[f'{prefix}{col}_ewm'] = by.transform(lambda s: s.ewm(alpha=1 - DECAY, adjust=False).mean())
    result = pd.DataFrame(out, index=frame.index)
    for col in list(keys) + list(order):
        result[col] = frame[col]
    return result


# --------------------------------------------------------------------------- player usage

def player_usage(stats):
    """Lagged per-game usage and share features for every player-week."""
    frame = stats[stats.season_type == 'REG'].copy()
    frame = frame[frame.position.isin(POSITIONS)]
    for col in USAGE_COLUMNS:
        if col not in frame:
            frame[col] = np.nan
    frame['adot'] = frame.receiving_air_yards / frame.targets.replace(0, np.nan)
    frame['yac_per_reception'] = frame.receiving_yards_after_catch / frame.receptions.replace(0, np.nan)
    columns = list(USAGE_COLUMNS) + ['adot', 'yac_per_reception']
    lagged = expanding_lagged(frame, ['player_id'], ('season', 'week'), columns, 'use_')
    lagged['position'] = frame.position.values
    lagged['team'] = frame.team.values
    # Role trend: how far the recent window sits above or below the season to date.
    for col in ('targets', 'carries', 'target_share'):
        recent, season = f'use_{col}_last3', f'use_{col}_std'
        if recent in lagged and season in lagged:
            lagged[f'trend_{col}'] = lagged[recent] - lagged[season]
    return lagged


# --------------------------------------------------------------------------- snaps

def snap_features(snaps, stats):
    """Lagged offensive snap share and its recent change, joined on the nflverse player id."""
    if snaps.empty:
        return pd.DataFrame()
    frame = snaps[snaps.game_type == 'REG'].copy() if 'game_type' in snaps else snaps.copy()
    frame = frame[['season', 'week', 'pfr_player_id', 'team', 'offense_pct', 'offense_snaps']].dropna(subset=['pfr_player_id'])
    lagged = expanding_lagged(frame, ['pfr_player_id'], ('season', 'week'), ['offense_pct', 'offense_snaps'], 'snap_')
    lagged['snap_share_change'] = lagged['snap_offense_pct_last1'] - lagged['snap_offense_pct_std']
    lagged['snap_share_jump'] = lagged['snap_offense_pct_last1'] - lagged['snap_offense_pct_last5']
    return lagged


# --------------------------------------------------------------------------- red zone

def red_zone_frame(pbp):
    """Per player-week red-zone and goal-line opportunity counts, taken from play-by-play."""
    if pbp.empty:
        return pd.DataFrame()
    cols = ['season', 'week', 'game_id', 'posteam', 'yardline_100', 'receiver_player_id', 'rusher_player_id', 'pass', 'rush', 'season_type']
    frame = pbp[[c for c in cols if c in pbp.columns]].copy()
    if 'season_type' in frame:
        frame = frame[frame.season_type == 'REG']
    frame = frame[frame.yardline_100.notna()]
    rows = []
    for field, role in (('receiver_player_id', 'rz_targets'), ('rusher_player_id', 'rz_carries')):
        if field not in frame:
            continue
        part = frame[frame[field].notna()].copy()
        part['player_id'] = part[field]
        part['inside20'] = (part.yardline_100 <= 20).astype(float)
        part['inside10'] = (part.yardline_100 <= 10).astype(float)
        part['inside5'] = (part.yardline_100 <= 5).astype(float)
        agg = part.groupby(['season', 'week', 'player_id', 'posteam'], as_index=False)[['inside20', 'inside10', 'inside5']].sum()
        agg.columns = ['season', 'week', 'player_id', 'team', f'{role}_20', f'{role}_10', f'{role}_5']
        rows.append(agg)
    if not rows:
        return pd.DataFrame()
    out = rows[0]
    for extra in rows[1:]:
        out = out.merge(extra, on=['season', 'week', 'player_id', 'team'], how='outer')
    out = out.fillna(0.)
    # Team totals so a player's share of the red-zone work can be expressed.
    team = out.groupby(['season', 'week', 'team'], as_index=False)[[c for c in out.columns if c.startswith(('rz_targets', 'rz_carries'))]].sum()
    team.columns = ['season', 'week', 'team'] + [f'team_{c}' for c in team.columns[3:]]
    out = out.merge(team, on=['season', 'week', 'team'], how='left')
    for c in [c for c in out.columns if c.startswith(('rz_targets', 'rz_carries'))]:
        out[f'share_{c}'] = out[c] / out[f'team_{c}'].replace(0, np.nan)
    return out


def red_zone_features(pbp, upcoming=None):
    frame = red_zone_frame(pbp)
    if frame.empty:
        return pd.DataFrame()
    if upcoming:
        people = upcoming['players']
        frame = add_upcoming(frame, ['player_id', 'team'], upcoming['season'], upcoming['week'],
                             list(map(tuple, people[['player_id', 'team']].to_numpy())))
    columns = [c for c in frame.columns if c.startswith(('rz_', 'share_rz_', 'team_rz_'))]
    return expanding_lagged(frame, ['player_id'], ('season', 'week'), columns, 'rz_')


# --------------------------------------------------------------------------- team pace / pass rate

def team_pass_features(pbp, upcoming=None):
    """Lagged team plays, pass rate and neutral pass rate — the pass-rate forecast as an input."""
    if pbp.empty:
        return pd.DataFrame()
    cols = ['season', 'week', 'game_id', 'posteam', 'pass', 'rush', 'score_differential', 'game_seconds_remaining', 'season_type']
    frame = pbp[[c for c in cols if c in pbp.columns]].copy()
    if 'season_type' in frame:
        frame = frame[frame.season_type == 'REG']
    frame = frame[(frame.get('pass', 0) == 1) | (frame.get('rush', 0) == 1)]
    frame = frame[frame.posteam.notna()]
    frame['neutral'] = (frame.score_differential.abs() <= 7) & (frame.game_seconds_remaining > 120)
    agg = frame.groupby(['season', 'week', 'posteam'], as_index=False).agg(
        team_plays=('pass', 'size'), team_pass_rate=('pass', 'mean'))
    neutral = frame[frame.neutral].groupby(['season', 'week', 'posteam'], as_index=False).agg(team_neutral_pass_rate=('pass', 'mean'))
    agg = agg.merge(neutral, on=['season', 'week', 'posteam'], how='left')
    agg = agg.rename(columns={'posteam': 'team'})
    if upcoming:
        agg = add_upcoming(agg, ['team'], upcoming['season'], upcoming['week'], sorted(set(upcoming['players'].team)))
    return expanding_lagged(agg, ['team'], ('season', 'week'), ['team_plays', 'team_pass_rate', 'team_neutral_pass_rate'], 'off_')


# --------------------------------------------------------------------------- opponent defense

def defense_features(pbp, upcoming=None):
    """Lagged opponent pass/rush defensive splits. Only games already played contribute."""
    if pbp.empty:
        return pd.DataFrame()
    cols = ['season', 'week', 'defteam', 'pass', 'rush', 'epa', 'success', 'yards_gained', 'touchdown', 'sack', 'season_type']
    frame = pbp[[c for c in cols if c in pbp.columns]].copy()
    if 'season_type' in frame:
        frame = frame[frame.season_type == 'REG']
    frame = frame[((frame.get('pass', 0) == 1) | (frame.get('rush', 0) == 1)) & frame.defteam.notna()]
    frame['is_pass'] = frame['pass'] == 1
    frame['explosive'] = frame.yards_gained >= 20
    out = []
    for label, mask in (('pass', frame.is_pass), ('rush', ~frame.is_pass)):
        part = frame[mask]
        agg = part.groupby(['season', 'week', 'defteam'], as_index=False).agg(**{
            f'def_{label}_epa': ('epa', 'mean'),
            f'def_{label}_success': ('success', 'mean'),
            f'def_{label}_explosive': ('explosive', 'mean'),
            f'def_{label}_td_rate': ('touchdown', 'mean'),
            f'def_{label}_plays': ('epa', 'size'),
        })
        out.append(agg)
    merged = out[0]
    for extra in out[1:]:
        merged = merged.merge(extra, on=['season', 'week', 'defteam'], how='outer')
    if 'sack' in frame:
        sacks = frame[frame.is_pass].groupby(['season', 'week', 'defteam'], as_index=False).agg(def_sack_rate=('sack', 'mean'))
        merged = merged.merge(sacks, on=['season', 'week', 'defteam'], how='left')
    merged = merged.rename(columns={'defteam': 'opponent'})
    if upcoming:
        merged = add_upcoming(merged, ['opponent'], upcoming['season'], upcoming['week'], sorted(set(upcoming['players'].opponent)))
    columns = [c for c in merged.columns if c.startswith('def_')]
    return expanding_lagged(merged, ['opponent'], ('season', 'week'), columns, 'opp_')


# --------------------------------------------------------------------------- assembly

def load_sources(seasons, cache_path=None):
    cache_path = Path(cache_path or DATA/'cache')
    stats = pd.concat([read(cache_path/f'stats_{y}.parquet') for y in seasons], ignore_index=True)
    snaps = pd.concat([read(cache_path/f'snaps_{y}.parquet') for y in seasons], ignore_index=True)
    pbp = pd.concat([read(cache_path/f'pbp_{y}.parquet') for y in seasons], ignore_index=True)
    schedule = read(cache_path/'schedules.parquet')
    return stats, snaps, pbp, schedule


def build(seasons, cache_path=None, upcoming=None):
    """One pregame-safe row per player-week for the seasons requested.

    ``upcoming`` optionally names a week that has not been played
    ``{'season', 'week', 'players': DataFrame[player_id, position, team, opponent]}``; those
    player-weeks are given rows so they receive lagged features like any other week. Without
    it the model can only score games that already happened.
    """
    stats, snaps, pbp, schedule = load_sources(seasons, cache_path)
    if stats.empty:
        return pd.DataFrame()
    if upcoming:
        season, week = upcoming['season'], upcoming['week']
        people = upcoming['players']
        stats = add_upcoming(stats, ['player_id', 'position', 'team', 'opponent_team'], season, week,
                             list(map(tuple, people[['player_id', 'position', 'team', 'opponent']].to_numpy())),
                             extra={'season_type': 'REG'})
        if not snaps.empty:
            ids = read(Path(cache_path or DATA/'cache')/'players.parquet')
            if not ids.empty and {'gsis_id', 'pfr_id'} <= set(ids.columns):
                mapping = ids[['gsis_id', 'pfr_id']].dropna().drop_duplicates('gsis_id').set_index('gsis_id').pfr_id
                pfr = [mapping.get(p) for p in people.player_id if mapping.get(p) is not None]
                snaps = add_upcoming(snaps, ['pfr_player_id'], season, week, pfr, extra={'game_type': 'REG'})
    base = stats[(stats.season_type == 'REG') & (stats.position.isin(POSITIONS))].copy()
    frame = base[['season', 'week', 'player_id', 'player_display_name', 'position', 'team', 'opponent_team']].copy()
    frame = frame.rename(columns={'opponent_team': 'opponent'})

    usage = player_usage(stats)
    frame = frame.merge(usage.drop(columns=['position', 'team'], errors='ignore'), on=['player_id', 'season', 'week'], how='left')

    rz = red_zone_features(pbp, upcoming)
    if not rz.empty:
        frame = frame.merge(rz, on=['player_id', 'season', 'week'], how='left')

    ids = read(Path(cache_path or DATA/'cache')/'players.parquet')
    if not ids.empty and 'gsis_id' in ids and 'pfr_id' in ids:
        mapping = ids[['gsis_id', 'pfr_id']].dropna().drop_duplicates('gsis_id')
        frame = frame.merge(mapping.rename(columns={'gsis_id': 'player_id'}), on='player_id', how='left')
        snapf = snap_features(snaps, stats)
        if not snapf.empty:
            frame = frame.merge(snapf.rename(columns={'pfr_player_id': 'pfr_id'}), on=['pfr_id', 'season', 'week'], how='left')

    off = team_pass_features(pbp, upcoming)
    if not off.empty:
        frame = frame.merge(off, on=['team', 'season', 'week'], how='left')

    dfn = defense_features(pbp, upcoming)
    if not dfn.empty:
        frame = frame.merge(dfn, on=['opponent', 'season', 'week'], how='left')

    market = market_features(schedule)
    frame = frame.merge(market, on=['team', 'opponent', 'season', 'week'], how='left')
    return frame


FAMILIES = {
    'market': ('team_implied_points', 'opponent_implied_points', 'game_total', 'team_spread', 'abs_spread',
               'is_favorite', 'is_home', 'is_dome', 'is_turf', 'is_neutral'),
    'usage': tuple(),      # filled at fit time from the built frame
    'snaps': tuple(),
    'redzone': tuple(),
    'passrate': tuple(),
    'defense': tuple(),
}


def family_columns(frame):
    """Group the built columns into the ablation families used by the backtest."""
    cols = list(frame.columns)
    fam = {
        'market': [c for c in FAMILIES['market'] if c in cols],
        'usage': [c for c in cols if c.startswith('use_') or c.startswith('trend_')],
        'snaps': [c for c in cols if c.startswith('snap_')],
        'redzone': [c for c in cols if c.startswith('rz_')],
        'passrate': [c for c in cols if c.startswith('off_')],
        'defense': [c for c in cols if c.startswith('opp_')],
    }
    return {k: v for k, v in fam.items() if v}
