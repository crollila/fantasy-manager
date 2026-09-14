"""Guarantees for the player-derived team features fed to the NFL game model.

Two properties matter and both are tested directly: the vector carries no market
information (otherwise the experiment measures Vegas against itself), and a team's week-N
row cannot see week N.
"""
import numpy as np
import pandas as pd
import pytest
from app.nfl.features.players import (team_features, game_features, assert_no_market_columns,
                                      EXPECTATIONS, FEATURE_PREFIX)


def stats_frame(weeks, team='SEA', season=2025):
    """One QB and one WR per week. Player ids are namespaced by team, because a shared id
    would (correctly) carry one player's history across both clubs."""
    rows = []
    for week, yards in enumerate(weeks, start=1):
        rows.append({'season': season, 'week': week, 'player_id': f'{team}-qb1', 'position': 'QB', 'team': team,
                     'season_type': 'REG', 'passing_yards': 250., 'passing_tds': 2., 'attempts': 33.,
                     'carries': 2., 'rushing_yards': 10., 'rushing_tds': 0., 'targets': 0., 'receptions': 0.,
                     'receiving_yards': 0., 'receiving_tds': 0., 'target_share': 0., 'air_yards_share': 0.,
                     'receiving_air_yards': 0., 'receiving_yards_after_catch': 0., 'opponent_team': 'SF'})
        rows.append({'season': season, 'week': week, 'player_id': f'{team}-wr1', 'position': 'WR', 'team': team,
                     'season_type': 'REG', 'passing_yards': 0., 'passing_tds': 0., 'attempts': 0.,
                     'carries': 0., 'rushing_yards': 0., 'rushing_tds': 0., 'targets': 8., 'receptions': 5.,
                     'receiving_yards': yards, 'receiving_tds': 1., 'target_share': .25, 'air_yards_share': .3,
                     'receiving_air_yards': yards, 'receiving_yards_after_catch': 20., 'opponent_team': 'SF'})
    return pd.DataFrame(rows)


@pytest.fixture
def cache(tmp_path, monkeypatch):
    """A minimal on-disk cache so the feature builder reads only what a test wrote."""
    def write(frame, name):
        frame.to_parquet(tmp_path/name)
    return tmp_path, write


# --- market exclusion -----------------------------------------------------------------

def test_the_market_exclusion_guard_rejects_market_columns():
    frame = pd.DataFrame({'game_id': ['g1'], 'home_pl_proj_pass_yards': [250.], 'mkt_spread': [-3.]})
    with pytest.raises(ValueError, match='market information'):
        assert_no_market_columns(frame)


@pytest.mark.parametrize('column', ['spread_line', 'total_line', 'home_implied_points', 'away_moneyline', 'vegas_total', 'mkt_home_prob'])
def test_every_flavour_of_market_column_is_caught(column):
    frame = pd.DataFrame({'game_id': ['g1'], column: [1.]})
    with pytest.raises(ValueError):
        assert_no_market_columns(frame)


def test_a_clean_player_vector_passes_the_guard():
    frame = pd.DataFrame({'game_id': ['g1'], 'home_pl_proj_rush_yards': [110.], 'diff_pl_top_skill_yards': [12.]})
    assert assert_no_market_columns(frame) is True


def test_the_built_feature_vector_contains_no_market_column(cache):
    path, write = cache
    write(stats_frame([60., 70., 80.]), 'stats_2025.parquet')
    schedule = pd.DataFrame([{'game_id': 'g1', 'season': 2025, 'week': 3, 'game_type': 'REG',
                              'home_team': 'SEA', 'away_team': 'SF'}])
    built = game_features([2025], schedule, cache_path=path)
    assert assert_no_market_columns(built) is True
    assert not built.empty


# --- no leakage -----------------------------------------------------------------------

def test_a_teams_week_features_exclude_that_week(cache):
    path, write = cache
    write(stats_frame([60., 70., 80., 9999.]), 'stats_2025.parquet')
    frame = team_features([2025], cache_path=path).sort_values('week')
    final = frame.iloc[-1]
    assert final.pl_proj_rec_yards < 500., 'the 9999-yard week leaked into its own features'
    # The exponentially weighted value should reflect 60/70/80 only.
    assert 55. < final.pl_proj_rec_yards < 95.


def test_rewriting_the_target_week_changes_no_team_feature(cache):
    path, write = cache
    write(stats_frame([60., 70., 80.]), 'stats_2025.parquet')
    base = team_features([2025], cache_path=path).sort_values('week').reset_index(drop=True)
    write(stats_frame([60., 70., 400.]), 'stats_2025.parquet')
    altered = team_features([2025], cache_path=path).sort_values('week').reset_index(drop=True)
    columns = [c for c in base.columns if c.startswith(FEATURE_PREFIX)]
    pd.testing.assert_frame_equal(base[columns], altered[columns], check_dtype=False)


def test_the_first_week_of_a_season_has_no_prior_production(cache):
    path, write = cache
    write(stats_frame([80.]), 'stats_2025.parquet')
    frame = team_features([2025], cache_path=path)
    assert float(frame.iloc[0].pl_proj_rec_yards) == 0., 'week 1 has no completed games to draw on'


# --- the vector is about football quantities, not fantasy points ------------------------

def test_expectations_are_underlying_football_quantities_not_fantasy_points():
    for name in EXPECTATIONS:
        assert any(k in name for k in ('yards', 'td', 'targets', 'carries', 'attempts')), name
    assert not any('fantasy' in name or 'points' in name for name in EXPECTATIONS)


def test_position_groups_are_reported_separately(cache):
    path, write = cache
    write(stats_frame([60., 70., 80.]), 'stats_2025.parquet')
    frame = team_features([2025], cache_path=path)
    for position in ('rb', 'wr', 'te'):
        assert f'pl_{position}_rec_yards' in frame.columns
        assert f'pl_{position}_targets' in frame.columns
    assert frame.iloc[-1].pl_wr_rec_yards > 0, 'the WR group should carry the receiving production'
    assert frame.iloc[-1].pl_rb_rec_yards == 0, 'no RB played, so the RB group is empty'


def test_home_away_and_difference_columns_are_produced(cache):
    path, write = cache
    home = stats_frame([60., 70., 80.], team='SEA')
    away = stats_frame([10., 10., 10.], team='SF')
    away['opponent_team'] = 'SEA'
    write(pd.concat([home, away], ignore_index=True), 'stats_2025.parquet')
    schedule = pd.DataFrame([{'game_id': 'g1', 'season': 2025, 'week': 3, 'game_type': 'REG',
                              'home_team': 'SEA', 'away_team': 'SF'}])
    built = game_features([2025], schedule, cache_path=path)
    row = built.iloc[0]
    assert row.home_pl_proj_rec_yards > row.away_pl_proj_rec_yards
    assert row.diff_pl_proj_rec_yards == pytest.approx(row.home_pl_proj_rec_yards - row.away_pl_proj_rec_yards)
