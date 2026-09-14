"""No-leakage guarantees for the pregame-safe player feature set.

These are the tests that matter most: every one of them fails if a feature for week N can
see anything that happened in week N or later. They are built on synthetic histories with
deliberately extreme week-N values, so any leak changes a number by a wide margin.
"""
import numpy as np
import pandas as pd
import pytest
from app.player_features import (expanding_lagged, player_usage, snap_features, red_zone_frame,
                                 red_zone_features, defense_features, team_pass_features, market_features)


def weekly_stats(values, player='p1', position='WR', team='SEA', opponent='SF', season=2025):
    """One row per week; `values` gives that week's targets."""
    return pd.DataFrame([{
        'season': season, 'week': w + 1, 'player_id': player, 'player_display_name': 'P',
        'position': position, 'team': team, 'opponent_team': opponent, 'season_type': 'REG',
        'targets': v, 'carries': 0., 'attempts': 0., 'receptions': v, 'receiving_yards': v * 10,
        'rushing_yards': 0., 'passing_yards': 0., 'receiving_tds': 0., 'rushing_tds': 0.,
        'passing_tds': 0., 'target_share': v / 30, 'air_yards_share': v / 30,
        'receiving_air_yards': v * 8, 'receiving_yards_after_catch': v * 3,
    } for w, v in enumerate(values)])


# --- the core rule --------------------------------------------------------------------

def test_a_weeks_own_value_never_reaches_its_own_features():
    frame = weekly_stats([1., 2., 3., 999.])
    lagged = player_usage(frame).sort_values('week')
    final = lagged.iloc[-1]
    assert final.use_targets_last1 == 3., 'last1 must be the previous week, not this one'
    for column in [c for c in lagged.columns if c.startswith('use_targets_')]:
        assert final[column] < 999., f'{column} leaked the target week'


def test_the_first_week_of_a_history_has_no_usage_features():
    lagged = player_usage(weekly_stats([42.])).sort_values('week')
    assert pd.isna(lagged.iloc[0].use_targets_last1), 'a first game cannot have a lagged value'


def test_changing_only_the_target_week_changes_no_feature():
    """The decisive test: rewrite week N's box score and every week-N feature must be identical."""
    base = player_usage(weekly_stats([4., 6., 8., 10.])).sort_values(['season', 'week']).reset_index(drop=True)
    altered = player_usage(weekly_stats([4., 6., 8., 500.])).sort_values(['season', 'week']).reset_index(drop=True)
    columns = [c for c in base.columns if c.startswith(('use_', 'trend_'))]
    pd.testing.assert_frame_equal(base[columns], altered[columns], check_dtype=False)


def test_later_weeks_cannot_flow_backwards():
    short = player_usage(weekly_stats([4., 6., 8.])).sort_values('week').reset_index(drop=True)
    longer = player_usage(weekly_stats([4., 6., 8., 900., 900.])).sort_values('week').reset_index(drop=True)
    columns = [c for c in short.columns if c.startswith('use_')]
    pd.testing.assert_frame_equal(short[columns], longer[columns].iloc[:len(short)], check_dtype=False)


def test_one_players_history_does_not_contaminate_another():
    a = weekly_stats([5., 5., 5.], player='a')
    b = weekly_stats([100., 100., 100.], player='b')
    lagged = player_usage(pd.concat([a, b], ignore_index=True))
    last_a = lagged[lagged.player_id == 'a'].sort_values('week').iloc[-1]
    assert last_a.use_targets_last1 == 5.


# --- snaps ----------------------------------------------------------------------------

def snap_rows(pcts, player='pfr1', season=2025):
    return pd.DataFrame([{'season': season, 'week': w + 1, 'pfr_player_id': player, 'team': 'SEA',
                          'game_type': 'REG', 'offense_pct': p, 'offense_snaps': p * 60}
                         for w, p in enumerate(pcts)])


def test_snap_share_is_lagged_and_detects_a_role_jump():
    frame = snap_features(snap_rows([.35, .35, .35, .80, .80]), None).sort_values('week')
    final = frame.iloc[-1]
    assert final.snap_offense_pct_last1 == pytest.approx(.80), 'last week, not this week'
    assert final.snap_share_change > .1, 'a 35% to 80% role jump must be visible'
    first_jump = frame.iloc[3]
    assert first_jump.snap_offense_pct_last1 == pytest.approx(.35), 'the jump week itself is not yet known'


def test_snap_features_never_read_the_target_week():
    base = snap_features(snap_rows([.4, .5, .6]), None).sort_values('week').reset_index(drop=True)
    altered = snap_features(snap_rows([.4, .5, 1.0]), None).sort_values('week').reset_index(drop=True)
    columns = [c for c in base.columns if c.startswith('snap_')]
    pd.testing.assert_frame_equal(base[columns], altered[columns], check_dtype=False)


# --- red zone -------------------------------------------------------------------------

def pbp_rows(per_week, team='SEA', defteam='SF', season=2025):
    rows = []
    for week, count in enumerate(per_week, start=1):
        for i in range(count):
            rows.append({'season': season, 'week': week, 'game_id': f'g{week}', 'posteam': team,
                         'defteam': defteam, 'yardline_100': 5., 'receiver_player_id': 'p1',
                         'rusher_player_id': None, 'pass': 1, 'rush': 0, 'epa': .1, 'success': 1.,
                         'yards_gained': 5., 'touchdown': 0., 'sack': 0., 'season_type': 'REG',
                         'score_differential': 0., 'game_seconds_remaining': 1800.})
    return pd.DataFrame(rows)


def test_red_zone_counts_are_lagged():
    frame = red_zone_features(pbp_rows([1, 2, 3, 40])).sort_values('week')
    final = frame.iloc[-1]
    assert final['rz_rz_targets_20_last1'] == 3., 'the 40-target week must not see itself'


def test_red_zone_features_ignore_the_target_week():
    base = red_zone_features(pbp_rows([2, 2, 2])).sort_values(['season', 'week']).reset_index(drop=True)
    altered = red_zone_features(pbp_rows([2, 2, 99])).sort_values(['season', 'week']).reset_index(drop=True)
    columns = [c for c in base.columns if c.startswith('rz_')]
    pd.testing.assert_frame_equal(base[columns], altered[columns], check_dtype=False)


# --- opponent defence -----------------------------------------------------------------

def test_opponent_defence_excludes_the_game_being_predicted():
    """Week N's defensive numbers must come only from weeks the defence has already played."""
    good = pbp_rows([4, 4, 4])
    good.loc[good.week == 3, 'epa'] = -9.   # a dominant week 3 for the defence
    frame = defense_features(good).sort_values('week')
    final = frame.iloc[-1]
    assert final['opp_def_pass_epa_last1'] == pytest.approx(.1), 'week 3 must not grade itself'
    assert final['opp_def_pass_epa_std'] == pytest.approx(.1)


def test_defence_features_are_unchanged_when_the_target_week_changes():
    base = defense_features(pbp_rows([3, 3, 3])).sort_values(['season', 'week']).reset_index(drop=True)
    altered = pbp_rows([3, 3, 3])
    altered.loc[altered.week == 3, ['epa', 'yards_gained', 'touchdown']] = [-5., 80., 1.]
    altered = defense_features(altered).sort_values(['season', 'week']).reset_index(drop=True)
    columns = [c for c in base.columns if c.startswith('opp_')]
    pd.testing.assert_frame_equal(base[columns], altered[columns], check_dtype=False)


def test_team_pass_rate_is_lagged():
    frame = team_pass_features(pbp_rows([5, 5, 5])).sort_values('week')
    assert pd.isna(frame.iloc[0]['off_team_plays_last1']), 'week 1 has no prior team history'
    assert frame.iloc[-1]['off_team_plays_last1'] == 5.


# --- market ---------------------------------------------------------------------------

def test_market_features_use_the_pregame_line_and_split_implied_points():
    schedule = pd.DataFrame([{'game_id': 'g1', 'season': 2025, 'week': 1, 'game_type': 'REG',
                              'home_team': 'SEA', 'away_team': 'SF', 'spread_line': 3., 'total_line': 44.,
                              'roof': 'outdoors', 'surface': 'grass', 'location': 'Home'}])
    out = market_features(schedule).set_index('team')
    assert out.loc['SEA'].team_implied_points == pytest.approx(23.5)
    assert out.loc['SF'].team_implied_points == pytest.approx(20.5)
    assert out.loc['SEA'].is_favorite == 1. and out.loc['SF'].is_favorite == 0.
    assert out.loc['SEA'].is_home == 1. and out.loc['SF'].is_home == 0.
    # The two sides must agree on the shared game facts.
    assert out.loc['SEA'].game_total == out.loc['SF'].game_total == 44.
    assert out.loc['SEA'].opponent_implied_points == pytest.approx(out.loc['SF'].team_implied_points)


def test_no_score_or_result_column_reaches_the_market_features():
    schedule = pd.DataFrame([{'game_id': 'g1', 'season': 2025, 'week': 1, 'game_type': 'REG',
                              'home_team': 'SEA', 'away_team': 'SF', 'spread_line': 3., 'total_line': 44.,
                              'roof': 'outdoors', 'surface': 'grass', 'location': 'Home',
                              'home_score': 31., 'away_score': 10.}])
    out = market_features(schedule)
    banned = [c for c in out.columns if 'score' in c.lower() or 'result' in c.lower()]
    assert banned == [], f'post-kickoff columns leaked into the feature set: {banned}'


# --- the generic guarantee ------------------------------------------------------------

def test_expanding_lagged_shifts_before_every_window():
    frame = pd.DataFrame({'k': ['a'] * 5, 'season': [2025] * 5, 'week': [1, 2, 3, 4, 5], 'v': [1., 2., 3., 4., 1000.]})
    out = expanding_lagged(frame, ['k'], ('season', 'week'), ['v'], 'x_').sort_values('week')
    assert pd.isna(out.iloc[0].x_v_last1)
    assert list(out.x_v_last1)[1:] == [1., 2., 3., 4.]
    assert out.iloc[-1].x_v_last1 == 4.
    assert out.iloc[-1].x_v_last3 == pytest.approx(3.)   # mean of 2,3,4
    assert out.x_v_std.max() < 1000.


# --- serving an upcoming week ----------------------------------------------------------

def test_an_upcoming_week_gets_lagged_features_even_with_no_box_score():
    """The bug this guards: features were built from the stats table, so a game that had not
    been played had no row at all and the model silently did nothing."""
    from app.player_features import add_upcoming
    played = weekly_stats([5., 7., 9.])
    extended = add_upcoming(played, ['player_id', 'position', 'team', 'opponent_team'], 2025, 4,
                            [('p1', 'WR', 'SEA', 'SF')], extra={'season_type': 'REG'})
    assert len(extended) == len(played) + 1
    lagged = player_usage(extended).sort_values('week')
    future = lagged.iloc[-1]
    assert future.week == 4
    assert future.use_targets_last1 == 9., 'the upcoming week sees the last completed week'
    assert future.use_targets_last3 == pytest.approx(7.), 'mean of 5, 7, 9'


def test_the_appended_upcoming_row_contributes_nothing_to_history():
    from app.player_features import add_upcoming
    played = weekly_stats([5., 7., 9.])
    base = player_usage(played).sort_values('week').reset_index(drop=True)
    extended = player_usage(add_upcoming(played, ['player_id', 'position', 'team', 'opponent_team'], 2025, 4,
                                         [('p1', 'WR', 'SEA', 'SF')], extra={'season_type': 'REG'}))
    extended = extended.sort_values('week').reset_index(drop=True)
    columns = [c for c in base.columns if c.startswith('use_')]
    pd.testing.assert_frame_equal(base[columns], extended[columns].iloc[:len(base)], check_dtype=False)


def test_add_upcoming_does_not_duplicate_a_week_that_exists():
    from app.player_features import add_upcoming
    played = weekly_stats([5., 7., 9.])
    same = add_upcoming(played, ['player_id', 'position', 'team', 'opponent_team'], 2025, 3,
                        [('p1', 'WR', 'SEA', 'SF')], extra={'season_type': 'REG'})
    assert len(same) == len(played), 'week 3 already has a row; it must not be duplicated'


def test_team_and_red_zone_aggregates_extend_to_an_upcoming_week():
    from app.player_features import team_pass_features, red_zone_features
    upcoming = {'season': 2025, 'week': 4,
                'players': pd.DataFrame([{'player_id': 'p1', 'team': 'SEA', 'opponent': 'SF'}])}
    team = team_pass_features(pbp_rows([5, 5, 5]), upcoming).sort_values('week')
    assert int(team.iloc[-1].week) == 4
    assert team.iloc[-1]['off_team_plays_last1'] == 5., 'upcoming week inherits the last played week'
    rz = red_zone_features(pbp_rows([2, 3, 4]), upcoming).sort_values('week')
    assert int(rz.iloc[-1].week) == 4
    assert rz.iloc[-1]['rz_rz_targets_20_last1'] == 4.
