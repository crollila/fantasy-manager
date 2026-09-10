from datetime import datetime, timezone, timedelta
import json
import numpy as np
import pandas as pd
import pytest
from app.storage import Store
from app.game_evidence import observations, fit_expectations, expectations
from app.game_learning import (apply_learning, artifact, state, update_learning,
                               mature_examples, diagnostics, FAMILY, digest)
from app.game_benchmarks import BenchmarkImport, import_benchmarks, for_game
from app.tracking import save_game, saved_games, settle_games, dashboard
from test_intelligence import forecast, final, NOW


def prepared(gid, when, week=1, learning=None, store=None):
    f = forecast(gid, when) | {'week': week, 'components': {'home': {'inputs': {'home': 1.}}, 'away': {'inputs': {'home': 0.}}},
                             'process_expectations': {'home': {'turnovers': {'label': 'Turnovers lost', 'unit': 'turnovers', 'expected': 1., 'scatter': 1., 'low': 0., 'high': 2.3}}}}
    current = learning or {'active': None, 'pending': None}
    return apply_learning(f, current, artifact(store, current['active']) if store and current.get('active') else None,
                          artifact(store, current['pending']) if store and current.get('pending') else None)


def batch(store, start, weeks, first_week=1, learning=None, offset=4):
    created = []
    for week in range(weeks):
        for game in range(16):
            when = start + timedelta(days=week*7, hours=game)
            gid = f'{first_week+week}-{game}'
            f = prepared(gid, when, first_week+week, learning, store)
            assert save_game(store, f, when-timedelta(days=1))
            g = final(gid, 24+offset, 20+offset) | {'week': first_week+week}
            settle_games(store, [g]); created.append(f)
    return created


def test_reviews_compare_saved_expectations_and_preserve_corrections(tmp_path):
    store = Store(tmp_path); f = prepared('g', NOW+timedelta(days=1))
    f['personnel_snapshot'] = {'home': [{'espn_id': '10', 'name': 'Example player', 'likely': False, 'probability': .1}]}
    save_game(store, f, NOW)
    actuals = {('g', 'H'): {'metrics': {'turnovers': 4.}, 'sources': {'turnovers': {'url': 'https://example.com/stats'}}, 'players': {'10': {'played': True}}}}
    settle_games(store, [final(home=17)], actuals)
    report = dashboard(store)['results'][0]
    assert report['review']['comparisons'][0]['status'] == 'surprise'
    assert report['review']['score_errors']['home'] == -7
    assert report['review']['personnel'][0]['matched'] is False
    assert report['review']['right'] and report['review']['missed']
    settle_games(store, [final(home=17)]) # A source outage cannot erase previously observed statistics.
    assert dashboard(store)['results'][0]['review']['comparisons'][0]['actual'] == 4
    assert dashboard(store)['results'][0]['review_revisions'] == 1
    settle_games(store, [final(home=17)], {('g', 'H'): {'metrics': {'turnovers': 2.}}})
    assert dashboard(store)['results'][0]['review_revisions'] == 2
    assert saved_games(store)['g']['process_expectations'] == f['process_expectations']


def test_missing_observations_are_not_zero_or_a_dnp(tmp_path):
    store = Store(tmp_path); f = prepared('g', NOW+timedelta(days=1))
    f['personnel_snapshot'] = {'home': [{'espn_id': '10', 'name': 'Example', 'likely': True}]}
    save_game(store, f, NOW); settle_games(store, [final()])
    review = dashboard(store)['results'][0]['review']
    assert review['comparisons'][0]['actual'] is None
    assert review['comparisons'][0]['status'] == 'awaiting_data'
    assert review['personnel'][0]['observed'] is None
    assert review['personnel'][0]['matched'] is None


def test_legacy_forecast_does_not_get_hindsight_expectations(tmp_path):
    store = Store(tmp_path); save_game(store, forecast(), NOW); settle_games(store, [final()])
    assert not dashboard(store)['results'][0]['review']['comparisons']
    assert 'older forecast' in dashboard(store)['results'][0]['review']['unknown'][0]
    assert update_learning(store, NOW+timedelta(days=10))['eligible_games'] == 0


def test_candidate_requires_future_shadow_test_and_promotes_once(tmp_path):
    store = Store(tmp_path)
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    batch(store, start, 5)
    as_of = start+timedelta(days=36)
    current = update_learning(store, as_of)
    assert current['eligible_games'] == 80 and current['pending'] and not current['active']
    candidate_id = current['pending']; frozen = artifact(store, candidate_id)
    assert len(frozen['training_ids']) == 80
    batch(store, as_of+timedelta(days=3), 2, 6, current)
    tested = update_learning(store, as_of+timedelta(days=18))
    assert tested['validation_games'] == 32 and not tested['active'] # Need three distinct NFL weeks.
    assert digest(artifact(store, candidate_id)) == digest(frozen)
    batch(store, as_of+timedelta(days=17), 1, 8, current)
    tested = update_learning(store, as_of+timedelta(days=24))
    assert tested['active'] == candidate_id and not tested['pending']
    assert tested['last_evaluation']['gain'] == pytest.approx(3)
    assert set(tested['last_evaluation']['forecast_ids']).isdisjoint(frozen['training_ids'])
    update_learning(store, as_of+timedelta(days=25))
    with store.connect() as c:
        events = [json.loads(r[0]) for r in c.execute('SELECT body FROM game_learning_events')]
    assert [e['kind'] for e in events].count('promoted') == 1
    next_f = prepared('next', as_of+timedelta(days=26), 9, tested, store)
    assert next_f['home_score'] == 27 and next_f['away_score'] == 23
    assert next_f['learning']['base']['home_score'] == 24
    assert abs(next_f['home_win_probability']+next_f['away_win_probability']+next_f['tie_probability']-1) < 1e-10


def test_failed_candidate_is_retired_and_not_retested(tmp_path):
    store = Store(tmp_path); start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    batch(store, start, 5)
    current = update_learning(store, start+timedelta(days=36))
    batch(store, start+timedelta(days=40), 3, 6, current, offset=0)
    result = update_learning(store, start+timedelta(days=62))
    assert result['last_evaluation']['decision'] == 'rejected'
    assert not result['active'] and not result['pending']
    again = update_learning(store, start+timedelta(days=63))
    assert again['last_evaluation'] == result['last_evaluation']


def test_promoted_correction_can_rollback_on_a_later_fixed_batch(tmp_path):
    store = Store(tmp_path); start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    batch(store, start, 5); current = update_learning(store, start+timedelta(days=36))
    batch(store, start+timedelta(days=40), 3, 6, current)
    active = update_learning(store, start+timedelta(days=62))
    batch(store, start+timedelta(days=66), 3, 9, active, offset=0)
    rolled = update_learning(store, start+timedelta(days=87))
    assert rolled['active'] is None and rolled['last_evaluation']['decision'] == 'rolled_back'


def test_week_is_not_training_until_all_archived_games_finish_and_age(tmp_path):
    store = Store(tmp_path); f = prepared('g', NOW+timedelta(days=1))
    save_game(store, f, NOW); settle_games(store, [final()])
    assert not mature_examples(store, NOW+timedelta(days=4))
    assert len(mature_examples(store, NOW+timedelta(days=6))) == 1
    save_game(store, prepared('late', NOW+timedelta(days=7)), NOW)
    assert not mature_examples(store, NOW+timedelta(days=6))


def test_ngs_uses_weekly_qualifying_players_and_missing_stays_unknown(tmp_path):
    schedule = pd.DataFrame([{'game_id': 'g', 'season': 2026, 'week': 1, 'game_type': 'REG', 'home_team': 'H', 'away_team': 'A'}])
    pd.DataFrame([{'game_id': 'g', 'team': 'H', 'passing_interceptions': 1, 'fumbles_lost_total': np.nan}]).to_parquet(tmp_path/'team_stats_2026.parquet')
    pd.DataFrame([{'season': 2026, 'week': week, 'season_type': 'REG', 'team_abbr': 'H', 'attempts': weight, 'completion_percentage_above_expectation': value} for week, weight, value in [(0, 1000, 100), (1, 30, 4), (1, 10, -4)]]).to_parquet(tmp_path/'ngs_passing.parquet')
    actual = observations(tmp_path, schedule, 2026)[('g', 'H')]
    assert actual['metrics']['ngs_cpoe'] == 2
    assert 'turnovers' not in actual['metrics']
    assert 'thresholds' in actual['sources']['ngs_cpoe']['coverage']


def test_expectation_fit_excludes_target_and_unfinished_game(tmp_path):
    schedule = []; actual = {}
    for i in range(100):
        gid = str(i)
        schedule.append({'game_id': gid, 'game_type': 'REG', 'gameday': '2026-08-01', 'gametime': '13:00', 'home_team': 'H', 'away_team': 'A', 'home_score': 24, 'away_score': 20})
        for team in ['H', 'A']: actual[(gid, team)] = {'metrics': {'rushing_yards': 110+(i % 10)}}
    target = {'game_id': 'future', 'game_type': 'REG', 'gameday': '2026-09-08', 'gametime': '13:00', 'home_team': 'H', 'away_team': 'A', 'home_score': 70, 'away_score': 0}
    schedule.append(target); actual[('future', 'H')] = {'metrics': {'rushing_yards': 100000}}
    first = fit_expectations(actual, pd.DataFrame(schedule), NOW)
    actual[('future', 'H')]['metrics']['rushing_yards'] = -100000
    second = fit_expectations(actual, pd.DataFrame(schedule), NOW)
    assert first == second
    expected = expectations(first, target, {}, {})
    assert 100 < expected['home']['rushing_yards']['expected'] < 130


def test_same_game_market_diagnostics_do_not_mix_coverage(tmp_path):
    store = Store(tmp_path)
    save_game(store, forecast() | {'market_home_moneyline': -150, 'market_away_moneyline': 130}, NOW)
    save_game(store, forecast('unpriced') | {'market_home_margin': None, 'market_total': None}, NOW)
    settle_games(store, [final(), final('unpriced')])
    d = diagnostics(saved_games(store), dashboard(store)['results'])
    assert d['market_probability']['games'] == d['market_margin']['games'] == d['market_total']['games'] == 1
    assert d['market_margin']['model_mae'] == 0
    assert d['market_margin']['market_mae'] == 3
    assert d['reliability'][0]['games'] == 2


def test_benchmark_import_rejects_backfills_and_identity_mismatch_atomically(tmp_path):
    store = Store(tmp_path); f = forecast() | {'home_team': 'DEN', 'away_team': 'KC'}
    row = {k: f[k] for k in ('game_id', 'home_team', 'away_team', 'home_score', 'away_score')}
    body = BenchmarkImport(source='Licensed test export', issued_at=NOW, rows=[row])
    with pytest.raises(ValueError, match='past imports'):
        import_benchmarks(store, body, [f], NOW+timedelta(days=2))
    wrong = BenchmarkImport(source='Licensed test export', issued_at=NOW, rows=[row, row | {'game_id': 'wrong'}])
    with pytest.raises(ValueError): import_benchmarks(store, wrong, [f], NOW)
    assert not for_game(store, f)
    assert import_benchmarks(store, body, [f], NOW)['imported'] == 1
    assert for_game(store, f)[0]['source'] == 'Licensed test export'


def test_live_schedule_scores_do_not_enter_training_labels():
    from app.game_model import train_games
    schedule=[{'game_id':str(i),'season':2025,'week':1,'game_type':'REG','gameday':'2025-09-01','gametime':'13:00','home_team':'H','away_team':'A','home_score':24,'away_score':20} for i in range(80)]
    as_of=datetime(2026,9,7,19,tzinfo=timezone.utc)
    live={'game_id':'live','season':2026,'week':1,'game_type':'REG','gameday':'2026-09-07','gametime':'13:00','home_team':'H','away_team':'A','home_score':0,'away_score':0}
    a=train_games(pd.DataFrame(schedule+[live]),as_of)
    b=train_games(pd.DataFrame(schedule+[live|{'home_score':70}]),as_of)
    assert a==b and a['games']==80


def test_audit_export_is_authenticated_and_contains_immutable_inputs(tmp_path,monkeypatch):
    from fastapi.testclient import TestClient
    import app.api as api
    store=Store(tmp_path);monkeypatch.setattr(api,'store',store)
    save_game(store,prepared('g',NOW+timedelta(days=1)),NOW);settle_games(store,[final()])
    with TestClient(api.app) as client:
        assert client.get('/api/intelligence/games/g/audit').status_code==401
        response=client.get('/api/intelligence/games/g/audit',headers={'X-Local-Token':store.token})
        assert response.status_code==200
        data=response.json()
        assert data['forecasts'][0]['learning']['family']==FAMILY
        assert data['forecasts'][0]['home_score']==24
        assert data['reviews'][0]['review']['comparisons'][0]['actual'] is None
