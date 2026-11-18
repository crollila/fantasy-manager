"""The archive grows when a forecast or review changes, not every time the app refreshes."""
import json
from datetime import datetime, timedelta, timezone

from app.storage import Store
from app.tracking import initialize, save_game, saved_games, settle_games, save_player_rows, dashboard
from app.game_learning import archive_artifact, discard_artifact, artifact
from app.ledger_compaction import compact, maybe_compact, status
from app.player_archive import archive_league

NOW = datetime(2026, 9, 9, 12, 30, tzinfo=timezone.utc)   # mid-way through a six-hour window
KICKOFF = NOW + timedelta(days=3)


def forecast(gid='g', kickoff=KICKOFF, **changes):
    base = {'game_id': gid, 'season': 2026, 'week': 1, 'home_team': 'H', 'away_team': 'A', 'home_score': 24., 'away_score': 20.,
            'pick': 'H', 'home_win_probability': .6, 'away_win_probability': .39, 'tie_probability': .01, 'margin': 4., 'total': 44.,
            'market_home_margin': 7., 'market_total': 45., 'ats_pick': 'away', 'total_pick': 'under', 'model_version': 'test',
            'kickoff': kickoff.isoformat(), 'state': 'pre', 'reasons': [], 'input_as_of': NOW.isoformat(), 'base_artifact': 'a1',
            'weather': {'status': 'forecast', 'wind_mph': 5., 'fetched_at': NOW.isoformat()},
            'process_expectations': {'home': {'turnovers': {'label': 'Turnovers lost', 'unit': 'turnovers', 'expected': 1., 'scatter': 1., 'low': 0., 'high': 2.3}}}}
    return base | changes


def final(gid='g', home=24, away=20):
    return {'game_id': gid, 'season': 2026, 'week': 1, 'home_team': 'H', 'away_team': 'A', 'home_score': home, 'away_score': away, 'completed': True}


def count(store, table):
    with store.connect() as c: return c.execute(f'SELECT COUNT(*) FROM {table}').fetchone()[0]


def drifted(minutes):
    """The same pick after another refit: new run stamps and a wobble in a secondary number."""
    f = forecast(input_as_of=(NOW+timedelta(minutes=minutes)).isoformat(), base_artifact=f'a{minutes}')
    f['process_expectations']['home']['turnovers']['expected'] = 1.+minutes/10000
    return f


def test_a_refresh_that_only_moves_run_stamps_saves_nothing(tmp_path):
    store = Store(tmp_path)
    assert save_game(store, forecast(), NOW)
    again = forecast(input_as_of=(NOW+timedelta(minutes=15)).isoformat(), base_artifact='a2')
    again['weather']['fetched_at'] = (NOW+timedelta(minutes=15)).isoformat()
    assert not save_game(store, again, NOW+timedelta(minutes=15))
    assert count(store, 'game_forecasts') == 1


def test_refit_drift_is_archived_once_per_window_not_once_per_refresh(tmp_path):
    store = Store(tmp_path)
    assert save_game(store, forecast(), NOW)
    saved = [save_game(store, drifted(m), NOW+timedelta(minutes=m)) for m in range(15, 24*60, 15)]
    # One day of quarter-hour refreshes: four six-hour windows, the first already holds a version.
    assert sum(saved) == 4
    assert saved_games(store)['g']['process_expectations']['home']['turnovers']['expected'] > 1.1


def test_a_change_the_reader_sees_is_archived_at_once(tmp_path):
    store = Store(tmp_path)
    assert save_game(store, forecast(), NOW)
    for minutes, change in enumerate([{'home_score': 24.2}, {'home_win_probability': .62}, {'market_total': 46.}, {'pick': 'A'},
                                      {'personnel_snapshot': {'home': [{'espn_id': '1', 'likely': False}]}}], 1):
        assert save_game(store, forecast(**change), NOW+timedelta(minutes=minutes)), change
    assert not save_game(store, forecast(personnel_snapshot={'home': [{'espn_id': '1', 'likely': False}]}, home_score=24.01), NOW+timedelta(minutes=9))


def test_every_change_is_kept_in_the_last_hours_before_kickoff(tmp_path):
    store = Store(tmp_path); kickoff = NOW+timedelta(minutes=90)
    assert save_game(store, forecast(kickoff=kickoff), NOW)
    late = forecast(kickoff=kickoff); late['process_expectations']['home']['turnovers']['expected'] = 1.001
    assert save_game(store, late, NOW+timedelta(minutes=15)), 'the graded version must carry the latest inputs'
    assert not save_game(store, late, kickoff)


def review_sources(fetched, sha='s1', turnovers=4.):
    return {('g', 'H'): {'metrics': {'turnovers': turnovers}, 'sources': {'turnovers': {'url': 'https://example.com/stats', 'fetched_at': fetched, 'sha256': sha, 'status': 'ok'}}}}


def test_refetching_the_same_statistics_does_not_add_a_review_revision(tmp_path):
    store = Store(tmp_path); save_game(store, forecast(), NOW)
    settle_games(store, [final(home=17)], review_sources('2026-09-14T10:00:00+00:00'))
    for hour in range(11, 20):
        # The season file is downloaded again and has grown by later games; this game is unchanged.
        settle_games(store, [final(home=17)], review_sources(f'2026-09-14T{hour}:00:00+00:00', sha=f's{hour}'))
    assert count(store, 'game_result_versions') == 1
    assert dashboard(store)['results'][0]['review_revisions'] == 1
    settle_games(store, [final(home=17)], review_sources('2026-09-15T10:00:00+00:00', turnovers=3.))
    assert count(store, 'game_result_versions') == 2, 'a corrected statistic is a real revision'
    assert dashboard(store)['results'][0]['review']['comparisons'][0]['source']['fetched_at'] == '2026-09-15T10:00:00+00:00'


def test_a_refit_wobble_is_not_a_new_player_projection(tmp_path):
    store = Store(tmp_path); initialize(store); league = archive_league(2026)
    row = {'id': 'p1', 'name': 'P', 'position': 'RB', 'mean': 10., 'p10': 3., 'p90': 19., 'play_probability': 1., 'game': {'game_id': 'g1', 'kickoff': KICKOFF.isoformat()}}
    assert save_player_rows(store, league, 1, [dict(row)], NOW) == 1
    assert save_player_rows(store, league, 1, [dict(row, mean=10.004, p10=3.002, p90=19.01)], NOW+timedelta(minutes=15)) == 0
    assert save_player_rows(store, league, 1, [dict(row, mean=10.2)], NOW+timedelta(minutes=30)) == 1
    assert save_player_rows(store, league, 1, [dict(row, mean=10.2, injury_status='QUESTIONABLE')], NOW+timedelta(minutes=45)) == 1


def test_an_uncited_foundation_artifact_is_discarded(tmp_path):
    store = Store(tmp_path)
    kept = archive_artifact(store, 'foundation', {'model': 1}, NOW)
    dropped = archive_artifact(store, 'foundation', {'model': 2}, NOW+timedelta(minutes=15))
    correction = archive_artifact(store, 'score-correction', {'model': 3}, NOW+timedelta(minutes=15))
    discard_artifact(store, dropped, NOW+timedelta(minutes=15))
    discard_artifact(store, kept, NOW+timedelta(minutes=15))        # not from this refresh
    discard_artifact(store, correction, NOW+timedelta(minutes=15))  # never a foundation
    assert artifact(store, kept) and artifact(store, correction) and artifact(store, dropped) is None


def bloated(store):
    """A data directory as an older release left it: one row per refresh in every ledger."""
    initialize(store); league = archive_league(2026)
    with store.connect() as c:
        for i in range(40):
            stamp = NOW+timedelta(minutes=15*i); f = drifted(15*i) if i else forecast()
            if i >= 20: f['pick'] = 'A'
            body = json.dumps(f, sort_keys=True)
            c.execute('INSERT INTO game_forecasts(game_id,created_at,kickoff,body,fingerprint) VALUES(?,?,?,?,?)', ('g', stamp.isoformat(), f['kickoff'], body, f'old{i}'))
            p = {'name': 'P', 'position': 'RB', 'mean': 10.+i/10000, 'p10': 3., 'p90': 19., 'play_probability': 1., 'season': 2026, 'week': 1, 'scoring': {}, 'bonuses': []}
            c.execute('INSERT INTO player_forecasts(league_id,player_id,game_id,created_at,kickoff,body) VALUES(?,?,?,?,?,?)', (league.id, 'p1', 'g', stamp.isoformat(), f['kickoff'], json.dumps(p)))
    from app.game_learning import initialize as init_learning
    init_learning(store)
    with store.connect() as c:
        for i in range(40):
            c.execute('INSERT INTO game_model_artifacts VALUES(?,?,?,?)', (f'a{15*i}' if i else 'a1', (NOW+timedelta(minutes=15*i)).isoformat(), 'foundation', '{}'))
        c.execute('INSERT INTO game_model_artifacts VALUES(?,?,?,?)', ('correction', NOW.isoformat(), 'score-correction', '{}'))
        final_id = c.execute('SELECT MAX(id) FROM game_forecasts').fetchone()[0]
        for i in range(30):
            result = {'forecast_id': final_id, 'game_id': 'g', 'review': {'comparisons': [{'actual': 4., 'source': {'fetched_at': f't{i}', 'sha256': f's{i}'}}]}}
            if i >= 25: result['review']['comparisons'][0]['actual'] = 3.
            c.execute('INSERT INTO game_result_versions(game_id,forecast_id,created_at,body,fingerprint) VALUES(?,?,?,?,?)', ('g', final_id, f't{i}', json.dumps(result), f'old{i}'))
        c.execute('INSERT INTO game_results VALUES(?,?,?)', ('g', json.dumps(result | {'base_artifact': 'a585'}), 't'))
        graded = c.execute('SELECT MAX(id) FROM player_forecasts').fetchone()[0]
        c.execute('INSERT INTO player_results VALUES(?,?)', (graded, '{}'))
    return final_id, graded


def test_compaction_removes_the_duplicates_and_keeps_everything_graded(tmp_path):
    store = Store(tmp_path); final_id, graded = bloated(store)
    with store.connect() as c: before = c.execute('SELECT id,created_at,body FROM game_forecasts WHERE id=?', (final_id,)).fetchone()[:]
    removed = compact(store)
    with store.connect() as c:
        ids = [r[0] for r in c.execute('SELECT id FROM game_forecasts ORDER BY id')]
        assert c.execute('SELECT id,created_at,body FROM game_forecasts WHERE id=?', (final_id,)).fetchone()[:] == before
        picks = [json.loads(r[0])['pick'] for r in c.execute('SELECT body FROM game_forecasts ORDER BY id')]
        reviews = [json.loads(r[0])['review']['comparisons'][0]['actual'] for r in c.execute('SELECT body FROM game_result_versions ORDER BY id')]
        players = [r[0] for r in c.execute('SELECT id FROM player_forecasts ORDER BY id')]
        artifacts = {r[0] for r in c.execute('SELECT id FROM game_model_artifacts')}
        cited = {json.loads(r[0])['base_artifact'] for r in c.execute('SELECT body FROM game_forecasts')}
    # The first version, the pick change (row 21), the next six-hour window (18:00, row 23), the final version.
    assert ids == [1, 21, 23, final_id]
    assert picks == ['H', 'A', 'A', 'A']
    assert reviews == [4., 3.]
    assert players == [1, graded]
    assert cited == {'a1', 'a300', 'a330', 'a585'} and artifacts == cited | {'correction'}
    assert removed == {'game_result_versions': 28, 'game_forecasts': 36, 'player_forecasts': 38, 'game_model_artifacts': 36}
    assert compact(store) == {'game_result_versions': 0, 'game_forecasts': 0, 'player_forecasts': 0, 'game_model_artifacts': 0}
    # Saving continues to deduplicate against the rows that survived.
    settle_games(store, [])
    assert not save_game(store, drifted(15*39) | {'pick': 'A'}, NOW+timedelta(minutes=15*39+1))


def test_the_cleanup_runs_once_and_shrinks_the_file(tmp_path):
    store = Store(tmp_path); bloated(store)
    with store.connect() as c:
        c.executemany('INSERT INTO game_result_versions(game_id,forecast_id,created_at,body,fingerprint) VALUES(?,?,?,?,?)',
                      [('g', 1, 't', json.dumps({'review': {'pad': 'x'*20000, 'fetched_at': str(i)}}), f'pad{i}') for i in range(200)])
    size = store.db.stat().st_size; stages = []
    state = maybe_compact(store, stages.append)
    assert state['version'] == 1 and state['vacuumed'] and state['removed']['game_result_versions'] >= 199
    assert store.db.stat().st_size < size/4
    assert len(stages) == 2
    assert maybe_compact(store, stages.append) == status(store) and len(stages) == 2, 'nothing to do the second time'
