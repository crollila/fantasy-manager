"""The automatic loop: refit on a completed week, archive every player, grade the last
pregame snapshot, feed results back, and resume from persisted state after a restart.

Nothing here runs a real refit or reaches the network; the expensive calls are replaced so
the wiring, the triggers and the no-leakage rules are what is actually under test.
"""
import json
from datetime import datetime, timedelta, timezone
import pandas as pd
import pytest
from app.storage import Store
from app.domain import Player
from app.tracking import initialize, save_player_rows, settle_players, final_pregame_forecasts
from app.intelligence import get_meta
from app.auto_refit import maybe_refit, last_completed_week, STATE_KEY
from app.player_archive import archive_league, relevant_players, archive_projections, ARCHIVE_LEAGUE

NOW = datetime(2026, 9, 15, 12, 0, tzinfo=timezone.utc)


def schedule(complete_through=2, weeks=4, season=2026):
    rows = []
    for week in range(1, weeks + 1):
        for i in range(2):
            done = week <= complete_through
            rows.append({'season': season, 'week': week, 'game_type': 'REG', 'game_id': f'{season}_{week:02d}_G{i}',
                         'home_score': 24. if done else None, 'away_score': 17. if done else None})
    return pd.DataFrame(rows)


class FakeRegistry:
    """Stands in for the on-disk model registry; `state` is shared so a refit is visible."""
    state = {}

    def champion(self, kind):
        return dict(self.state) if self.state else None


@pytest.fixture
def registry(monkeypatch):
    FakeRegistry.state = {'model_id': 'champ-1'}
    import app.nfl.models.registry as reg
    monkeypatch.setattr(reg, 'Registry', FakeRegistry)
    return FakeRegistry


def stub_refit(monkeypatch, calls, model_id='champ-2', fail=None):
    import app.nfl.models.train as train

    def refit_champion(root=None):
        calls.append(root)
        if fail:
            raise RuntimeError(fail)
        FakeRegistry.state = {'model_id': model_id, 'trained_through': {'season': 2026, 'week': 2, 'games': 7300}}
        return {'model_id': model_id, 'refit_of': 'champ-1', 'training_games': 7300}
    monkeypatch.setattr(train, 'refit_champion', refit_champion)
    return calls


# --- 1. The refit is automatic -------------------------------------------------------

def test_the_latest_fully_completed_week_is_detected():
    assert last_completed_week(schedule(complete_through=2), 2026) == (2026, 2)
    assert last_completed_week(schedule(complete_through=0), 2026) is None


def test_a_half_finished_week_does_not_count_as_completed():
    frame = schedule(complete_through=1)
    mask = (frame.week == 2) & (frame.game_id.str.endswith('G0'))
    frame.loc[mask, 'home_score'] = 31.
    frame.loc[mask, 'away_score'] = 28.
    assert last_completed_week(frame, 2026) == (2026, 1), 'one finished game does not complete a week'


def test_a_completed_week_triggers_a_refit_with_no_command(tmp_path, registry, monkeypatch):
    store = Store(tmp_path)
    initialize(store)
    calls = stub_refit(monkeypatch, [])
    result = maybe_refit(store, schedule(complete_through=2), 2026, NOW)
    assert result['action'] == 'refit' and len(calls) == 1
    assert result['model_id'] == 'champ-2'


def test_no_refit_when_the_champion_already_covers_the_completed_week(tmp_path, registry, monkeypatch):
    store = Store(tmp_path)
    initialize(store)
    FakeRegistry.state = {'model_id': 'champ-1', 'trained_through': {'season': 2026, 'week': 2}}
    calls = stub_refit(monkeypatch, [])
    assert maybe_refit(store, schedule(complete_through=2), 2026, NOW)['action'] == 'up to date'
    assert calls == [], 'an up-to-date champion must not be refit again'


def test_the_next_completed_week_triggers_another_refit(tmp_path, registry, monkeypatch):
    store = Store(tmp_path)
    initialize(store)
    calls = stub_refit(monkeypatch, [])
    maybe_refit(store, schedule(complete_through=2), 2026, NOW)
    assert maybe_refit(store, schedule(complete_through=2), 2026, NOW)['action'] == 'up to date'
    stub_refit(monkeypatch, calls, model_id='champ-3')
    FakeRegistry.state['trained_through'] = {'season': 2026, 'week': 2}
    assert maybe_refit(store, schedule(complete_through=3), 2026, NOW)['action'] == 'refit'
    assert len(calls) == 2, 'the loop refits once per newly completed week'


# --- 2. Restart recovery -------------------------------------------------------------

def test_state_survives_a_restart_and_does_not_repeat_the_refit(tmp_path, registry, monkeypatch):
    store = Store(tmp_path)
    initialize(store)
    calls = stub_refit(monkeypatch, [])
    assert maybe_refit(store, schedule(complete_through=2), 2026, NOW)['action'] == 'refit'
    saved = get_meta(store, STATE_KEY)
    assert saved['model_id'] == 'champ-2' and saved['week'] == 2
    # A brand-new Store on the same directory is exactly what a restart produces.
    restarted = Store(tmp_path)
    assert get_meta(restarted, STATE_KEY)['model_id'] == 'champ-2', 'training history is persisted, not in memory'
    assert maybe_refit(restarted, schedule(complete_through=2), 2026, NOW)['action'] == 'up to date'
    assert len(calls) == 1, 'a restart must not re-run a refit that already happened'


def test_a_failed_refit_is_recorded_and_not_retried_every_refresh(tmp_path, registry, monkeypatch):
    store = Store(tmp_path)
    initialize(store)
    calls = stub_refit(monkeypatch, [], fail='feature table missing')
    first = maybe_refit(store, schedule(complete_through=2), 2026, NOW)
    assert first['action'] == 'failed' and 'feature table missing' in first['error']
    second = maybe_refit(store, schedule(complete_through=2), 2026, NOW)
    assert second['action'].startswith('skipped') and len(calls) == 1
    # A later completed week is still attempted: one bad week does not stop the loop forever.
    stub_refit(monkeypatch, calls, model_id='champ-9')
    assert maybe_refit(store, schedule(complete_through=3), 2026, NOW)['action'] == 'refit'


def test_a_refit_failure_never_propagates(tmp_path, registry, monkeypatch):
    store = Store(tmp_path)
    initialize(store)
    stub_refit(monkeypatch, [], fail='boom')
    assert maybe_refit(store, schedule(complete_through=2), 2026, NOW)['action'] == 'failed'


# --- 3. Future predictions use the newly refit model ---------------------------------

def test_forecasts_load_the_champion_fresh_so_a_refit_takes_effect_immediately(monkeypatch):
    """engine_forecasts must re-read the registry per call, not hold a model in memory."""
    import polars as pl
    import app.nfl.integration as integration
    import app.nfl.predict as predict
    import app.nfl.features.builder as builder
    seen = []

    class Reg:
        def champion(self, kind):
            return dict(FakeRegistry.state)

        def load(self, model_id):
            seen.append(model_id)
            return {'model_id': model_id}
    monkeypatch.setattr(integration, 'Registry', Reg)
    monkeypatch.setattr(builder, 'load_features', lambda kind, root=None: (pl.DataFrame({'game_id': ['g1']}), {'meta': True}))
    monkeypatch.setattr(predict, 'forecast_games', lambda rows, meta, artifact, n_sims=0: [])

    FakeRegistry.state = {'model_id': 'champ-1'}
    _, first = integration.engine_forecasts(['g1'], refresh=False)
    FakeRegistry.state = {'model_id': 'champ-2'}   # a refit lands between refreshes
    _, second = integration.engine_forecasts(['g1'], refresh=False)
    assert seen == ['champ-1', 'champ-2'], 'the refit model must be picked up without a restart'
    assert first.get('model_id') == 'champ-1' and second.get('model_id') == 'champ-2'


# --- 4. Every relevant player is archived --------------------------------------------

def board(kickoff, state='pre'):
    return [{'game_id': 'g-sea-ne', 'state': state, 'kickoff': kickoff, 'home_team': 'SEA', 'away_team': 'NE'},
            {'game_id': 'g-tb-cin', 'state': state, 'kickoff': kickoff, 'home_team': 'TB', 'away_team': 'CIN'}]


def test_every_player_with_an_upcoming_game_is_relevant():
    players = [Player(id='a', name='A', position='QB', team='SEA'), Player(id='b', name='B', position='WR', team='TB'),
               Player(id='c', name='C', position='RB', team='DAL')]   # DAL is not playing this week
    picked = {p.id for p in relevant_players(players, board('2026-09-20T17:00:00+00:00'), 3)}
    assert picked == {'a', 'b'}, 'players without a game this week are not projected'


def test_players_in_a_game_already_underway_are_not_archived():
    players = [Player(id='a', name='A', position='QB', team='SEA')]
    assert relevant_players(players, board('2026-09-20T17:00:00+00:00', state='in'), 3) == []


def test_the_archive_saves_a_row_for_every_projected_player(tmp_path, monkeypatch):
    import app.player_archive as archive
    store = Store(tmp_path)
    initialize(store)
    players = [Player(id=f'p{i}', name=f'P{i}', position='RB', team='SEA') for i in range(5)]
    store.save_players(2026, players)
    kickoff = (NOW + timedelta(days=2)).isoformat()
    # project_week itself is exercised by the weekly tests; here the archiving path is under test.
    monkeypatch.setattr(archive, 'project_week', lambda p, lg, wk, prov, inj, health, cache, draws=0, context=None: (
        {'id': p.id, 'name': p.name, 'position': p.position, 'mean': 10., 'p10': 3., 'p90': 19.,
         'play_probability': 1., 'game': {'game_id': 'g-sea-ne', 'kickoff': kickoff}}, None), raising=False)
    result = archive_projections(store, 2026, 3, board(kickoff), {'status': 'fresh', 'players': {}}, {}, tmp_path, NOW)
    assert result['projected'] == 5 and result['saved'] == 5
    with store.connect() as c:
        rows = c.execute('SELECT league_id,player_id FROM player_forecasts').fetchall()
    assert len(rows) == 5 and {r['league_id'] for r in rows} == {ARCHIVE_LEAGUE}


def test_an_unchanged_projection_does_not_create_a_duplicate_version(tmp_path):
    store = Store(tmp_path)
    initialize(store)
    league = archive_league(2026)
    kickoff = (NOW + timedelta(days=2)).isoformat()
    row = {'id': 'p1', 'name': 'P', 'position': 'RB', 'mean': 10., 'p10': 3., 'p90': 19.,
           'play_probability': 1., 'game': {'game_id': 'g1', 'kickoff': kickoff}}
    assert save_player_rows(store, league, 3, [dict(row)], NOW) == 1
    assert save_player_rows(store, league, 3, [dict(row)], NOW + timedelta(minutes=15)) == 0
    moved = dict(row, mean=13.5)
    assert save_player_rows(store, league, 3, [moved], NOW + timedelta(minutes=30)) == 1, 'a changed projection is archived'


def test_a_projection_is_never_archived_at_or_after_kickoff(tmp_path):
    store = Store(tmp_path)
    initialize(store)
    league = archive_league(2026)
    row = {'id': 'p1', 'name': 'P', 'position': 'RB', 'mean': 10., 'p10': 3., 'p90': 19.,
           'game': {'game_id': 'g1', 'kickoff': NOW.isoformat()}}
    assert save_player_rows(store, league, 3, [row], NOW) == 0, 'saving at kickoff is refused'
    assert save_player_rows(store, league, 3, [row], NOW + timedelta(minutes=1)) == 0


# --- 5. The last pre-kickoff snapshot is the graded one -------------------------------

def insert_forecast(store, created_at, kickoff, mean, league='L', player='p1', game='g1'):
    with store.connect() as c:
        c.execute('INSERT INTO player_forecasts(league_id,player_id,game_id,created_at,kickoff,body) VALUES(?,?,?,?,?,?)',
                  (league, player, game, created_at, kickoff,
                   json.dumps({'position': 'RB', 'mean': mean, 'p10': 1., 'p90': 30., 'scoring': {'rushing_yards': .1}, 'bonuses': []})))
        return c.execute('SELECT last_insert_rowid()').fetchone()[0]


def test_the_final_pregame_version_is_selected_not_the_first(tmp_path):
    store = Store(tmp_path)
    initialize(store)
    kickoff = '2026-09-20T17:00:00+00:00'
    insert_forecast(store, '2026-09-17T10:00:00+00:00', kickoff, 9.)
    last = insert_forecast(store, '2026-09-20T16:30:00+00:00', kickoff, 14.)
    with store.connect() as c:
        picked = final_pregame_forecasts(c, {'g1'})
    assert [r['id'] for r in picked] == [last]


def test_a_version_saved_after_kickoff_never_supersedes_the_pregame_one(tmp_path):
    store = Store(tmp_path)
    initialize(store)
    kickoff = '2026-09-20T17:00:00Z'   # the other ISO spelling, on purpose
    pregame = insert_forecast(store, '2026-09-20T16:55:00+00:00', kickoff, 12.)
    insert_forecast(store, '2026-09-20T19:30:00+00:00', kickoff, 25.)   # written mid-game
    with store.connect() as c:
        picked = final_pregame_forecasts(c, {'g1'})
    assert [r['id'] for r in picked] == [pregame], 'a post-kickoff row must never be graded'


def test_a_player_with_only_post_kickoff_versions_is_not_graded(tmp_path):
    store = Store(tmp_path)
    initialize(store)
    insert_forecast(store, '2026-09-20T19:00:00+00:00', '2026-09-20T17:00:00+00:00', 25.)
    with store.connect() as c:
        assert final_pregame_forecasts(c, {'g1'}) == []


def test_grading_uses_the_last_pregame_snapshot_end_to_end(tmp_path):
    store = Store(tmp_path)
    initialize(store)
    kickoff = '2026-09-20T17:00:00+00:00'
    insert_forecast(store, '2026-09-17T10:00:00+00:00', kickoff, 9.)
    last = insert_forecast(store, '2026-09-20T16:30:00+00:00', kickoff, 14.)
    insert_forecast(store, '2026-09-20T20:00:00+00:00', kickoff, 30.)
    stats = pd.DataFrame([{'game_id': 'g1', 'player_id': 'p1', 'rushing_yards': 100.}])
    settle_players(store, stats, [Player(id='p1', name='P', position='RB')], {'g1'})
    with store.connect() as c:
        rows = [json.loads(r[0]) for r in c.execute('SELECT body FROM player_results')]
    assert len(rows) == 1
    assert rows[0]['forecast_id'] == last and rows[0]['forecast'] == 14.
    assert rows[0]['actual'] == pytest.approx(10.), 'graded against the real result, not a projection'


# --- 6. Completed results become training data automatically -------------------------

def test_a_completed_week_immediately_enters_player_training(tmp_path):
    from app.player_learning import fit_correction
    store = Store(tmp_path)
    initialize(store)
    stats = []
    for i in range(40):
        insert_forecast(store, '2026-09-19T10:00:00+00:00', '2026-09-20T17:00:00+00:00', 8., player=f'p{i}', game='g1')
        stats.append({'game_id': 'g1', 'player_id': f'p{i}', 'rushing_yards': 140.})
    assert fit_correction(store) is None, 'nothing is learned before the games are graded'
    settle_players(store, pd.DataFrame(stats), [Player(id=f'p{i}', name='P', position='RB') for i in range(40)], {'g1'})
    model = fit_correction(store)
    assert model is not None and model['observations'] == 40
    assert model['mean_residual'] > 0, 'the completed week is now training data'


def test_an_unfinished_game_contributes_nothing_to_training(tmp_path):
    from app.player_learning import fit_correction
    store = Store(tmp_path)
    initialize(store)
    for i in range(40):
        insert_forecast(store, '2026-09-19T10:00:00+00:00', '2026-09-20T17:00:00+00:00', 8., player=f'p{i}', game='g-live')
    stats = pd.DataFrame([{'game_id': 'g-live', 'player_id': f'p{i}', 'rushing_yards': 140.} for i in range(40)])
    settle_players(store, stats, [Player(id=f'p{i}', name='P', position='RB') for i in range(40)], completed_games=set())
    assert fit_correction(store) is None, 'a game that has not finished can never become training data'
