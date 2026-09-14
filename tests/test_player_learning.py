"""The weekly player correction: it learns, it stays bounded, and it cannot see its own result."""
import json
import numpy as np
from app.storage import Store
from app.tracking import initialize
from app.player_learning import fit_correction, adjustment, summary, MAX_POINTS, MAX_FRACTION, MIN_OBSERVATIONS, PRIOR_STRENGTH


def seed_results(store, rows):
    initialize(store)
    with store.connect() as c:
        for i, r in enumerate(rows):
            c.execute('INSERT OR REPLACE INTO player_results VALUES(?,?)', (i, json.dumps(r)))


def graded(position, forecast, actual, play=1., weight=.65):
    return {'position': position, 'forecast': forecast, 'actual': actual, 'play_probability': play, 'independent_weight': weight}


def test_no_correction_before_enough_graded_forecasts(tmp_path):
    store = Store(tmp_path)
    seed_results(store, [graded('RB', 10., 12.) for _ in range(MIN_OBSERVATIONS-1)])
    assert fit_correction(store) is None
    assert adjustment(None, 'RB', 10.) == 0.
    assert summary(None)['available'] is False


def test_a_systematic_under_projection_is_corrected_upward(tmp_path):
    store = Store(tmp_path)
    # Running backs consistently outscore their projection by four points.
    seed_results(store, [graded('RB', 8.+i%5, 12.+i%5) for i in range(400)])
    model = fit_correction(store)
    assert model is not None and model['observations'] == 400
    delta = adjustment(model, 'RB', 10.)
    assert delta > 0, 'a persistent under-projection should push the projection up'
    # Shrinkage keeps it below the raw fitted residual until far more evidence accrues.
    assert delta < 4.


def test_the_correction_grows_with_evidence(tmp_path):
    small, large = Store(tmp_path/'a'), Store(tmp_path/'b')
    seed_results(small, [graded('WR', 9., 13.) for _ in range(40)])
    seed_results(large, [graded('WR', 9., 13.) for _ in range(800)])
    weak = adjustment(fit_correction(small), 'WR', 9.)
    strong = adjustment(fit_correction(large), 'WR', 9.)
    assert 0 < weak < strong, 'more graded weeks should mean a larger correction'


def test_the_correction_is_bounded(tmp_path):
    store = Store(tmp_path)
    # An absurd, impossible residual must still not move a projection without limit.
    seed_results(store, [graded('QB', 20., 200.) for _ in range(2000)])
    model = fit_correction(store)
    for center in (2., 20., 60.):
        delta = adjustment(model, 'QB', center)
        assert abs(delta) <= min(MAX_POINTS, MAX_FRACTION*center)+1e-9


def test_an_unseen_position_is_left_alone(tmp_path):
    store = Store(tmp_path)
    seed_results(store, [graded('RB', 8., 14.) for _ in range(300)])
    assert adjustment(fit_correction(store), 'K', 8.) == 0., 'a position with no graded history gets no correction'


def test_only_finished_games_can_train_the_correction(tmp_path):
    """player_results holds graded (finished) forecasts only, so a live forecast cannot train on itself."""
    from app.tracking import settle_players
    import pandas as pd
    from app.domain import Player
    store = Store(tmp_path); initialize(store)
    with store.connect() as c:
        c.execute('INSERT INTO player_forecasts(league_id,player_id,game_id,created_at,kickoff,body) VALUES(?,?,?,?,?,?)',
                  ('l', 'p1', 'g-upcoming', '2026-09-01T00:00:00+00:00', '2026-09-08T00:00:00+00:00',
                   json.dumps({'position': 'RB', 'mean': 10., 'p10': 2., 'p90': 20., 'scoring': {'rushing_yards': .1}, 'bonuses': []})))
    stats = pd.DataFrame([{'game_id': 'g-upcoming', 'player_id': 'p1', 'rushing_yards': 100.}])
    # The game is not in the completed set, so nothing is graded and nothing can be learned.
    settle_players(store, stats, [Player(id='p1', name='P', position='RB')], completed_games=set())
    with store.connect() as c:
        assert c.execute('SELECT COUNT(*) FROM player_results').fetchone()[0] == 0
    assert fit_correction(store) is None
