"""A duplicated player in the catalog must never block ESPN imports or catalog rebuilds."""
import json
import numpy as np
import pandas as pd
from app.domain import Player
from app.identity import Identity, merge_catalog, merge_duplicates, referenced_ids
from app.league_sync import save_snapshot, sync_metadata
from app.player_model import control_points, feature_row
from app.storage import Store
from tests.test_weekly import fixture


def kicker(pid, **ids):
    return Player(id=pid, name='Jake Bates', position='K', team='DET', ids=ids)


def test_duplicates_sharing_an_external_id_collapse_into_the_referenced_entry():
    espn = kicker('espn:4689936', gsis='00-0039172', espn='4689936')
    nflverse = kicker('00-0039172', gsis='00-0039172', espn='4689936').model_copy(update={'stats': {'fg_made': 1.5}})
    merged = merge_duplicates([nflverse, espn], prefer={'espn:4689936'})
    assert [p.id for p in merged] == ['espn:4689936']
    assert merged[0].stats == {'fg_made': 1.5}   # history is kept from the other copy
    Identity(merged)


def test_without_references_the_entry_with_history_survives():
    bare = kicker('espn:1', espn='1')
    known = kicker('00-1', gsis='00-1', espn='1').model_copy(update={'stats': {'fg_made': 2.}})
    assert [p.id for p in merge_duplicates([bare, known])] == ['00-1']


def test_entries_at_different_positions_are_not_merged_but_the_conflict_is_removed():
    a = Player(id='a', name='X', position='RB', ids={'espn': '7'})
    b = Player(id='b', name='X', position='WR', ids={'espn': '7', 'gsis': 'g'})
    merged = merge_duplicates([a, b])
    assert {p.id for p in merged} == {'a', 'b'}
    Identity(merged)


def test_a_rebuild_folds_new_ids_into_existing_entries_without_duplicating():
    # nflverse learns the ESPN id of a player first catalogued under two separate entries.
    existing = [Player(id='espn:4686658', name='Mike Washington Jr.', position='RB', ids={'espn': '4686658'}),
                Player(id='00-0040878', name='Mike Washington Jr.', position='RB', ids={'gsis': '00-0040878'})]
    built = [Player(id='00-0040878', name='Mike Washington Jr.', position='RB', team='LV',
                    ids={'gsis': '00-0040878', 'espn': '4686658'}, stats={'carries': 9.})]
    merged = merge_catalog(existing, built, prefer={'espn:4686658'})
    assert len(merged) == 1 and merged[0].id == 'espn:4686658' and merged[0].stats == {'carries': 9.}
    Identity(merged)


def test_espn_import_succeeds_when_the_saved_catalog_holds_a_duplicate(tmp_path):
    store = Store(tmp_path)
    save_snapshot(store, fixture(), 5, 2026, 1)
    players = store.players(2026)
    duplicate = players[0].model_copy(update={'id': '00-0000100', 'ids': players[0].ids | {'gsis': '00-0000100'}})
    store.save_players(2026, players + [duplicate])   # the state that blocked every re-sync
    save_snapshot(store, fixture(), 5, 2026, 2)
    assert sync_metadata(store, '123')['week'] == 2
    Identity(store.players(2026))
    assert 'espn:100' in referenced_ids(store)


def test_a_repeated_feature_row_does_not_break_a_projection():
    rows = pd.DataFrame({'use_receptions_ewm': [3., 5.], 'use_receiving_yards_ewm': [40., 60.]},
                        index=pd.Index(['p1', 'p1'], name='player_id'))
    row = feature_row(rows, 'p1')
    assert row == {'use_receptions_ewm': 5., 'use_receiving_yards_ewm': 60.}
    assert np.isfinite(control_points(row))
    assert feature_row(rows, 'missing') is None and feature_row(None, 'p1') is None


def test_control_points_ignores_non_numeric_values():
    assert control_points({'use_receptions_ewm': 'n/a', 'use_receiving_yards_ewm': None}) == 0.
