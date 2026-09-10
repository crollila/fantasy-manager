"""Local licensed forecast exports. Imports cannot backfill a past benchmark."""
from datetime import datetime, timezone
import json
from pydantic import Field, field_validator
from app.domain import Strict
from app.game_learning import stamp


class BenchmarkRow(Strict):
    game_id: str = Field(min_length=1, max_length=80)
    home_team: str = Field(min_length=2, max_length=4)
    away_team: str = Field(min_length=2, max_length=4)
    home_score: float = Field(ge=0, le=100, allow_inf_nan=False)
    away_score: float = Field(ge=0, le=100, allow_inf_nan=False)
    home_win_probability: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)


class BenchmarkImport(Strict):
    source: str = Field(min_length=2, max_length=80)
    issued_at: datetime
    rows: list[BenchmarkRow] = Field(min_length=1, max_length=32)

    @field_validator('issued_at')
    @classmethod
    def aware(cls, value):
        if value.tzinfo is None: raise ValueError('issued_at must include a timezone')
        return value


def initialize(store):
    with store.connect() as c:
        c.execute('CREATE TABLE IF NOT EXISTS game_benchmarks(id INTEGER PRIMARY KEY AUTOINCREMENT,game_id TEXT NOT NULL,source TEXT NOT NULL,issued_at TEXT NOT NULL,received_at TEXT NOT NULL,body TEXT NOT NULL)')


def import_benchmarks(store, data, games, as_of=None):
    as_of = as_of or datetime.now(timezone.utc)
    if data.issued_at > as_of: raise ValueError('A forecast cannot be issued in the future')
    if (as_of-data.issued_at).days > 14: raise ValueError('Use a forecast export issued within the last 14 days')
    known = {g['game_id']: g for g in games}; seen = set(); prepared = []
    for row in data.rows:
        game = known.get(row.game_id)
        if not game or game.get('state') != 'pre' or stamp(game['kickoff']) <= as_of:
            raise ValueError('Every row must match an upcoming game shown in NFL games; past imports cannot enter the live record')
        if row.game_id in seen: raise ValueError('Duplicate game_id in import')
        if (row.home_team, row.away_team) != (game['home_team'], game['away_team']): raise ValueError('Home/away teams do not match the schedule')
        seen.add(row.game_id)
        prepared.append((row.game_id, data.source, data.issued_at.isoformat(), as_of.isoformat(), row.model_dump_json()))
    initialize(store)
    with store.connect() as c:
        c.executemany('INSERT INTO game_benchmarks(game_id,source,issued_at,received_at,body) VALUES(?,?,?,?,?)', prepared)
    return {'imported': len(prepared), 'source': data.source, 'note': 'Saved locally. Refresh forecasts before kickoff to archive these comparisons alongside your picks.'}


def for_game(store, game):
    initialize(store)
    with store.connect() as c:
        rows = c.execute('SELECT * FROM game_benchmarks WHERE id IN (SELECT MAX(id) FROM game_benchmarks WHERE game_id=? GROUP BY source)', (game['game_id'],)).fetchall()
    return [json.loads(r['body']) | {'source': r['source'], 'issued_at': r['issued_at'], 'received_at': r['received_at']}
            for r in rows if stamp(r['received_at']) < stamp(game['kickoff']) and stamp(r['issued_at']) < stamp(game['kickoff'])]


def comparison(forecasts, results):
    groups = {}
    for r in results:
        for b in forecasts.get(r['game_id'], {}).get('external_benchmarks', []):
            entry = groups.setdefault(b['source'], {'source': b['source'], 'games': 0, 'model_mae': 0., 'provider_mae': 0.})
            entry['games'] += 1; entry['model_mae'] += r['score_mae']
            entry['provider_mae'] += (abs(r['home_score']-b['home_score'])+abs(r['away_score']-b['away_score']))/2
    for row in groups.values():
        row['model_mae'] /= row['games']; row['provider_mae'] /= row['games']
    return list(groups.values())
