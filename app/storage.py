from __future__ import annotations
import json
import os
import secrets
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from app.domain import League, Player, Pick, Event, validate_picks

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("FANTASY_DATA_DIR", ROOT / "storage"))


def now():
    return datetime.now(timezone.utc).isoformat()


class Store:
    def __init__(self, path: Path = DATA):
        self.path = path
        path.mkdir(parents=True, exist_ok=True)
        self.db = path / "fantasy.sqlite3"
        with self.connect() as c:
            c.executescript("""
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS leagues(id TEXT PRIMARY KEY, body TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS catalogs(season INTEGER PRIMARY KEY, body TEXT NOT NULL, updated TEXT NOT NULL, revision INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS drafts(league TEXT PRIMARY KEY, body TEXT NOT NULL, revision INTEGER NOT NULL, updated TEXT NOT NULL, source TEXT);
            CREATE TABLE IF NOT EXISTS events(id TEXT PRIMARY KEY, body TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
            """)
            c.execute("INSERT OR IGNORE INTO meta VALUES('token', ?)", (secrets.token_urlsafe(32),))

    @contextmanager
    def connect(self):
        c = sqlite3.connect(self.db, timeout=30)
        c.row_factory = sqlite3.Row
        try:
            with c:
                yield c
        finally:
            c.close()

    @property
    def token(self):
        with self.connect() as c:
            return c.execute("SELECT value FROM meta WHERE key='token'").fetchone()[0]

    def leagues(self):
        with self.connect() as c:
            return [League.model_validate_json(r[0]) for r in c.execute("SELECT body FROM leagues ORDER BY id")]

    def league(self, league_id):
        with self.connect() as c:
            r = c.execute("SELECT body FROM leagues WHERE id=?", (league_id,)).fetchone()
            if not r:
                raise ValueError("League not found")
            return League.model_validate_json(r[0])

    def save_league(self, league: League):
        with self.connect() as c:
            c.execute("BEGIN IMMEDIATE")
            existing = c.execute("SELECT body FROM drafts WHERE league=?", (league.id,)).fetchone()
            if existing and json.loads(existing[0]):
                picks = [Pick.model_validate(p) for p in json.loads(existing[0])]
                validate_picks(league, picks, self.players(league.season))
            c.execute("INSERT OR REPLACE INTO leagues VALUES(?,?)", (league.id, league.model_dump_json()))

    def save_players(self, season: int, players: list[Player]):
        if len({p.id for p in players}) != len(players):
            raise ValueError("Duplicate canonical player IDs")
        with self.connect() as c:
            c.execute("INSERT INTO catalogs VALUES(?,?,?,1) ON CONFLICT(season) DO UPDATE SET body=excluded.body, updated=excluded.updated, revision=catalogs.revision+1", (season, json.dumps([p.model_dump() for p in players]), now()))

    def players(self, season):
        with self.connect() as c:
            r = c.execute("SELECT body FROM catalogs WHERE season=?", (season,)).fetchone()
        return [Player.model_validate(x) for x in json.loads(r[0])] if r else []

    def catalog_meta(self, season):
        with self.connect() as c:
            r = c.execute("SELECT updated,revision FROM catalogs WHERE season=?", (season,)).fetchone()
        return dict(r) if r else {"updated": None, "revision": 0}

    def draft(self, league_id):
        with self.connect() as c:
            r = c.execute("SELECT * FROM drafts WHERE league=?", (league_id,)).fetchone()
        return {"picks": json.loads(r["body"]), "revision": r["revision"], "updated": r["updated"], "source": r["source"]} if r else {"picks": [], "revision": 0, "updated": None, "source": None}

    def save_draft(self, league, picks, source="manual", expected_revision=None, allow_reset=False):
        picks = sorted(picks, key=lambda p: p.number)
        validate_picks(league, picks, self.players(league.season))
        with self.connect() as c:
            c.execute("BEGIN IMMEDIATE")
            old = c.execute("SELECT * FROM drafts WHERE league=?", (league.id,)).fetchone()
            revision = old["revision"] if old else 0
            old_picks = json.loads(old["body"]) if old else []
            body = [p.model_dump() for p in picks]
            if expected_revision is not None and expected_revision != revision:
                raise ValueError("Draft changed; reload before editing")
            if not allow_reset and body[:len(old_picks)] != old_picks:
                raise ValueError("Conflicting or truncated snapshot; explicitly reset to replace draft")
            if body == old_picks:
                return revision
            revision += 1
            c.execute("INSERT OR REPLACE INTO drafts VALUES(?,?,?,?,?)", (league.id, json.dumps(body), revision, now(), source))
            return revision

    def events(self):
        with self.connect() as c:
            return [Event.model_validate_json(r[0]) for r in c.execute("SELECT body FROM events ORDER BY id")]

    def save_event(self, event):
        with self.connect() as c:
            c.execute("INSERT OR REPLACE INTO events VALUES(?,?)", (event.id, event.model_dump_json()))
