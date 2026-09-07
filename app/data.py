"""Versioned, cached public data. Optional sources never erase the last good catalog."""
from __future__ import annotations
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import httpx
import pandas as pd
from app.storage import DATA, now

BASE = "https://github.com/nflverse/nflverse-data/releases/download"


class Cache:
    def __init__(self, path=DATA / "cache"):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)

    def fetch(self, key, url, ttl=86400, force=False):
        target = self.path / key
        meta = target.with_suffix(target.suffix + ".json")
        if target.exists() and not force and time.time() - target.stat().st_mtime < ttl:
            return target
        error = None
        for attempt in range(3):
            try:
                with httpx.Client(follow_redirects=True, timeout=60) as client:
                    r = client.get(url, headers={"User-Agent": "FantasyManager/0.1 local research"})
                    r.raise_for_status()
                if not r.content or r.content[:20].lower().startswith(b"<!doctype"):
                    raise ValueError("Empty response or HTML instead of data")
                temp = target.with_suffix(target.suffix + ".tmp")
                temp.write_bytes(r.content)
                # Validate before replacing a valid cache.
                if key.endswith(".parquet"):
                    pd.read_parquet(temp)
                elif key.endswith(".csv"):
                    pd.read_csv(temp, nrows=2)
                temp.replace(target)
                meta.write_text(json.dumps({"url": url, "fetched_at": now(), "sha256": hashlib.sha256(r.content).hexdigest(), "bytes": len(r.content), "status": "ok"}, indent=2))
                return target
            except (httpx.HTTPError, ValueError, OSError) as exc:
                error = str(exc)
                if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code in (403, 404):
                    break
                time.sleep(.25 * 2 ** attempt)
        if target.exists():
            old = json.loads(meta.read_text()) if meta.exists() else {}
            meta.write_text(json.dumps(old | {"status": "stale", "error": error, "last_attempt": now()}))
            return target
        raise ValueError(f"{key}: {error}")

    def status(self):
        return [{"file": p.name.removesuffix(".json"), **json.loads(p.read_text())} for p in self.path.glob("*.json")]


def refresh_sources(season=2026, years=5, force=False, path=DATA / "cache"):
    cache = Cache(path)
    sources = {
        "players.parquet": f"{BASE}/players/players.parquet",
        f"team_stats_{season-1}.parquet": f"{BASE}/stats_team/stats_team_week_{season-1}.parquet",
        f"roster_{season}.parquet": f"{BASE}/rosters/roster_{season}.parquet",
        "schedules.parquet": f"{BASE}/schedules/games.parquet",
    }
    for year in range(season - years, season):
        sources[f"stats_{year}.parquet"] = f"{BASE}/stats_player/stats_player_week_{year}.parquet"
    def one(item):
        key, url = item
        try:
            file = cache.fetch(key, url, force=force)
            return {"file": key, "ok": True, "rows": len(pd.read_parquet(file))}
        except Exception as exc:
            return {"file": key, "ok": False, "error": str(exc)}
    with ThreadPoolExecutor(max_workers=4) as pool:
        return list(pool.map(one, sources.items()))


STAT_COLUMNS = ["passing_yards", "passing_tds", "passing_interceptions", "attempts", "completions", "carries", "rushing_yards", "rushing_tds", "targets", "receptions", "receiving_yards", "receiving_tds", "fumbles_lost", "two_point_conversions"]


def historical(cache_path=DATA / "cache"):
    frames = []
    for path in Path(cache_path).glob("stats_*.parquet"):
        df = pd.read_parquet(path)
        if "season_type" in df:
            df = df[df.season_type == "REG"]
        if "player_id" not in df:
            continue
        if "fumbles_lost" not in df:
            df["fumbles_lost"] = sum((df.get(k, 0).fillna(0) if k in df else 0) for k in ["sack_fumbles_lost", "rushing_fumbles_lost", "receiving_fumbles_lost"])
        if "two_point_conversions" not in df:
            df["two_point_conversions"] = sum((df.get(k, 0).fillna(0) if k in df else 0) for k in ["passing_2pt_conversions", "rushing_2pt_conversions", "receiving_2pt_conversions"])
        for col in STAT_COLUMNS:
            if col not in df:
                df[col] = 0.
        df = df.sort_values(["season", "week"]).drop_duplicates(["player_id", "season", "week"])
        agg = {c: "sum" for c in STAT_COLUMNS}
        agg |= {"week": "nunique", "player_display_name": "last", "position": "last"}
        if "recent_team" in df:
            agg["recent_team"] = "last"
        if "team" in df:
            agg["team"] = "last"
        season = df.groupby(["player_id", "season"], as_index=False).agg(agg).rename(columns={"week": "games"})
        if "recent_team" in season:
            season["team"] = season["recent_team"]
        frames.append(season)
    if not frames:
        raise ValueError("No historical data. Run refresh or explicitly load demo data.")
    return pd.concat(frames, ignore_index=True).drop_duplicates(["player_id", "season"])


def supplemental_schema():
    return {"market": "List of {id, market_stats, adp, adp_sd, espn_rank, ecr}. IDs must already exist. Market weights require pre-season historical errors or explicit provisional weighting.", "events": "Structured Event objects with known_at, occurred_at and source_url; only confirmed events affect projections.", "players": "Player objects for missing rookies/K/DST, with stable canonical IDs and source provenance; never inferred via names."}
