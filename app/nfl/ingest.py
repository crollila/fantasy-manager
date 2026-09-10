"""Immutable raw downloads from nflverse with manifests (source, retrieval time, hash, schema).

Raw files are never modified after download. A refresh with ``force=True`` re-downloads and
replaces the file only after the new content parses. A 404 (season not yet published) is
recorded in the manifest as ``unavailable`` and never blocks the rest of the pipeline.
"""
from __future__ import annotations
import hashlib
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import httpx
import polars as pl
from app.nfl.config import NFLVERSE, FIRST_SEASON, paths

log = logging.getLogger(__name__)
USER_AGENT = "FantasyManager-NFL/1.0 (local research)"


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def current_season(now: datetime | None = None) -> int:
    now = now or datetime.now(timezone.utc)
    return now.year if now.month >= 3 else now.year - 1


def dataset_specs(season: int | None = None) -> list[dict]:
    """Every nflverse asset the engine uses, with the seasons it is published for."""
    season = season or current_season()
    specs = [
        {"dataset": "schedules", "name": "games.parquet", "url": f"{NFLVERSE}/schedules/games.parquet", "ttl": 900, "cadence": "several times daily in season"},
        {"dataset": "players", "name": "players.parquet", "url": f"{NFLVERSE}/players/players.parquet", "ttl": 86400, "cadence": "daily"},
        {"dataset": "officials", "name": "officials.parquet", "url": f"{NFLVERSE}/officials/officials.parquet", "ttl": 86400, "cadence": "weekly in season"},
        {"dataset": "espn_qbr", "name": "qbr_week_level.parquet", "url": f"{NFLVERSE}/espn_data/qbr_week_level.parquet", "ttl": 21600, "cadence": "weekly in season"},
        {"dataset": "draft_picks", "name": "draft_picks.parquet", "url": f"{NFLVERSE}/draft_picks/draft_picks.parquet", "ttl": 7 * 86400, "cadence": "yearly"},
        {"dataset": "combine", "name": "combine.parquet", "url": f"{NFLVERSE}/combine/combine.parquet", "ttl": 7 * 86400, "cadence": "yearly"},
    ]
    for kind in ("passing", "rushing", "receiving"):
        specs.append({"dataset": "nextgen_stats", "name": f"ngs_{kind}.parquet", "url": f"{NFLVERSE}/nextgen_stats/ngs_{kind}.parquet", "ttl": 21600, "cadence": "nightly in season"})
    for year in range(FIRST_SEASON, season + 1):
        live = 3600 if year == season else 30 * 86400
        specs.append({"dataset": "pbp", "name": f"play_by_play_{year}.parquet", "url": f"{NFLVERSE}/pbp/play_by_play_{year}.parquet", "ttl": live, "season": year, "cadence": "nightly in season"})
    for year in range(2001, season + 1):
        specs.append({"dataset": "depth_charts", "name": f"depth_charts_{year}.parquet", "url": f"{NFLVERSE}/depth_charts/depth_charts_{year}.parquet", "ttl": 3600 if year >= season - 1 else 30 * 86400, "season": year, "cadence": "several times daily in season (2025+); weekly before"})
    for year in range(2002, season + 1):
        specs.append({"dataset": "weekly_rosters", "name": f"roster_weekly_{year}.parquet", "url": f"{NFLVERSE}/weekly_rosters/roster_weekly_{year}.parquet", "ttl": 3600 if year == season else 30 * 86400, "season": year, "cadence": "weekly in season"})
    for year in range(2009, season + 1):
        specs.append({"dataset": "injuries", "name": f"injuries_{year}.parquet", "url": f"{NFLVERSE}/injuries/injuries_{year}.parquet", "ttl": 1800 if year == season else 30 * 86400, "season": year, "cadence": "several times daily in season"})
    for year in range(2012, season + 1):
        specs.append({"dataset": "snap_counts", "name": f"snap_counts_{year}.parquet", "url": f"{NFLVERSE}/snap_counts/snap_counts_{year}.parquet", "ttl": 3600 if year == season else 30 * 86400, "season": year, "cadence": "nightly in season"})
    for year in range(2016, season + 1):
        specs.append({"dataset": "participation", "name": f"pbp_participation_{year}.parquet", "url": f"{NFLVERSE}/pbp_participation/pbp_participation_{year}.parquet", "ttl": 86400 if year == season else 30 * 86400, "season": year, "cadence": "weekly in season"})
    for year in range(2022, season + 1):
        specs.append({"dataset": "ftn_charting", "name": f"ftn_charting_{year}.parquet", "url": f"{NFLVERSE}/ftn_charting/ftn_charting_{year}.parquet", "ttl": 86400 if year == season else 30 * 86400, "season": year, "cadence": "weekly in season"})
    for year in range(2018, season + 1):
        for kind in ("pass", "rush", "rec", "def"):
            specs.append({"dataset": "pfr_advstats", "name": f"advstats_week_{kind}_{year}.parquet", "url": f"{NFLVERSE}/pfr_advstats/advstats_week_{kind}_{year}.parquet", "ttl": 86400 if year == season else 30 * 86400, "season": year, "cadence": "weekly in season"})
    return specs


class RawStore:
    def __init__(self, root: Path | None = None):
        self.root = Path(root) if root else paths().raw
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, dataset: str, name: str) -> Path:
        return self.root / dataset / name

    def manifest_path(self, dataset: str, name: str) -> Path:
        return self.root / dataset / (name + ".manifest.json")

    def manifest(self, dataset: str, name: str) -> dict:
        p = self.manifest_path(dataset, name)
        return json.loads(p.read_text()) if p.exists() else {}

    def fetch(self, spec: dict, force: bool = False) -> Path | None:
        dataset, name, url, ttl = spec["dataset"], spec["name"], spec["url"], spec.get("ttl", 86400)
        target = self.path(dataset, name)
        target.parent.mkdir(parents=True, exist_ok=True)
        manifest = self.manifest(dataset, name)
        if target.exists() and not force and time.time() - target.stat().st_mtime < ttl:
            return target
        if not target.exists() and manifest.get("status") == "unavailable" and not force:
            checked = manifest.get("checked_at")
            if checked and (datetime.now(timezone.utc) - datetime.fromisoformat(checked)).total_seconds() < ttl:
                return None
        error = None
        for attempt in range(4):
            try:
                with httpx.Client(follow_redirects=True, timeout=180) as client:
                    r = client.get(url, headers={"User-Agent": USER_AGENT})
                if r.status_code == 404:
                    body = {"url": url, "status": "unavailable", "http_status": 404, "checked_at": utcnow(), "note": "Not published upstream (yet)"}
                    if "season" in spec:
                        body["season"] = spec["season"]
                    self._write_manifest(dataset, name, body)
                    return target if target.exists() else None
                r.raise_for_status()
                if not r.content or r.content[:20].lower().startswith(b"<!doctype"):
                    raise ValueError("Empty response or HTML instead of data")
                temp = target.with_suffix(target.suffix + ".tmp")
                temp.write_bytes(r.content)
                frame = pl.read_parquet(temp)  # validate before replacing a good file
                temp.replace(target)
                info = {"url": url, "dataset": dataset, "retrieved_at": utcnow(), "sha256": hashlib.sha256(r.content).hexdigest(), "bytes": len(r.content), "rows": frame.height, "columns": frame.width, "schema": {c: str(t) for c, t in zip(frame.columns, frame.dtypes)}, "status": "ok", "cadence": spec.get("cadence"), "license": "nflverse data (CC-BY 4.0); FTN charting shared through nflverse for non-commercial use"}
                if "season" in spec:
                    info["season"] = spec["season"]
                for col in ("season", "week"):
                    if col in frame.columns and frame.height:
                        try:
                            info[f"{col}_min"] = int(frame[col].min())
                            info[f"{col}_max"] = int(frame[col].max())
                        except Exception:  # noqa: BLE001 - coverage metadata is best effort
                            pass
                self._write_manifest(dataset, name, info)
                return target
            except (httpx.HTTPError, ValueError, OSError) as exc:
                error = str(exc)
                if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code in (403,):
                    break
                time.sleep(0.5 * 2 ** attempt)
        if target.exists():
            self._write_manifest(dataset, name, manifest | {"status": "stale", "error": error, "last_attempt": utcnow()})
            return target
        self._write_manifest(dataset, name, {"url": url, "status": "failed", "error": error, "checked_at": utcnow()})
        return None

    def _write_manifest(self, dataset: str, name: str, body: dict) -> None:
        self.manifest_path(dataset, name).write_text(json.dumps(body, indent=2, default=str))

    def files(self, dataset: str) -> list[Path]:
        folder = self.root / dataset
        return sorted(p for p in folder.glob("*.parquet")) if folder.exists() else []

    def status(self) -> list[dict]:
        rows = []
        for manifest in sorted(self.root.glob("*/*.manifest.json")):
            body = json.loads(manifest.read_text())
            rows.append({"dataset": manifest.parent.name, "file": manifest.name.removesuffix(".manifest.json")} | {k: body.get(k) for k in ("status", "retrieved_at", "rows", "bytes", "season", "season_min", "season_max", "week_min", "week_max", "error", "checked_at")})
        return rows


def ingest(season: int | None = None, force: bool = False, datasets: set[str] | None = None, workers: int = 8, root: Path | None = None) -> list[dict]:
    """Download (or refresh) every raw asset. Returns one status row per asset."""
    store = RawStore(root)
    specs = [s for s in dataset_specs(season) if not datasets or s["dataset"] in datasets]

    def one(spec):
        started = time.time()
        try:
            path = store.fetch(spec, force=force)
            manifest = store.manifest(spec["dataset"], spec["name"])
            return {"dataset": spec["dataset"], "file": spec["name"], "status": manifest.get("status", "missing"), "rows": manifest.get("rows"), "seconds": round(time.time() - started, 1), "path": str(path) if path else None}
        except Exception as exc:  # noqa: BLE001 - one failing asset must not stop ingestion
            return {"dataset": spec["dataset"], "file": spec["name"], "status": "error", "error": str(exc)}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(one, specs))
    summary_path = paths().reports / "ingest_status.json"
    summary_path.write_text(json.dumps({"run_at": utcnow(), "season": season or current_season(), "results": results}, indent=2))
    return results


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    rows = ingest(force="--force" in sys.argv)
    for r in rows:
        print(f"{r['status']:>12} {r['dataset']:>16} {r['file']}")
