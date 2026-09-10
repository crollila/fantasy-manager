"""Champion / challenger model registry.

The registry is a JSON index plus one artifact folder per model version. A challenger is
promoted only when its out-of-sample metrics beat the champion by more than the noise
floor (paired weekly bootstrap) on the same games.
"""
from __future__ import annotations
import hashlib
import json
import pickle
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from app.nfl.config import paths, FEATURE_VERSION


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL, timeout=5).decode().strip()
    except Exception:  # noqa: BLE001
        return None


def fingerprint(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    except OSError:
        return None


def bundled_models_dir() -> Path | None:
    """Champion models shipped inside the packaged application (PyInstaller data folder)."""
    from app.storage import ROOT
    candidate = ROOT / "nfl_models"
    return candidate if (candidate / "registry.json").exists() else None


class Registry:
    def __init__(self, root: Path | None = None):
        self.models = paths(root).models
        self.index = self.models / "registry.json"
        if not self.index.exists():
            bundled = bundled_models_dir()
            if bundled is not None:
                self.seed_from(bundled)
            else:
                self.index.write_text(json.dumps({"champion": {}, "models": []}, indent=2))

    def seed_from(self, bundled: Path) -> None:
        """Copy a bundled registry (champion + artifacts) into an empty data directory."""
        import shutil
        body = json.loads((bundled / "registry.json").read_text())
        for record in body.get("models", []):
            source = bundled / "artifacts" / record["model_id"]
            target = self.models / "artifacts" / record["model_id"]
            if source.exists() and not target.exists():
                shutil.copytree(source, target)
            record["artifact"] = str(target / "model.pkl")
        body.setdefault("history", []).append({"kind": "game_forecast", "seeded_from_bundle": str(bundled), "at": datetime.now(timezone.utc).isoformat()})
        self.write(body)

    def read(self) -> dict:
        return json.loads(self.index.read_text())

    def write(self, body: dict) -> None:
        self.index.write_text(json.dumps(body, indent=2, default=str))

    def register(self, model_id: str, kind: str, artifact: dict, metrics: dict, config: dict, notes: str = "") -> dict:
        folder = self.models / "artifacts" / model_id
        folder.mkdir(parents=True, exist_ok=True)
        with open(folder / "model.pkl", "wb") as fh:
            pickle.dump(artifact, fh)
        record = {
            "model_id": model_id, "kind": kind, "created_at": datetime.now(timezone.utc).isoformat(), "git_commit": git_commit(), "feature_version": FEATURE_VERSION,
            "metrics": metrics, "config": config, "notes": notes, "artifact": str(folder / "model.pkl"),
            "dataset_fingerprint": fingerprint(paths().features / FEATURE_VERSION / "games_pregame.parquet"),
        }
        (folder / "card.json").write_text(json.dumps(record, indent=2, default=str))
        body = self.read()
        body["models"] = [m for m in body["models"] if m["model_id"] != model_id] + [record]
        self.write(body)
        return record

    def load(self, model_id: str) -> dict:
        path = self.models / "artifacts" / model_id / "model.pkl"
        if not path.exists():
            bundled = bundled_models_dir()
            if bundled is not None and (bundled / "artifacts" / model_id / "model.pkl").exists():
                path = bundled / "artifacts" / model_id / "model.pkl"
        with open(path, "rb") as fh:
            return pickle.load(fh)

    def champion(self, kind: str) -> dict | None:
        body = self.read()
        model_id = body["champion"].get(kind)
        return next((m for m in body["models"] if m["model_id"] == model_id), None) if model_id else None

    def promote(self, kind: str, model_id: str, reason: str) -> None:
        body = self.read()
        previous = body["champion"].get(kind)
        body["champion"][kind] = model_id
        body.setdefault("history", []).append({"kind": kind, "promoted": model_id, "previous": previous, "at": datetime.now(timezone.utc).isoformat(), "reason": reason})
        self.write(body)

    @staticmethod
    def challenger_beats_champion(challenger: dict, champion: dict | None, min_margin_gain: float = 0.05, min_logloss_gain: float = 0.002) -> tuple[bool, str]:
        """Promotion rule on shared holdout games: meaningful, bootstrap-supported gains and no regression in probability quality."""
        if champion is None:
            return True, "no champion registered"
        c = challenger.get("holdout") or challenger.get("metrics", {}).get("holdout", {})
        k = champion.get("holdout") or champion.get("metrics", {}).get("holdout", {})
        if not c or not k:
            return False, "missing holdout metrics"
        gain = k.get("margin_mae", 0) - c.get("margin_mae", 0)
        ll_gain = k.get("log_loss", 0) - c.get("log_loss", 0)
        boot = challenger.get("bootstrap_vs_champion", {})
        supported = boot.get("ci95", [0, 0])[1] < 0 if boot else False
        if gain >= min_margin_gain and supported and ll_gain >= -min_logloss_gain:
            return True, f"margin MAE gain {gain:.3f} (bootstrap CI {boot.get('ci95')}) and log loss change {ll_gain:+.4f}"
        return False, f"insufficient evidence: margin MAE gain {gain:.3f}, bootstrap {boot.get('ci95') if boot else 'n/a'}, log loss change {ll_gain:+.4f}"
