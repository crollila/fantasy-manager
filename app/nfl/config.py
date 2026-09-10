"""Central configuration for the NFL forecasting engine.

Everything lives under ``DATA/nfl`` (``FANTASY_DATA_DIR`` controls ``DATA``) so the
existing application conventions, backups and .gitignore rules keep working.
"""
from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path
from app.storage import DATA

NFL_ROOT = Path(os.environ.get("NFL_DATA_DIR", DATA / "nfl"))
NFLVERSE = "https://github.com/nflverse/nflverse-data/releases/download"
FIRST_SEASON = 1999
SEED = 20260909
FEATURE_VERSION = "fv1"

# Chronological evaluation design. Tuning may only look at DEVELOPMENT seasons; the
# HOLDOUT seasons are predicted once, with the frozen configuration, at the end.
DEVELOPMENT_TEST_SEASONS = tuple(range(2005, 2022))
HOLDOUT_SEASONS = (2022, 2023, 2024, 2025)
MIN_TRAIN_SEASONS = 5

# Historical prediction horizons (hours before kickoff) that the feature builder can
# reconstruct. "pregame" is the reference horizon: inactive lists and closing lines are
# public roughly 90 minutes before kickoff.
HORIZONS = {"pregame": 1.0, "early": 24.0 * 6}


@dataclass(frozen=True)
class Paths:
    root: Path = NFL_ROOT

    @property
    def raw(self) -> Path:
        return self.root / "raw"

    @property
    def normalized(self) -> Path:
        return self.root / "normalized"

    @property
    def features(self) -> Path:
        return self.root / "features"

    @property
    def models(self) -> Path:
        return self.root / "models"

    @property
    def predictions(self) -> Path:
        return self.root / "predictions"

    @property
    def reports(self) -> Path:
        return self.root / "reports"

    def ensure(self) -> "Paths":
        for p in (self.root, self.raw, self.normalized, self.features, self.models, self.predictions / "historical", self.predictions / "current", self.reports):
            p.mkdir(parents=True, exist_ok=True)
        return self


def paths(root: Path | None = None) -> Paths:
    return Paths(Path(root) if root else NFL_ROOT).ensure()
