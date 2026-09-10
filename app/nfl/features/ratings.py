"""Continuously updated team strength ratings, all computed point-in-time.

* Elo (margin-aware, preseason regression) updated sequentially game by game: the rating a
  team carries into a game only reflects games that kicked off earlier.
* Ridge margin ratings (SRS-style, opponent adjusted, recency weighted) refit at every
  (season, week) snapshot using games that ended before that week's first kickoff.
* Ridge efficiency ratings: opponent-adjusted offensive and defensive EPA / success /
  points-per-drive, refit at the same weekly snapshots.

Snapshots are deliberately conservative: a Thursday result is not used for the same week's
Sunday games. That loses a little information but can never leak.
"""
from __future__ import annotations
import numpy as np
import polars as pl
from app.nfl.reference.teams import TEAMS

TEAM_INDEX = {t: i for i, t in enumerate(TEAMS)}
N = len(TEAMS)


def elo_features(games: pl.DataFrame, k: float = 20.0, hfa: float = 55.0, revert: float = 1 / 3, base: float = 1505.0) -> pl.DataFrame:
    """Sequential Elo. Returns pre-game ratings and derived expectations per game."""
    g = games.sort(["kickoff_utc", "game_id"]).to_dicts()
    rating = {t: 1500.0 for t in TEAMS}
    last_season = {t: None for t in TEAMS}
    rows = []
    for r in g:
        h, a, season = r["home_team"], r["away_team"], r["season"]
        for t in (h, a):
            if t not in rating:
                rating[t] = 1500.0
                last_season[t] = None
            if last_season[t] is not None and last_season[t] != season:
                rating[t] = rating[t] + revert * (base - rating[t])  # offseason regression toward the mean
            last_season[t] = season
        home_adv = 0.0 if r["neutral"] else hfa
        diff = rating[h] + home_adv - rating[a]
        p_home = 1.0 / (1.0 + 10 ** (-diff / 400.0))
        rows.append({"game_id": r["game_id"], "home_elo_pre": rating[h], "away_elo_pre": rating[a], "elo_diff": diff, "elo_prob_home": p_home, "elo_margin": diff / 25.0})
        if r["completed"] and r["home_score"] is not None:
            margin = float(r["home_score"] - r["away_score"])
            outcome = 1.0 if margin > 0 else 0.0 if margin < 0 else 0.5
            mult = np.log(abs(margin) + 1.0) * (2.2 / ((diff if margin > 0 else -diff) * 0.001 + 2.2))
            shift = k * mult * (outcome - p_home)
            rating[h] += shift
            rating[a] -= shift
    return pl.DataFrame(rows, infer_schema_length=None)


def _design(team_a: np.ndarray, team_b: np.ndarray, home: np.ndarray, extra: np.ndarray | None = None) -> np.ndarray:
    n = len(team_a)
    x = np.zeros((n, 2 * N + 1))
    x[np.arange(n), team_a] = 1.0
    x[np.arange(n), N + team_b] = 1.0
    x[:, 2 * N] = home
    return x


def _ridge(x: np.ndarray, y: np.ndarray, w: np.ndarray, alpha: float) -> np.ndarray:
    xw = x * w[:, None]
    a = x.T @ xw + alpha * np.eye(x.shape[1])
    a[-1, -1] -= alpha * 0.9  # only lightly shrink the home-field term
    return np.linalg.solve(a, xw.T @ y)


def snapshot_keys(games: pl.DataFrame) -> pl.DataFrame:
    """(season, week) -> cutoff = earliest kickoff of that week."""
    return games.group_by(["season", "week"]).agg(pl.col("kickoff_utc").min().alias("cutoff")).sort(["season", "week"])


def margin_ratings(games: pl.DataFrame, half_life_days: float = 400.0, alpha: float = 4.0, window_days: float = 1100.0) -> pl.DataFrame:
    """Weekly-snapshot ridge ratings on game margins (offense+defense combined)."""
    done = games.filter(pl.col("completed") & pl.col("kickoff_utc").is_not_null()).sort("kickoff_utc")
    ko = done["kickoff_utc"].to_numpy().astype("datetime64[s]").astype("int64")
    hi = np.array([TEAM_INDEX[t] for t in done["home_team"].to_list()])
    ai = np.array([TEAM_INDEX[t] for t in done["away_team"].to_list()])
    home = np.where(done["neutral"].to_numpy(), 0.0, 1.0)
    y = done["margin"].to_numpy().astype(float)
    x_all = np.zeros((len(y), N + 1))
    x_all[np.arange(len(y)), hi] += 1.0
    x_all[np.arange(len(y)), ai] -= 1.0
    x_all[:, N] = home
    keys = snapshot_keys(games)
    out = []
    for season, week, cutoff in keys.iter_rows():
        if cutoff is None:
            continue
        c = np.datetime64(cutoff.replace(tzinfo=None), "s").astype("int64")
        mask = (ko < c) & (ko > c - window_days * 86400)
        if mask.sum() < 60:
            out.append({"season": season, "week": week, "ratings": np.zeros(N), "hfa": 2.0, "n": int(mask.sum())})
            continue
        age_days = (c - ko[mask]) / 86400.0
        w = 0.5 ** (age_days / half_life_days)
        x = x_all[mask]
        # Sum-to-zero constraint via ridge on team effects; home term barely shrunk.
        xw = x * w[:, None]
        a = x.T @ xw + alpha * np.eye(N + 1)
        a[N, N] = a[N, N] - alpha + 1e-6
        beta = np.linalg.solve(a, xw.T @ y[mask])
        out.append({"season": season, "week": week, "ratings": beta[:N] - beta[:N].mean(), "hfa": float(beta[N]), "n": int(mask.sum())})
    rows = []
    for item in out:
        for t, i in TEAM_INDEX.items():
            rows.append({"season": item["season"], "week": item["week"], "team": t, "srs": float(item["ratings"][i]), "srs_hfa": item["hfa"], "srs_games": item["n"]})
    return pl.DataFrame(rows, infer_schema_length=None)


EFFICIENCY_METRICS = {"epa_play": 0.8, "epa_pass": 0.8, "epa_rush": 0.8, "success_rate": 0.8, "points_drive": 0.8, "epa_neutral": 0.8, "explosive20": 0.8}


def efficiency_ratings(games: pl.DataFrame, team_games: pl.DataFrame, half_life_days: float = 300.0, window_days: float = 800.0) -> pl.DataFrame:
    """Opponent-adjusted offense/defense ratings per metric at weekly snapshots.

    Model per metric: value_ij = off_i + def_j + home + e, ridge-shrunk toward the league mean.
    ``off`` is the team's offensive effect, ``def`` the effect a defense has on opponents
    (negative = good defense)."""
    tg = team_games.filter(pl.col("completed") & pl.col("kickoff_utc").is_not_null()).sort("kickoff_utc")
    ko = tg["kickoff_utc"].to_numpy().astype("datetime64[s]").astype("int64")
    oi = np.array([TEAM_INDEX[t] for t in tg["team"].to_list()])
    di = np.array([TEAM_INDEX[t] for t in tg["opponent"].to_list()])
    home = tg["is_home"].to_numpy().astype(float)
    x_all = _design(oi, di, home)
    metrics = [m for m in EFFICIENCY_METRICS if m in tg.columns]
    ys = {m: tg[m].to_numpy().astype(float) for m in metrics}
    keys = snapshot_keys(games)
    rows = []
    for season, week, cutoff in keys.iter_rows():
        if cutoff is None:
            continue
        c = np.datetime64(cutoff.replace(tzinfo=None), "s").astype("int64")
        mask = (ko < c) & (ko > c - window_days * 86400)
        n = int(mask.sum())
        if n < 100:
            for t in TEAMS:
                rows.append({"season": season, "week": week, "team": t} | {f"off_adj_{m}": None for m in metrics} | {f"def_adj_{m}": None for m in metrics} | {"eff_games": n})
            continue
        age_days = (c - ko[mask]) / 86400.0
        w = 0.5 ** (age_days / half_life_days)
        x = x_all[mask]
        results = {}
        for m in metrics:
            y = ys[m][mask]
            ok = np.isfinite(y)
            mean = float(np.average(y[ok], weights=w[ok]))
            beta = _ridge(x[ok], y[ok] - mean, w[ok], alpha=EFFICIENCY_METRICS[m] * max(1.0, n / 400.0))
            off = beta[:N] - beta[:N].mean()
            de = beta[N:2 * N] - beta[N:2 * N].mean()
            results[m] = (off, de, mean)
        for t, i in TEAM_INDEX.items():
            row = {"season": season, "week": week, "team": t, "eff_games": n}
            for m in metrics:
                off, de, mean = results[m]
                row[f"off_adj_{m}"] = float(off[i])
                row[f"def_adj_{m}"] = float(de[i])
                row[f"league_{m}"] = mean
            rows.append(row)
    return pl.DataFrame(rows, infer_schema_length=None)
