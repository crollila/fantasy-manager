"""Monte Carlo score simulation from calibrated (margin, total) forecasts.

Residual pairs are resampled jointly from the out-of-fold pool (this preserves the real
margin/total dependence and the key-number structure of NFL scores), scaled to the game's
predicted spread of outcomes, then converted to integer scores. Every downstream quantity
(win probability, spread/total cover probabilities, intervals, most likely score) comes from
the same sample so the outputs are mutually consistent.
"""
from __future__ import annotations
import numpy as np
from app.nfl.models.calibration import margin_sd, total_sd

COMMON_SPREADS = (-10.5, -7.5, -7, -6.5, -3.5, -3, -2.5, -1.5, 0, 1.5, 2.5, 3, 3.5, 6.5, 7, 7.5, 10.5)
COMMON_TOTALS = (37.5, 40.5, 41.5, 42.5, 43.5, 44.5, 45.5, 46.5, 47.5, 48.5, 49.5, 51.5, 54.5)


def simulate_game(pred_margin: float, pred_total: float, residual_model: dict, n: int = 50000, seed: int = 0, extra_margin_sd: float = 0.0) -> dict:
    rng = np.random.default_rng(seed)
    # Centre the pooled residuals: the point forecast already carries the ensemble's expectation.
    pool_m = np.asarray(residual_model["residual_pool"]["margin"], dtype=float)
    pool_t = np.asarray(residual_model["residual_pool"]["total"], dtype=float)
    pool_m = pool_m - pool_m.mean()
    pool_t = pool_t - pool_t.mean()
    idx = rng.integers(0, len(pool_m), n)
    # Scale the pooled residuals to this game's heteroscedastic sd (pool sd -> game sd).
    sd_m = float(margin_sd(residual_model, np.array([pred_total]))[0])
    sd_t = float(total_sd(residual_model, np.array([pred_total]))[0])
    sd_m = float(np.sqrt(sd_m ** 2 + extra_margin_sd ** 2))
    m = pred_margin + pool_m[idx] * (sd_m / max(pool_m.std(), 1e-6))
    t = pred_total + pool_t[idx] * (sd_t / max(pool_t.std(), 1e-6))
    home = np.rint((t + m) / 2).astype(int)
    away = np.rint((t - m) / 2).astype(int)
    home = np.clip(home, 0, None)
    away = np.clip(away, 0, None)
    # Overtime resolves most ties: keep the historical tie rate, split the remainder by a coin flip.
    tie = home == away
    keep_tie = rng.random(n) < residual_model.get("tie_rate", 0.002) / max(tie.mean(), 1e-9)
    flip = tie & ~keep_tie
    winner_home = rng.random(n) < 0.5
    home = np.where(flip & winner_home, home + 3, home)
    away = np.where(flip & ~winner_home, away + 3, away)
    margin = home - away
    total = home + away
    out = {
        "home_win": float(np.mean(margin > 0)), "away_win": float(np.mean(margin < 0)), "tie": float(np.mean(margin == 0)),
        "home_mean": float(home.mean()), "away_mean": float(away.mean()), "home_median": float(np.median(home)), "away_median": float(np.median(away)),
        "margin_mean": float(margin.mean()), "margin_median": float(np.median(margin)), "total_mean": float(total.mean()), "total_median": float(np.median(total)),
        "margin_sd": float(margin.std()), "total_sd": float(total.std()), "home_sd": float(home.std()), "away_sd": float(away.std()),
        "home_interval_80": [float(np.percentile(home, 10)), float(np.percentile(home, 90))], "away_interval_80": [float(np.percentile(away, 10)), float(np.percentile(away, 90))],
        "home_interval_50": [float(np.percentile(home, 25)), float(np.percentile(home, 75))], "away_interval_50": [float(np.percentile(away, 25)), float(np.percentile(away, 75))],
        "margin_interval_80": [float(np.percentile(margin, 10)), float(np.percentile(margin, 90))], "total_interval_80": [float(np.percentile(total, 10)), float(np.percentile(total, 90))],
        # Spread lines are quoted for the home team (negative = home favoured). Integer lines can push.
        "spread_cover": [{"home_line": float(s), "home_cover": float(np.mean(margin + s > 0)), "push": float(np.mean(margin + s == 0)), "away_cover": float(np.mean(margin + s < 0))} for s in COMMON_SPREADS],
        "total_over": [{"line": float(t_), "over": float(np.mean(total > t_)), "push": float(np.mean(total == t_)), "under": float(np.mean(total < t_))} for t_ in COMMON_TOTALS],
        "upset_probability": float(min(np.mean(margin > 0), np.mean(margin < 0))),
        "samples": int(n),
    }
    scores, counts = np.unique(np.stack([home, away], axis=1), axis=0, return_counts=True)
    top = np.argsort(counts)[::-1][:5]
    out["most_likely_scores"] = [{"home": int(scores[i][0]), "away": int(scores[i][1]), "probability": float(counts[i] / n)} for i in top]
    hist_m, edges_m = np.histogram(margin, bins=np.arange(-45.5, 46.5, 1.0))
    out["margin_distribution"] = {str(int(e + 0.5)): float(c / n) for e, c in zip(edges_m[:-1], hist_m) if c > 0}
    hist_t, edges_t = np.histogram(total, bins=np.arange(-0.5, 100.5, 1.0))
    out["total_distribution"] = {str(int(e + 0.5)): float(c / n) for e, c in zip(edges_t[:-1], hist_t) if c > 0}
    return out


def market_probabilities(sim_margin: np.ndarray, spread: float | None) -> dict:
    if spread is None or not np.isfinite(spread):
        return {}
    return {"home_cover": float(np.mean(sim_margin - spread > 0)), "push": float(np.mean(sim_margin - spread == 0))}
