"""Forecast evaluation: score/margin/total errors, proper scoring rules, calibration, bootstrap."""
from __future__ import annotations
import numpy as np
import pandas as pd


def _clip(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=float), eps, 1 - eps)


def log_loss(p: np.ndarray, y: np.ndarray) -> float:
    p = _clip(p)
    y = np.asarray(y, dtype=float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((np.asarray(p, dtype=float) - np.asarray(y, dtype=float)) ** 2))


def expected_calibration_error(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi) if hi < 1 else (p >= lo) & (p <= hi)
        if m.any():
            ece += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(ece)


def reliability_table(p: np.ndarray, y: np.ndarray, bins: int = 10) -> list[dict]:
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi) if hi < 1 else (p >= lo) & (p <= hi)
        if m.any():
            rows.append({"bin": f"{lo:.1f}-{hi:.1f}", "n": int(m.sum()), "predicted": float(p[m].mean()), "observed": float(y[m].mean())})
    return rows


def crps_normal(mu: np.ndarray, sigma: np.ndarray, x: np.ndarray) -> float:
    """CRPS of a normal predictive distribution (closed form)."""
    from scipy.stats import norm
    mu, sigma, x = (np.asarray(v, dtype=float) for v in (mu, sigma, x))
    z = (x - mu) / sigma
    return float(np.mean(sigma * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))))


def crps_samples(samples: np.ndarray, x: np.ndarray) -> float:
    """CRPS from Monte Carlo samples, shape (n_games, n_samples)."""
    samples = np.asarray(samples, dtype=float)
    x = np.asarray(x, dtype=float)[:, None]
    term1 = np.mean(np.abs(samples - x), axis=1)
    s = np.sort(samples, axis=1)
    n = s.shape[1]
    # E|X - X'| via order statistics: (2/n^2) * sum_i (2i - n - 1) * x_(i)
    coef = (2 * np.arange(1, n + 1) - n - 1) / (n * n)
    term2 = np.sum(coef * s, axis=1)
    return float(np.mean(term1 - term2))


def evaluate(frame: pd.DataFrame, prefix: str = "pred_") -> dict:
    """Evaluate a prediction table with actual_home_points/actual_away_points and pred_* columns."""
    f = frame.dropna(subset=["actual_home_points", "actual_away_points", f"{prefix}home_points", f"{prefix}away_points"])
    if f.empty:
        return {"games": 0}
    ah, aa = f["actual_home_points"].to_numpy(float), f["actual_away_points"].to_numpy(float)
    ph, pa = f[f"{prefix}home_points"].to_numpy(float), f[f"{prefix}away_points"].to_numpy(float)
    margin, total = ah - aa, ah + aa
    pm, pt = ph - pa, ph + pa
    out = {
        "games": int(len(f)), "score_mae": float(np.mean(np.concatenate([np.abs(ah - ph), np.abs(aa - pa)]))), "score_rmse": float(np.sqrt(np.mean(np.concatenate([(ah - ph) ** 2, (aa - pa) ** 2])))),
        "margin_mae": float(np.mean(np.abs(margin - pm))), "margin_rmse": float(np.sqrt(np.mean((margin - pm) ** 2))), "margin_bias": float(np.mean(pm - margin)),
        "total_mae": float(np.mean(np.abs(total - pt))), "total_rmse": float(np.sqrt(np.mean((total - pt) ** 2))), "total_bias": float(np.mean(pt - total)),
    }
    if f"{prefix}home_win_probability" in f.columns:
        p_all = f[f"{prefix}home_win_probability"].to_numpy(float)
        decided = (margin != 0) & np.isfinite(p_all)
        p = p_all[decided]
        y = (margin[decided] > 0).astype(float)
        if len(y):
            out.update({"log_loss": log_loss(p, y), "brier": brier(p, y), "ece": expected_calibration_error(p, y), "accuracy": float(np.mean((p > 0.5) == (y > 0.5))), "decided_games": int(len(y))})
    if f"{prefix}margin_sd" in f.columns:
        sd = f[f"{prefix}margin_sd"].to_numpy(float)
        out["crps_margin"] = crps_normal(pm, sd, margin)
        z = np.abs(margin - pm) / sd
        out["coverage_80_margin"] = float(np.mean(z <= 1.28155))
        out["coverage_50_margin"] = float(np.mean(z <= 0.67449))
    if f"{prefix}total_sd" in f.columns:
        sd = f[f"{prefix}total_sd"].to_numpy(float)
        out["crps_total"] = crps_normal(pt, sd, total)
        out["coverage_80_total"] = float(np.mean(np.abs(total - pt) / sd <= 1.28155))
    return out


def bootstrap_difference(a_err: np.ndarray, b_err: np.ndarray, groups: np.ndarray, n: int = 2000, seed: int = 7) -> dict:
    """Paired bootstrap (resampling whole groups, e.g. weeks) of mean(a_err) - mean(b_err)."""
    a_err, b_err, groups = np.asarray(a_err, float), np.asarray(b_err, float), np.asarray(groups)
    keys, inv = np.unique(groups, return_inverse=True)
    diff = a_err - b_err
    sums = np.bincount(inv, weights=diff, minlength=len(keys))
    counts = np.bincount(inv, minlength=len(keys)).astype(float)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n):
        idx = rng.integers(0, len(keys), len(keys))
        draws.append(sums[idx].sum() / counts[idx].sum())
    draws = np.array(draws)
    return {"mean_difference": float(diff.mean()), "ci95": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))], "groups": int(len(keys)), "p_better": float(np.mean(draws < 0))}
