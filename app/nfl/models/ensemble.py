"""Ensembles fitted on out-of-fold walk-forward predictions only.

``fit_weights`` learns non-negative weights (summing to one) for a set of base-model columns
by minimising absolute error on development-season OOF predictions. A simple average is
always evaluated alongside; the learned blend must beat it to be used.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.optimize import minimize


def fit_weights(preds: pd.DataFrame, columns: list[str], target: np.ndarray, loss: str = "mae") -> dict:
    X = preds[columns].to_numpy().astype(float)
    ok = np.all(np.isfinite(X), axis=1) & np.isfinite(target)
    X, y = X[ok], target[ok]
    k = X.shape[1]

    def objective(w):
        pred = X @ w
        return np.mean(np.abs(pred - y)) if loss == "mae" else np.mean((pred - y) ** 2)

    best = None
    for start in ([np.ones(k) / k] + [np.eye(k)[i] for i in range(k)]):
        res = minimize(objective, start, method="SLSQP", bounds=[(0, 1)] * k, constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}], options={"maxiter": 300})
        if best is None or res.fun < best.fun:
            best = res
    weights = np.clip(best.x, 0, 1)
    weights = weights / weights.sum()
    simple = objective(np.ones(k) / k)
    singles = {c: float(objective(np.eye(k)[i])) for i, c in enumerate(columns)}
    return {"columns": columns, "weights": {c: float(w) for c, w in zip(columns, weights)}, "loss": float(best.fun), "simple_average_loss": float(simple), "single_losses": singles, "n": int(ok.sum())}


def apply_weights(preds: pd.DataFrame, weights: dict) -> np.ndarray:
    cols = weights["columns"]
    X = preds[cols].to_numpy().astype(float)
    w = np.array([weights["weights"][c] for c in cols])
    # If a base prediction is missing, renormalise over the available ones.
    mask = np.isfinite(X)
    Xz = np.where(mask, X, 0.0)
    denom = (mask * w).sum(axis=1)
    return np.where(denom > 0, (Xz * w).sum(axis=1) / np.where(denom > 0, denom, 1.0), np.nan)


def stacking_ridge(preds: pd.DataFrame, columns: list[str], target: np.ndarray, alpha: float = 50.0) -> dict:
    """Linear stacker with intercept (ridge) as a challenger to the constrained blend."""
    X = preds[columns].to_numpy().astype(float)
    ok = np.all(np.isfinite(X), axis=1) & np.isfinite(target)
    Xc = np.hstack([X[ok], np.ones((ok.sum(), 1))])
    a = Xc.T @ Xc + alpha * np.eye(Xc.shape[1])
    a[-1, -1] -= alpha
    beta = np.linalg.solve(a, Xc.T @ target[ok])
    pred = Xc @ beta
    return {"columns": columns, "coef": {c: float(b) for c, b in zip(columns, beta[:-1])}, "intercept": float(beta[-1]), "loss": float(np.mean(np.abs(pred - target[ok]))), "n": int(ok.sum())}


def apply_stacking(preds: pd.DataFrame, model: dict) -> np.ndarray:
    X = preds[model["columns"]].to_numpy().astype(float)
    beta = np.array([model["coef"][c] for c in model["columns"]])
    return X @ beta + model["intercept"]
