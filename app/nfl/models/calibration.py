"""Uncertainty and probability calibration fitted on out-of-fold residuals.

* ``fit_residual_model``: joint (margin, total) residual distribution -> a heteroscedastic
  sigma model (sd as a linear function of predicted total and season era), the residual
  correlation, an empirical residual pool for simulation, and the tie rate.
* ``fit_probability_calibrator``: Platt (logistic on logit) and isotonic calibrators for
  home-win probabilities derived from the margin distribution or a classifier; the one with
  the lower OOF log loss on the development seasons is selected.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import norm, t as student_t
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


def fit_residual_model(preds: pd.DataFrame, margin_col: str, total_col: str, recent_seasons: int = 8) -> dict:
    margin = (preds["actual_home_points"] - preds["actual_away_points"]).to_numpy().astype(float)
    total = (preds["actual_home_points"] + preds["actual_away_points"]).to_numpy().astype(float)
    pm = preds[margin_col].to_numpy().astype(float)
    pt = preds[total_col].to_numpy().astype(float)
    ok = np.isfinite(pm) & np.isfinite(pt)
    rm, rt = margin[ok] - pm[ok], total[ok] - pt[ok]
    seasons = preds["season"].to_numpy()[ok]
    # Heteroscedastic sd: |resid| ~ a + b * predicted_total (E|X| = sd*sqrt(2/pi) for normal)
    A = np.vstack([np.ones(ok.sum()), pt[ok]]).T
    coef_m = np.linalg.lstsq(A, np.abs(rm) * np.sqrt(np.pi / 2), rcond=None)[0]
    coef_t = np.linalg.lstsq(A, np.abs(rt) * np.sqrt(np.pi / 2), rcond=None)[0]
    latest = seasons.max()
    recent = seasons >= latest - recent_seasons + 1
    # Student-t degrees of freedom by matching kurtosis (bounded); heavier tails than normal are typical for margins.
    k = float(pd.Series(rm[recent]).kurt())
    df = float(np.clip(6 / max(k, 1e-3) + 4, 5, 60)) if k > 0 else 60.0
    return {
        "margin_sd": float(np.std(rm)), "total_sd": float(np.std(rt)), "margin_sd_recent": float(np.std(rm[recent])), "total_sd_recent": float(np.std(rt[recent])),
        "sd_margin_coef": [float(coef_m[0]), float(coef_m[1])], "sd_total_coef": [float(coef_t[0]), float(coef_t[1])],
        "residual_corr": float(np.corrcoef(rm, rt)[0, 1]), "margin_bias": float(np.mean(rm)), "total_bias": float(np.mean(rt)), "tie_rate": float(np.mean(margin[ok] == 0)),
        "t_df": df, "n": int(ok.sum()), "residual_pool": {"margin": rm[recent].round(3).tolist(), "total": rt[recent].round(3).tolist(), "pred_total": pt[ok][recent].round(2).tolist(), "seasons": seasons[recent].tolist()},
    }


def margin_sd(model: dict, pred_total: np.ndarray) -> np.ndarray:
    a, b = model["sd_margin_coef"]
    return np.clip(a + b * np.asarray(pred_total, dtype=float), 9.0, 20.0)


def total_sd(model: dict, pred_total: np.ndarray) -> np.ndarray:
    a, b = model["sd_total_coef"]
    return np.clip(a + b * np.asarray(pred_total, dtype=float), 8.0, 20.0)


def win_probability_from_margin(pred_margin: np.ndarray, sd: np.ndarray, df: float | None = None, tie_rate: float = 0.002) -> np.ndarray:
    """P(home wins) given a margin distribution; ties (margin exactly 0) are split evenly."""
    z = np.asarray(pred_margin, dtype=float) / np.asarray(sd, dtype=float)
    p = student_t.cdf(z, df) if df else norm.cdf(z)
    return np.clip(p * (1 - tie_rate) + tie_rate / 2, 1e-4, 1 - 1e-4)


class ProbabilityCalibrator:
    def __init__(self, method: str = "platt"):
        self.method = method

    def fit(self, p: np.ndarray, y: np.ndarray):
        p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
        y = np.asarray(y, float)
        if self.method == "platt":
            self.model_ = LogisticRegression(C=1e6, max_iter=1000).fit(np.log(p / (1 - p))[:, None], y)
        elif self.method == "isotonic":
            self.model_ = IsotonicRegression(out_of_bounds="clip", y_min=0.01, y_max=0.99).fit(p, y)
        else:
            self.model_ = None
        return self

    def predict(self, p: np.ndarray) -> np.ndarray:
        p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
        if self.method == "platt":
            return self.model_.predict_proba(np.log(p / (1 - p))[:, None])[:, 1]
        if self.method == "isotonic":
            return self.model_.predict(p)
        return p

    def to_dict(self) -> dict:
        if self.method == "platt":
            return {"method": "platt", "coef": float(self.model_.coef_[0][0]), "intercept": float(self.model_.intercept_[0])}
        if self.method == "isotonic":
            return {"method": "isotonic", "x": self.model_.X_thresholds_.tolist(), "y": self.model_.y_thresholds_.tolist()}
        return {"method": "identity"}

    @classmethod
    def from_dict(cls, d: dict) -> "ProbabilityCalibrator":
        c = cls(d["method"])
        if d["method"] == "platt":
            c.model_ = LogisticRegression(C=1e6)
            c.model_.coef_ = np.array([[d["coef"]]])
            c.model_.intercept_ = np.array([d["intercept"]])
            c.model_.classes_ = np.array([0.0, 1.0])
        elif d["method"] == "isotonic":
            c.model_ = IsotonicRegression(out_of_bounds="clip", y_min=0.01, y_max=0.99).fit(np.array(d["x"]), np.array(d["y"]))
        else:
            c.model_ = None
        return c


def select_calibrator(p: np.ndarray, y: np.ndarray, seasons: np.ndarray) -> tuple[ProbabilityCalibrator, dict]:
    """Choose identity / Platt / isotonic by chronological OOF log loss (fit on earlier seasons, score later)."""
    from app.nfl.models.metrics import log_loss
    order = np.unique(seasons)
    if len(order) < 6:
        return ProbabilityCalibrator("identity").fit(p, y), {"method": "identity", "reason": "too few seasons"}
    cut = order[len(order) // 2]
    tr, te = seasons < cut, seasons >= cut
    scores = {}
    for method in ("identity", "platt", "isotonic"):
        cal = ProbabilityCalibrator(method).fit(p[tr], y[tr])
        scores[method] = log_loss(cal.predict(p[te]), y[te])
    best = min(scores, key=scores.get)
    # Only leave identity if a calibrator is materially better (avoid overfitting noise).
    if best != "identity" and scores["identity"] - scores[best] < 0.001:
        best = "identity"
    return ProbabilityCalibrator(best).fit(p, y), {"method": best, "oos_log_loss": scores}
