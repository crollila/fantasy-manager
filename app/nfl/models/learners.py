"""Model wrappers with one interface: fit(X, y, w, seasons) -> self; predict(X) -> np.ndarray.

Hyper-parameters that need data (ridge alpha, boosting rounds) are chosen on an internal
chronological split: the last two training seasons act as validation, the model is then
refit on all training data. Nothing from the test season is touched.
"""
from __future__ import annotations
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from app.nfl.config import SEED


class RidgeModel:
    def __init__(self, alphas=(30.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0, 30000.0, 100000.0), indicators: bool = True, classifier: bool = False, alpha: float | None = None):
        self.alphas = alphas
        self.indicators = indicators
        self.classifier = classifier
        self.alpha = alpha
        self.best_alpha_ = None

    def _prep(self, X: np.ndarray, fit: bool) -> np.ndarray:
        if fit:
            self.missing_cols_ = np.where(np.isnan(X).mean(axis=0) > 0.01)[0]
            self.imputer_ = SimpleImputer(strategy="median")
            self.scaler_ = StandardScaler()
            base = self.imputer_.fit_transform(X)
            base = np.nan_to_num(base, nan=0.0)
            self.scaler_.fit(base)
        base = np.nan_to_num(self.imputer_.transform(X), nan=0.0)
        out = self.scaler_.transform(base)
        out = np.clip(out, -8, 8)
        if self.indicators and len(self.missing_cols_):
            out = np.hstack([out, np.isnan(X[:, self.missing_cols_]).astype(float)])
        return out

    def _make(self, alpha: float):
        if self.classifier:
            return LogisticRegression(C=1.0 / max(alpha, 1e-6), max_iter=2000)
        return Ridge(alpha=alpha)

    def fit(self, X, y, w=None, seasons=None):
        Xp = self._prep(X, fit=True)
        if self.alpha is not None:
            self.best_alpha_ = self.alpha
        elif seasons is not None and len(np.unique(seasons)) >= 5:
            cut = np.sort(np.unique(seasons))[-2]
            tr, va = seasons < cut, seasons >= cut
            best, best_score = None, np.inf
            for a in self.alphas:
                m = self._make(a).fit(Xp[tr], y[tr], sample_weight=None if w is None else w[tr])
                if self.classifier:
                    p = np.clip(m.predict_proba(Xp[va])[:, 1], 1e-6, 1 - 1e-6)
                    score = -np.mean(y[va] * np.log(p) + (1 - y[va]) * np.log(1 - p))
                else:
                    score = np.mean(np.abs(m.predict(Xp[va]) - y[va]))
                if score < best_score:
                    best, best_score = a, score
            self.best_alpha_ = best
        else:
            self.best_alpha_ = self.alphas[len(self.alphas) // 2]
        self.model_ = self._make(self.best_alpha_).fit(Xp, y, sample_weight=w)
        return self

    def predict(self, X):
        Xp = self._prep(X, fit=False)
        if self.classifier:
            return self.model_.predict_proba(Xp)[:, 1]
        return self.model_.predict(Xp)

    def coefficients(self):
        return getattr(self.model_, "coef_", None)

    def contributions(self, X: np.ndarray) -> np.ndarray:
        """Per-feature contribution (coefficient x standardised value); missing indicators fold into their source feature."""
        Xp = self._prep(X, fit=False)
        coef = np.ravel(self.model_.coef_)
        contrib = Xp * coef[None, :]
        n = X.shape[1]
        out = contrib[:, :n].copy()
        if self.indicators and len(self.missing_cols_):
            out[:, self.missing_cols_] += contrib[:, n:]
        return out


class BoostModel:
    """LightGBM with early stopping on the last two training seasons, then a refit on everything."""

    def __init__(self, params: dict | None = None, classifier: bool = False, rounds: int = 3000):
        self.params = {"objective": "binary" if classifier else "regression_l2", "learning_rate": 0.02, "num_leaves": 15, "min_child_samples": 50, "feature_fraction": 0.4, "bagging_fraction": 0.8, "bagging_freq": 1,
                       "lambda_l2": 20.0, "max_bin": 63, "verbose": -1, "seed": SEED, "num_threads": 8, "min_gain_to_split": 0.0, "max_depth": -1} | (params or {})
        self.classifier = classifier
        self.rounds = rounds
        self.best_rounds_ = None

    def fit(self, X, y, w=None, seasons=None):
        import lightgbm as lgb
        if seasons is not None and len(np.unique(seasons)) >= 5:
            cut = np.sort(np.unique(seasons))[-2]
            tr, va = seasons < cut, seasons >= cut
            dtr = lgb.Dataset(X[tr], y[tr], weight=None if w is None else w[tr], free_raw_data=False)
            dva = lgb.Dataset(X[va], y[va], weight=None if w is None else w[va], reference=dtr, free_raw_data=False)
            booster = lgb.train(self.params, dtr, num_boost_round=self.rounds, valid_sets=[dva], callbacks=[lgb.early_stopping(150, verbose=False)])
            self.best_rounds_ = max(50, int(booster.best_iteration * 1.1))
        else:
            self.best_rounds_ = 400
        dall = lgb.Dataset(X, y, weight=w, free_raw_data=False)
        self.model_ = lgb.train(self.params, dall, num_boost_round=self.best_rounds_)
        return self

    def predict(self, X):
        return self.model_.predict(X)

    def contributions(self, X: np.ndarray) -> np.ndarray:
        """SHAP-style per-feature contributions (LightGBM pred_contrib, bias column dropped)."""
        return np.asarray(self.model_.predict(X, pred_contrib=True))[:, :-1]

    def importance(self, names: list[str], top: int = 40) -> list[tuple[str, float]]:
        gains = self.model_.feature_importance(importance_type="gain")
        order = np.argsort(gains)[::-1][:top]
        total = gains.sum() or 1.0
        return [(names[i], float(gains[i] / total)) for i in order]


class XGBModel:
    def __init__(self, params: dict | None = None, rounds: int = 3000):
        self.params = {"objective": "reg:squarederror", "eta": 0.02, "max_depth": 3, "min_child_weight": 30, "subsample": 0.8, "colsample_bytree": 0.4, "lambda": 20.0, "tree_method": "hist", "max_bin": 63, "seed": SEED, "nthread": 8} | (params or {})
        self.rounds = rounds

    def fit(self, X, y, w=None, seasons=None):
        import xgboost as xgb
        if seasons is not None and len(np.unique(seasons)) >= 5:
            cut = np.sort(np.unique(seasons))[-2]
            tr, va = seasons < cut, seasons >= cut
            dtr = xgb.DMatrix(X[tr], y[tr], weight=None if w is None else w[tr])
            dva = xgb.DMatrix(X[va], y[va], weight=None if w is None else w[va])
            booster = xgb.train(self.params, dtr, num_boost_round=self.rounds, evals=[(dva, "va")], early_stopping_rounds=150, verbose_eval=False)
            self.best_rounds_ = max(50, int(booster.best_iteration * 1.1))
        else:
            self.best_rounds_ = 400
        self.model_ = xgb.train(self.params, xgb.DMatrix(X, y, weight=w), num_boost_round=self.best_rounds_)
        return self

    def predict(self, X):
        import xgboost as xgb
        return self.model_.predict(xgb.DMatrix(X))


def make_learner(kind: str, **kw):
    if kind == "ridge":
        return RidgeModel(**kw)
    if kind == "logit":
        return RidgeModel(classifier=True, **kw)
    if kind == "lgbm":
        return BoostModel(**kw)
    if kind == "lgbm_clf":
        return BoostModel(classifier=True, **kw)
    if kind == "xgb":
        return XGBModel(**kw)
    raise ValueError(kind)
