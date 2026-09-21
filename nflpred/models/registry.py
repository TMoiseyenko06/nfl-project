"""Models: three baselines and two learned models, over an arbitrary target set.

Every model exposes the same ``fit``/``predict`` pair and fits one estimator
per requested target, so the backtester can treat them uniformly and the
comparison table is always apples to apples.

Baselines are *fitted* wherever fitting is possible - the home-field edge, the
Elo-to-probability mapping and the line-to-probability mapping are all
estimated on the training fold rather than assumed. Otherwise the baselines
would be handicapped and the learned models would look better than they are.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from nflpred.models.targets import BINARY_FROM_MARGIN, TargetSpec, resolve


class Predictions:
    """Per-target predictions, keyed by the target's prediction column."""

    def __init__(self, values: dict[str, np.ndarray] | None = None):
        self.values: dict[str, np.ndarray] = values or {}

    def set(self, spec: TargetSpec, arr) -> None:
        self.values[spec.pred_column] = np.asarray(arr, dtype=float)

    def to_frame(self, index) -> pd.DataFrame:
        return pd.DataFrame(self.values, index=index)


def _usable(train: pd.DataFrame, spec: TargetSpec) -> pd.DataFrame:
    """Training rows with a usable label for this target.

    Binary targets drop ties (0.5): a tie has no winner and cannot train a
    binary classifier. Ties are ~0.3% of full games but ~7% of first halves,
    so this matters more for the halftime targets than the full-game ones.
    """
    d = train[train[spec.column].notna()]
    if spec.is_binary:
        d = d[d[spec.column].isin([0.0, 1.0])]
    return d


class BaseModel:
    name = "base"
    groups: list[str] = []

    def __init__(self, targets: list[str] | None = None):
        self.specs = resolve(targets)

    def fit(self, train: pd.DataFrame, features: list[str]) -> None:
        raise NotImplementedError

    def predict(self, test: pd.DataFrame, features: list[str]) -> Predictions:
        raise NotImplementedError

    def feature_importance(self, target: str = "win") -> pd.Series | None:
        return None


# --------------------------------------------------------------------------
# a. Always pick the home team
# --------------------------------------------------------------------------
class HomeTeamBaseline(BaseModel):
    """Constant prediction: the training-fold base rate for each target."""

    name = "home_team"

    def fit(self, train: pd.DataFrame, features: list[str]) -> None:
        self.const_: dict[str, float] = {}
        for spec in self.specs:
            d = _usable(train, spec)
            self.const_[spec.name] = float(d[spec.column].mean()) if len(d) else 0.0

    def predict(self, test: pd.DataFrame, features: list[str]) -> Predictions:
        p = Predictions()
        for spec in self.specs:
            p.set(spec, np.full(len(test), self.const_[spec.name]))
        return p


# --------------------------------------------------------------------------
# b. Elo
# --------------------------------------------------------------------------
class EloBaseline(BaseModel):
    """Elo ratings, with the rating->target map fitted on the training fold
    rather than using the textbook constants."""

    name = "elo"

    def fit(self, train: pd.DataFrame, features: list[str]) -> None:
        self.models_: dict[str, object] = {}
        self.fallback_: dict[str, float] = {}
        for spec in self.specs:
            d = _usable(train, spec).dropna(subset=["elo_diff"])
            self.fallback_[spec.name] = float(d[spec.column].mean()) if len(d) else 0.0
            if len(d) < 30:
                self.models_[spec.name] = None
                continue
            X, y = d[["elo_diff"]], d[spec.column]
            if spec.is_binary:
                m = LogisticRegression(max_iter=1000).fit(X, y.astype(int))
            else:
                m = LinearRegression().fit(X, y)
            self.models_[spec.name] = m

    def predict(self, test: pd.DataFrame, features: list[str]) -> Predictions:
        X = test[["elo_diff"]].astype(float).fillna(0.0)
        p = Predictions()
        for spec in self.specs:
            m = self.models_[spec.name]
            if m is None:
                p.set(spec, np.full(len(test), self.fallback_[spec.name]))
            elif spec.is_binary:
                p.set(spec, m.predict_proba(X)[:, 1])
            else:
                p.set(spec, m.predict(X))
        return p


# --------------------------------------------------------------------------
# c. The Vegas line - the bar that actually matters
# --------------------------------------------------------------------------
class VegasBaseline(BaseModel):
    """The market line as the prediction.

    Where a real market exists for the target (full-game spread and total) the
    line is used directly, unmodified - the pure benchmark. Win probability is
    a fitted logistic map of the spread, because a line is not a probability.

    Halftime markets are NOT in nflverse. For those targets this fits a map
    from the full-game line to the halftime outcome, which is the fairest
    available "what does the market imply about the first half" benchmark. It
    is a DERIVED benchmark, not a real halftime line, and is labelled as such
    wherever it is reported.
    """

    name = "vegas"
    MARKET_FEATURES = ["vegas_spread", "vegas_total"]

    def fit(self, train: pd.DataFrame, features: list[str]) -> None:
        self.models_: dict[str, object] = {}
        self.fallback_: dict[str, float] = {}
        self.derived_: dict[str, bool] = {}
        for spec in self.specs:
            d = _usable(train, spec).dropna(subset=self.MARKET_FEATURES)
            self.fallback_[spec.name] = float(d[spec.column].mean()) if len(d) else 0.0
            # Direct market prediction: the line IS the forecast.
            if spec.name in ("spread", "total"):
                self.models_[spec.name] = None
                self.derived_[spec.name] = False
                continue
            self.derived_[spec.name] = spec.market_column is None
            if len(d) < 30:
                self.models_[spec.name] = None
                continue
            X = d[self.MARKET_FEATURES] if self.derived_[spec.name] else d[["vegas_spread"]]
            if spec.is_binary:
                m = LogisticRegression(max_iter=1000).fit(X, d[spec.column].astype(int))
            else:
                m = LinearRegression().fit(X, d[spec.column])
            self.models_[spec.name] = m

    def predict(self, test: pd.DataFrame, features: list[str]) -> Predictions:
        p = Predictions()
        vs = test["vegas_spread"].astype(float)
        vt = test["vegas_total"].astype(float)
        for spec in self.specs:
            if spec.name == "spread":
                p.set(spec, vs.fillna(self.fallback_["spread"]).to_numpy())
                continue
            if spec.name == "total":
                p.set(spec, vt.fillna(self.fallback_["total"]).to_numpy())
                continue
            m = self.models_[spec.name]
            if m is None:
                p.set(spec, np.full(len(test), self.fallback_[spec.name]))
                continue
            cols = self.MARKET_FEATURES if self.derived_[spec.name] else ["vegas_spread"]
            X = test[cols].astype(float)
            X = X.fillna({"vegas_spread": self.fallback_.get("spread", 0.0),
                          "vegas_total": self.fallback_.get("total", 44.0)})
            p.set(spec, m.predict_proba(X)[:, 1] if spec.is_binary else m.predict(X))
        return p


# --------------------------------------------------------------------------
# d. Regularised linear models
# --------------------------------------------------------------------------
class MarginDerivedProbability:
    """Turn a predicted margin into a win probability, coherently.

    Fits a single-parameter logistic P = sigmoid(b * margin) with NO intercept,
    on the training fold. Because there is no intercept the curve crosses 0.5
    exactly at margin 0, so "probability favours the home team" and "predicted
    margin favours the home team" can never disagree. Fitting b (rather than
    assuming a normal CDF with the residual sigma) lets the mapping be
    calibrated to the data.
    """

    def fit_margin_map(self, margin_train: np.ndarray, won: np.ndarray):
        m = np.isfinite(margin_train) & np.isfinite(won)
        margin_train, won = margin_train[m], won[m]
        if len(np.unique(won)) < 2 or len(won) < 30:
            return None
        clf = LogisticRegression(fit_intercept=False, max_iter=1000)
        clf.fit(margin_train.reshape(-1, 1), won.astype(int))
        return clf

    @staticmethod
    def apply_margin_map(clf, margin: np.ndarray) -> np.ndarray:
        return clf.predict_proba(np.asarray(margin, dtype=float).reshape(-1, 1))[:, 1]


def _linear_pipeline(estimator):
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", estimator),
    ])


class LinearModel(BaseModel, MarginDerivedProbability):
    """L2 logistic for binary targets, ridge for regression targets."""

    name = "linear"

    def __init__(self, C: float = 0.08, alpha: float = 30.0, groups: list[str] | None = None,
                 max_iter: int = 2000, name: str | None = None, targets: list[str] | None = None,
                 derive_binary_from_margin: bool = False):
        super().__init__(targets)
        self.derive_binary_from_margin = derive_binary_from_margin
        self.C, self.alpha, self.max_iter = C, alpha, max_iter
        self.groups = groups or ["context", "form", "adjusted", "matchup", "elo"]
        if name:
            self.name = name

    def fit(self, train: pd.DataFrame, features: list[str]) -> None:
        self.features_ = features
        self.models_: dict[str, object] = {}
        self.margin_maps_: dict[str, object] = {}
        names = {s.name for s in self.specs}
        for spec in self.specs:
            if self._derives(spec, names):
                continue
            d = _usable(train, spec)
            est = (LogisticRegression(C=self.C, max_iter=self.max_iter)
                   if spec.is_binary else Ridge(alpha=self.alpha))
            pipe = _linear_pipeline(est)
            y = d[spec.column].astype(int) if spec.is_binary else d[spec.column]
            pipe.fit(d[features], y)
            self.models_[spec.name] = pipe
        self._fit_margin_maps(train, features, names)

    def _derives(self, spec, names) -> bool:
        return (
            self.derive_binary_from_margin
            and spec.is_binary
            and BINARY_FROM_MARGIN.get(spec.name) in names
        )

    def _fit_margin_maps(self, train, features, names) -> None:
        for spec in self.specs:
            if not self._derives(spec, names):
                continue
            margin_name = BINARY_FROM_MARGIN[spec.name]
            d = _usable(train, spec)
            pred_margin = self.models_[margin_name].predict(d[features])
            self.margin_maps_[spec.name] = self.fit_margin_map(
                np.asarray(pred_margin, dtype=float), d[spec.column].to_numpy(dtype=float)
            )

    def predict(self, test: pd.DataFrame, features: list[str]) -> Predictions:
        X = test[features]
        p = Predictions()
        for spec in self.specs:
            if spec.name in self.margin_maps_:
                margin = self.models_[BINARY_FROM_MARGIN[spec.name]].predict(X)
                clf = self.margin_maps_[spec.name]
                p.set(spec, self.apply_margin_map(clf, margin) if clf is not None
                      else (np.asarray(margin) > 0).astype(float))
                continue
            m = self.models_[spec.name]
            p.set(spec, m.predict_proba(X)[:, 1] if spec.is_binary else m.predict(X))
        return p

    def feature_importance(self, target: str = "win") -> pd.Series | None:
        m = self.models_.get(target)
        if m is None:
            return None
        est = m.named_steps["model"]
        coef = est.coef_[0] if getattr(est.coef_, "ndim", 1) > 1 else est.coef_
        return pd.Series(np.abs(coef), index=self.features_).sort_values(ascending=False)


# --------------------------------------------------------------------------
# e. Gradient-boosted trees
# --------------------------------------------------------------------------
class LightGBMModel(BaseModel, MarginDerivedProbability):
    """LightGBM, deliberately shallow. ~3k rows punishes anything bigger."""

    name = "lightgbm"

    def __init__(self, params: dict | None = None, seed: int = 1729,
                 groups: list[str] | None = None, name: str | None = None,
                 targets: list[str] | None = None,
                 derive_binary_from_margin: bool = False):
        super().__init__(targets)
        self.derive_binary_from_margin = derive_binary_from_margin
        self.params = dict(params or {})
        self.seed = seed
        self.groups = groups or ["context", "form", "adjusted", "matchup", "elo"]
        if name:
            self.name = name

    def _make(self, spec: TargetSpec):
        import lightgbm as lgb

        kw = dict(self.params)
        kw.update(random_state=self.seed, n_jobs=1, deterministic=True, force_row_wise=True)
        if spec.is_binary:
            return lgb.LGBMClassifier(objective="binary", **kw)
        return lgb.LGBMRegressor(objective="regression", **kw)

    def fit(self, train: pd.DataFrame, features: list[str]) -> None:
        self.features_ = features
        self.models_: dict[str, object] = {}
        self.margin_maps_: dict[str, object] = {}
        names = {s.name for s in self.specs}
        for spec in self.specs:
            if self._derives(spec, names):
                continue
            d = _usable(train, spec)
            m = self._make(spec)
            y = d[spec.column].astype(int) if spec.is_binary else d[spec.column]
            m.fit(d[features], y)
            self.models_[spec.name] = m
        for spec in self.specs:
            if not self._derives(spec, names):
                continue
            d = _usable(train, spec)
            pred_margin = self.models_[BINARY_FROM_MARGIN[spec.name]].predict(d[features])
            self.margin_maps_[spec.name] = self.fit_margin_map(
                np.asarray(pred_margin, dtype=float), d[spec.column].to_numpy(dtype=float)
            )

    def _derives(self, spec, names) -> bool:
        return (
            self.derive_binary_from_margin
            and spec.is_binary
            and BINARY_FROM_MARGIN.get(spec.name) in names
        )

    def predict(self, test: pd.DataFrame, features: list[str]) -> Predictions:
        X = test[features]
        p = Predictions()
        for spec in self.specs:
            if spec.name in self.margin_maps_:
                margin = self.models_[BINARY_FROM_MARGIN[spec.name]].predict(X)
                clf = self.margin_maps_[spec.name]
                p.set(spec, self.apply_margin_map(clf, margin) if clf is not None
                      else (np.asarray(margin) > 0).astype(float))
                continue
            m = self.models_[spec.name]
            p.set(spec, m.predict_proba(X)[:, 1] if spec.is_binary else m.predict(X))
        return p

    def feature_importance(self, target: str = "win") -> pd.Series | None:
        m = self.models_.get(target)
        if m is None:
            return None
        imp = m.booster_.feature_importance(importance_type="gain")
        return pd.Series(imp, index=self.features_).sort_values(ascending=False)


def default_phase1_models(cfg, targets: list[str] | None = None) -> list[BaseModel]:
    """The standard slate, in the order results are reported."""
    seed = cfg.seed
    core = ["context", "form", "adjusted", "matchup", "elo"]
    lin = dict(C=cfg.get("models.logistic.C"), alpha=cfg.get("models.ridge.alpha"),
               max_iter=cfg.get("models.logistic.max_iter"))
    gbm = dict(params=cfg.get("models.lightgbm"), seed=seed)
    return [
        HomeTeamBaseline(targets),
        EloBaseline(targets),
        VegasBaseline(targets),
        LinearModel(**lin, groups=core, targets=targets),
        LightGBMModel(**gbm, groups=core, targets=targets),
        # Same learners, but allowed to see the market line. The gap between
        # these and the pair above is what the market knows that we do not.
        LinearModel(**lin, groups=core + ["vegas"], name="linear_plus_vegas", targets=targets),
        LightGBMModel(**gbm, groups=core + ["vegas"], name="lightgbm_plus_vegas", targets=targets),
        # Win probability derived from the predicted margin rather than fit as a
        # separate classifier, so the moneyline pick and the spread pick can
        # never contradict each other. This is the variant meant for display.
        LinearModel(**lin, groups=core, name="linear_coherent", targets=targets,
                    derive_binary_from_margin=True),
        LightGBMModel(**gbm, groups=core, name="lightgbm_coherent", targets=targets,
                      derive_binary_from_margin=True),
    ]
