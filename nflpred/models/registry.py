"""Phase 1 models: three trivial baselines and two learned models.

Every model exposes the same ``fit``/``predict`` pair and returns the same three
targets, so the backtester can treat them uniformly and the comparison table is
always apples to apples.

Baselines are *fitted* wherever fitting is possible - the home-field edge, the
Elo-to-probability mapping and the line-to-probability mapping are all estimated
on the training fold rather than assumed. Otherwise the baselines would be
handicapped and the learned models would look better than they are.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

WIN, SPREAD, TOTAL = "win", "spread", "total"


@dataclass
class Predictions:
    p_home: np.ndarray | None = None
    pred_spread: np.ndarray | None = None
    pred_total: np.ndarray | None = None

    def to_frame(self, index) -> pd.DataFrame:
        d = {}
        if self.p_home is not None:
            d["p_home"] = self.p_home
        if self.pred_spread is not None:
            d["pred_spread"] = self.pred_spread
        if self.pred_total is not None:
            d["pred_total"] = self.pred_total
        return pd.DataFrame(d, index=index)


class BaseModel:
    name = "base"
    #: feature groups this model consumes; empty means it needs no feature matrix
    groups: list[str] = []

    def fit(self, train: pd.DataFrame, features: list[str]) -> None:
        raise NotImplementedError

    def predict(self, test: pd.DataFrame, features: list[str]) -> Predictions:
        raise NotImplementedError

    def feature_importance(self) -> pd.Series | None:
        return None


def _binary(train: pd.DataFrame) -> pd.DataFrame:
    """Drop ties, which have no winner and cannot train a binary classifier."""
    return train[train["home_win"].isin([0.0, 1.0])]


# --------------------------------------------------------------------------
# a. Always pick the home team
# --------------------------------------------------------------------------
class HomeTeamBaseline(BaseModel):
    name = "home_team"

    def fit(self, train: pd.DataFrame, features: list[str]) -> None:
        b = _binary(train)
        self.p_ = float(b["home_win"].mean()) if len(b) else 0.5
        self.spread_ = float(train["spread_actual"].mean())
        self.total_ = float(train["total_actual"].mean())

    def predict(self, test: pd.DataFrame, features: list[str]) -> Predictions:
        n = len(test)
        return Predictions(
            p_home=np.full(n, self.p_),
            pred_spread=np.full(n, self.spread_),
            pred_total=np.full(n, self.total_),
        )


# --------------------------------------------------------------------------
# b. Elo
# --------------------------------------------------------------------------
class EloBaseline(BaseModel):
    """Elo ratings, with the rating->probability and rating->spread maps fitted
    on the training fold rather than using the textbook constants."""

    name = "elo"

    def fit(self, train: pd.DataFrame, features: list[str]) -> None:
        b = _binary(train).dropna(subset=["elo_diff"])
        self.clf_ = LogisticRegression(max_iter=1000)
        self.clf_.fit(b[["elo_diff"]], b["home_win"].astype(int))
        r = train.dropna(subset=["elo_diff", "spread_actual"])
        self.reg_ = LinearRegression().fit(r[["elo_diff"]], r["spread_actual"])
        self.total_ = float(train["total_actual"].mean())

    def predict(self, test: pd.DataFrame, features: list[str]) -> Predictions:
        x = test[["elo_diff"]].astype(float)
        x = x.fillna(0.0)
        return Predictions(
            p_home=self.clf_.predict_proba(x)[:, 1],
            pred_spread=self.reg_.predict(x),
            pred_total=np.full(len(test), self.total_),
        )


# --------------------------------------------------------------------------
# c. The Vegas line - the bar that actually matters
# --------------------------------------------------------------------------
class VegasBaseline(BaseModel):
    """The market line used directly as the prediction.

    NOTE: the line in nflverse's schedule file is the CLOSING line. It is fixed
    at kickoff and so is legitimately pre-kickoff information, but it is also
    the single hardest benchmark in sports prediction. See README.
    """

    name = "vegas"

    def fit(self, train: pd.DataFrame, features: list[str]) -> None:
        b = _binary(train).dropna(subset=["vegas_spread"])
        self.clf_ = LogisticRegression(max_iter=1000)
        self.clf_.fit(b[["vegas_spread"]], b["home_win"].astype(int))
        self.fallback_spread_ = float(train["spread_actual"].mean())
        self.fallback_total_ = float(train["total_actual"].mean())

    def predict(self, test: pd.DataFrame, features: list[str]) -> Predictions:
        vs = test["vegas_spread"].astype(float)
        x = vs.fillna(self.fallback_spread_).to_frame()
        return Predictions(
            p_home=self.clf_.predict_proba(x)[:, 1],
            pred_spread=vs.fillna(self.fallback_spread_).to_numpy(),
            pred_total=test["vegas_total"].astype(float).fillna(self.fallback_total_).to_numpy(),
        )


# --------------------------------------------------------------------------
# d. Regularised linear models on Tier 1 features
# --------------------------------------------------------------------------
def _linear_pipeline(estimator):
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", estimator),
        ]
    )


class LinearModel(BaseModel):
    """L2 logistic for win probability, ridge for spread and total."""

    name = "linear"

    def __init__(self, C: float = 0.08, alpha: float = 30.0, groups: list[str] | None = None,
                 max_iter: int = 2000, name: str | None = None):
        self.C, self.alpha, self.max_iter = C, alpha, max_iter
        self.groups = groups or ["context", "form", "adjusted", "matchup", "elo"]
        if name:
            self.name = name

    def fit(self, train: pd.DataFrame, features: list[str]) -> None:
        self.features_ = features
        b = _binary(train)
        self.clf_ = _linear_pipeline(LogisticRegression(C=self.C, max_iter=self.max_iter))
        self.clf_.fit(b[features], b["home_win"].astype(int))
        self.reg_spread_ = _linear_pipeline(Ridge(alpha=self.alpha))
        self.reg_spread_.fit(train[features], train["spread_actual"])
        self.reg_total_ = _linear_pipeline(Ridge(alpha=self.alpha))
        self.reg_total_.fit(train[features], train["total_actual"])

    def predict(self, test: pd.DataFrame, features: list[str]) -> Predictions:
        X = test[features]
        return Predictions(
            p_home=self.clf_.predict_proba(X)[:, 1],
            pred_spread=self.reg_spread_.predict(X),
            pred_total=self.reg_total_.predict(X),
        )

    def feature_importance(self) -> pd.Series | None:
        coef = self.clf_.named_steps["model"].coef_[0]
        return pd.Series(np.abs(coef), index=self.features_).sort_values(ascending=False)


# --------------------------------------------------------------------------
# e. Gradient-boosted trees
# --------------------------------------------------------------------------
class LightGBMModel(BaseModel):
    """LightGBM, deliberately shallow. ~3k rows punishes anything bigger."""

    name = "lightgbm"

    def __init__(self, params: dict | None = None, seed: int = 1729,
                 groups: list[str] | None = None, name: str | None = None):
        self.params = dict(params or {})
        self.seed = seed
        self.groups = groups or ["context", "form", "adjusted", "matchup", "elo"]
        if name:
            self.name = name

    def _make(self, objective: str):
        import lightgbm as lgb

        kw = dict(self.params)
        kw.update(
            random_state=self.seed,
            n_jobs=1,             # single-threaded keeps runs bit-reproducible
            deterministic=True,
            force_row_wise=True,
        )
        if objective == "binary":
            return lgb.LGBMClassifier(objective="binary", **kw)
        return lgb.LGBMRegressor(objective=objective, **kw)

    def fit(self, train: pd.DataFrame, features: list[str]) -> None:
        self.features_ = features
        b = _binary(train)
        self.clf_ = self._make("binary")
        self.clf_.fit(b[features], b["home_win"].astype(int))
        self.reg_spread_ = self._make("regression")
        self.reg_spread_.fit(train[features], train["spread_actual"])
        self.reg_total_ = self._make("regression")
        self.reg_total_.fit(train[features], train["total_actual"])

    def predict(self, test: pd.DataFrame, features: list[str]) -> Predictions:
        X = test[features]
        return Predictions(
            p_home=self.clf_.predict_proba(X)[:, 1],
            pred_spread=self.reg_spread_.predict(X),
            pred_total=self.reg_total_.predict(X),
        )

    def feature_importance(self) -> pd.Series | None:
        imp = self.clf_.booster_.feature_importance(importance_type="gain")
        return pd.Series(imp, index=self.features_).sort_values(ascending=False)


def default_phase1_models(cfg) -> list[BaseModel]:
    """The Phase 1 slate, in the order the README reports them."""
    seed = cfg.seed
    core = ["context", "form", "adjusted", "matchup", "elo"]
    return [
        HomeTeamBaseline(),
        EloBaseline(),
        VegasBaseline(),
        LinearModel(
            C=cfg.get("models.logistic.C"),
            alpha=cfg.get("models.ridge.alpha"),
            max_iter=cfg.get("models.logistic.max_iter"),
            groups=core,
        ),
        LightGBMModel(params=cfg.get("models.lightgbm"), seed=seed, groups=core),
        # Same two learners, but allowed to see the market line. The gap between
        # these and the pair above is what the market knows that we do not.
        LinearModel(
            C=cfg.get("models.logistic.C"),
            alpha=cfg.get("models.ridge.alpha"),
            max_iter=cfg.get("models.logistic.max_iter"),
            groups=core + ["vegas"],
            name="linear_plus_vegas",
        ),
        LightGBMModel(
            params=cfg.get("models.lightgbm"), seed=seed,
            groups=core + ["vegas"], name="lightgbm_plus_vegas",
        ),
    ]
