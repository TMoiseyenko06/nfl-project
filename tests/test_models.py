"""Model-level invariants.

The coherence tests matter because the failure they guard against is silent:
a moneyline pick and a spread pick that contradict each other still produce
plausible-looking output.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nflpred.config import Config
from nflpred.features.build import build_features, select_features
from nflpred.models.registry import (
    EloBaseline,
    HomeTeamBaseline,
    LightGBMModel,
    LinearModel,
    VegasBaseline,
)
from nflpred.models.targets import BINARY_FROM_MARGIN, DEFAULT_TARGETS, TARGETS, resolve


@pytest.fixture(scope="module")
def cfg():
    return Config.load("configs/default.yaml")


@pytest.fixture(scope="module")
def split(cfg):
    df = build_features(cfg)
    train = df[df["played"] & (df["season"] < 2024)]
    test = df[df["played"] & (df["season"] >= 2024)]
    return df, train, test


CORE = ["context", "form", "adjusted", "matchup", "elo"]


def test_all_models_produce_every_target(cfg, split):
    df, train, test = split
    expected = {t.pred_column for t in resolve(DEFAULT_TARGETS)}
    models = [
        HomeTeamBaseline(), EloBaseline(), VegasBaseline(),
        LinearModel(groups=CORE),
    ]
    for m in models:
        feats = select_features(df, m.groups) if m.groups else []
        m.fit(train, feats)
        p = m.predict(test, feats)
        assert set(p.values) == expected, f"{m.name} did not produce every target"
        for col, arr in p.values.items():
            assert len(arr) == len(test)
            assert np.isfinite(arr).all(), f"{m.name}/{col} produced non-finite values"


def test_probabilities_are_in_range(cfg, split):
    df, train, test = split
    for m in [EloBaseline(), VegasBaseline(), LinearModel(groups=CORE)]:
        feats = select_features(df, m.groups) if m.groups else []
        m.fit(train, feats)
        p = m.predict(test, feats)
        for spec in resolve(DEFAULT_TARGETS):
            if not spec.is_binary:
                continue
            v = p.values[spec.pred_column]
            assert (v >= 0).all() and (v <= 1).all(), f"{m.name}/{spec.name} out of [0,1]"


@pytest.mark.parametrize("factory", [
    lambda: LinearModel(groups=CORE, derive_binary_from_margin=True),
    lambda: LightGBMModel(params={"n_estimators": 60, "num_leaves": 7, "verbosity": -1},
                          groups=CORE, derive_binary_from_margin=True),
])
def test_margin_derived_probability_never_contradicts_the_margin(cfg, split, factory):
    """P(home) > 0.5 must hold exactly when the predicted margin favours home."""
    df, train, test = split
    m = factory()
    feats = select_features(df, m.groups)
    m.fit(train, feats)
    p = m.predict(test, feats)

    for binary, margin in BINARY_FROM_MARGIN.items():
        pb = p.values[TARGETS[binary].pred_column]
        pm = p.values[TARGETS[margin].pred_column]
        disagree = (pb > 0.5) != (pm > 0)
        assert not disagree.any(), (
            f"{m.name}: {binary} contradicts {margin} on {int(disagree.sum())} games"
        )


def test_separate_classifier_is_allowed_to_disagree(cfg, split):
    """Control: without the flag the two are independent, so they can disagree.

    This is what the coherent variant exists to fix; if this ever stops
    disagreeing the coherence test above has become vacuous.
    """
    df, train, test = split
    m = LinearModel(groups=CORE, derive_binary_from_margin=False)
    feats = select_features(df, m.groups)
    m.fit(train, feats)
    p = m.predict(test, feats)
    disagree = (p.values["p_home"] > 0.5) != (p.values["pred_spread"] > 0)
    assert disagree.any(), "expected the independent classifier to disagree somewhere"


def test_vegas_uses_the_raw_line_for_spread_and_total(cfg, split):
    """The market benchmark must be the line itself, not a model of the line."""
    df, train, test = split
    m = VegasBaseline()
    m.fit(train, [])
    p = m.predict(test, [])
    have = test["vegas_spread"].notna().to_numpy()
    assert np.allclose(p.values["pred_spread"][have], test["vegas_spread"].to_numpy()[have])
    have_t = test["vegas_total"].notna().to_numpy()
    assert np.allclose(p.values["pred_total"][have_t], test["vegas_total"].to_numpy()[have_t])


def test_halftime_predictions_are_smaller_than_full_game(cfg, split):
    df, train, test = split
    m = LinearModel(groups=CORE)
    feats = select_features(df, m.groups)
    m.fit(train, feats)
    p = m.predict(test, feats)
    assert p.values["pred_h1_total"].mean() < p.values["pred_total"].mean()
    assert abs(p.values["pred_h1_spread"]).mean() < abs(p.values["pred_spread"]).mean()


def test_binary_targets_drop_ties_when_training(cfg, split):
    """Ties are ~7% of first halves; a classifier cannot train on 0.5."""
    from nflpred.models.registry import _usable

    df, train, _ = split
    d = _usable(train, TARGETS["h1_win"])
    assert set(d["h1_home_win"].unique()) <= {0.0, 1.0}
    assert len(d) < len(train[train["h1_home_win"].notna()]), "no ties were dropped"
