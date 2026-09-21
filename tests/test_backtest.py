"""Backtest smoke tests.

The important one is ``test_training_data_never_reaches_into_the_test_fold``:
it uses a spy model to capture the exact training frame handed to every fold
and asserts the temporal boundary directly, rather than trusting the loop.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nflpred.backtest.metrics import ats_record, evaluate
from nflpred.backtest.walkforward import folds, run_backtest, summarize
from nflpred.config import Config
from nflpred.features.build import build_features
from nflpred.models.registry import BaseModel, HomeTeamBaseline, Predictions, VegasBaseline


@pytest.fixture(scope="module")
def cfg():
    return Config.load("configs/default.yaml")


@pytest.fixture(scope="module")
def df(cfg):
    return build_features(cfg)


class SpyModel(BaseModel):
    """Records the boundary of every training frame it is handed."""

    name = "spy"
    groups: list[str] = []

    def __init__(self):
        self.folds: list[dict] = []

    def fit(self, train, features):
        self.folds.append(
            {
                "n": len(train),
                "max_train_kickoff": train["kickoff"].max(),
                "train_game_ids": set(train["game_id"]),
                "all_played": bool(train["played"].all()),
            }
        )

    def predict(self, test, features):
        self.folds[-1]["min_test_kickoff"] = test["kickoff"].min()
        self.folds[-1]["test_game_ids"] = set(test["game_id"])
        return Predictions(p_home=np.full(len(test), 0.5))


def test_folds_are_chronological_and_start_where_configured(df, cfg):
    f = folds(df, 2017, 1)
    assert f == sorted(f), "folds must be in chronological order"
    assert f[0] == (2017, 1)
    assert all(s >= 2017 for s, _ in f)
    assert len(f) > 150


def test_training_data_never_reaches_into_the_test_fold(df, cfg):
    """Walk-forward guarantee: every training game kicked off before the fold."""
    spy = SpyModel()
    run_backtest(df, [spy], cfg, verbose=False)
    assert len(spy.folds) > 150

    for f in spy.folds:
        assert f["max_train_kickoff"] < f["min_test_kickoff"], (
            f"training data reached into the test fold: "
            f"train max {f['max_train_kickoff']} >= test min {f['min_test_kickoff']}"
        )
        assert not (f["train_game_ids"] & f["test_game_ids"]), "test game appeared in training"
        assert f["all_played"], "an unplayed game was used for training"


def test_training_set_grows_monotonically(df, cfg):
    spy = SpyModel()
    run_backtest(df, [spy], cfg, verbose=False)
    sizes = [f["n"] for f in spy.folds]
    assert all(b >= a for a, b in zip(sizes, sizes[1:])), "training set shrank between folds"
    assert sizes[-1] > sizes[0]


def test_backtest_smoke_produces_scored_predictions(df, cfg):
    models = [HomeTeamBaseline(), VegasBaseline()]
    preds = run_backtest(df, models, cfg, verbose=False)
    assert set(preds["model"]) == {"home_team", "vegas"}
    assert preds["game_id"].nunique() > 1500

    tab = summarize(preds, cfg, model_order=["home_team", "vegas"])
    assert len(tab) == 2
    home = tab[tab.model == "home_team"].iloc[0]
    vegas = tab[tab.model == "vegas"].iloc[0]

    # Sanity anchors: home-field is worth roughly 54-56%, and the market must
    # beat it comfortably. If either breaks, something upstream is wrong.
    assert 0.52 < home["accuracy"] < 0.58, home["accuracy"]
    assert 0.63 < vegas["accuracy"] < 0.70, vegas["accuracy"]
    assert vegas["accuracy"] > home["accuracy"]
    assert vegas["spread_mae"] < home["spread_mae"]


def test_vegas_against_its_own_line_is_a_coin_flip(df, cfg):
    """The market cannot beat itself: ATS% for the line vs the line is ~0.5.

    This is the strongest available check that the ATS accounting has no sign
    error - a flipped comparison would show up as a wild number here.
    """
    preds = run_backtest(df, [VegasBaseline()], cfg, verbose=False)
    rec = ats_record(preds["pred_spread"], preds["spread_actual"], preds["vegas_spread"])
    # Predicted spread equals the line, so every pick is a coin flip by
    # construction: ats_pct is whatever the tie-break rule yields, but the
    # decided count must equal the non-push count.
    assert rec["ats_n"] + rec["ats_pushes"] == int(
        np.isfinite(preds["spread_actual"]).sum()
    )
    assert 0.40 < rec["ats_pct"] < 0.60, rec


def test_evaluate_handles_missing_targets_gracefully():
    d = pd.DataFrame({
        "home_win": [1.0, 0.0, 1.0],
        "spread_actual": [7.0, -3.0, 1.0],
        "total_actual": [45.0, 38.0, 51.0],
        "vegas_spread": [3.0, -1.0, 2.0],
        "vegas_total": [44.0, 40.0, 49.0],
        "p_home": [0.6, 0.4, 0.55],
    })
    m = evaluate(d)
    assert "accuracy" in m and "spread_mae" not in m
    assert m["n_games"] == 3


def test_ties_are_excluded_from_classification_metrics():
    d = pd.DataFrame({
        "home_win": [1.0, 0.0, 0.5],       # third game is a tie
        "spread_actual": [7.0, -3.0, 0.0],
        "total_actual": [45.0, 38.0, 40.0],
        "vegas_spread": [3.0, -1.0, 0.0],
        "vegas_total": [44.0, 40.0, 41.0],
        "p_home": [0.9, 0.1, 0.5],
    })
    m = evaluate(d)
    assert m["n_games"] == 3
    assert m["n_cls"] == 2
    assert m["accuracy"] == pytest.approx(1.0)
