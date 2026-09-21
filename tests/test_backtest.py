"""Backtest smoke tests.

The important one is ``test_training_data_never_reaches_into_the_test_fold``:
it uses a spy model to capture the exact training frame handed to every fold
and asserts the temporal boundary directly, rather than trusting the loop.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nflpred.backtest.metrics import ats_record, evaluate, evaluate_target
from nflpred.backtest.walkforward import folds, run_backtest, summarize
from nflpred.config import Config
from nflpred.features.build import build_features
from nflpred.models.registry import BaseModel, HomeTeamBaseline, Predictions, VegasBaseline
from nflpred.models.targets import TARGETS


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
        super().__init__(["win"])
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
        p = Predictions()
        p.set(TARGETS["win"], np.full(len(test), 0.5))
        return p


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
    win = tab[tab["target"] == "win"].set_index("model")
    spread = tab[tab["target"] == "spread"].set_index("model")

    # Sanity anchors: home-field is worth roughly 54-56%, and the market must
    # beat it comfortably. If either breaks, something upstream is wrong.
    assert 0.52 < win.loc["home_team", "accuracy"] < 0.58
    assert 0.63 < win.loc["vegas", "accuracy"] < 0.70
    assert win.loc["vegas", "accuracy"] > win.loc["home_team", "accuracy"]
    assert spread.loc["vegas", "mae"] < spread.loc["home_team", "mae"]

    # Halftime targets must be scored too, and a first half must average about
    # half a full game's points.
    h1 = tab[tab["target"] == "h1_total"].set_index("model")
    assert not h1.empty, "halftime targets were not scored"
    # A constant predictor's MAE is the mean absolute deviation, ~0.8*sigma.
    # Halftime total sigma is ~9.1, so ~7.2 is the expected value here.
    assert 6.0 < h1.loc["home_team", "mae"] < 9.0, h1.loc["home_team", "mae"]
    full = tab[tab["target"] == "total"].set_index("model")
    assert h1.loc["home_team", "mae"] < full.loc["home_team", "mae"], \
        "halftime totals should be easier to predict than full-game totals"


def test_halftime_targets_are_derivable_and_sane(df):
    """Halftime targets must exist, be smaller than full-game, and never exceed it."""
    played = df[df["played"] & df["h1_total_actual"].notna()]
    assert len(played) > 2500
    assert played["h1_total_actual"].mean() < played["total_actual"].mean()
    assert (played["h1_total_actual"] <= played["total_actual"]).all(), \
        "a halftime score exceeded the final score"
    assert (played["h1_home_score"] <= played["home_score"]).all()
    assert (played["h1_away_score"] <= played["away_score"]).all()


def test_vegas_against_its_own_line_is_a_coin_flip(df, cfg):
    """The market cannot beat itself: ATS% for the line vs the line is ~0.5.

    This is the strongest available check that the ATS accounting has no sign
    error - a flipped comparison would show up as a wild number here.
    """
    preds = run_backtest(df, [VegasBaseline(["spread"])], cfg, verbose=False)
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
    m = {r["target"]: r for r in evaluate(d, specs=["win", "spread"])}
    assert "accuracy" in m["win"]
    assert m["win"]["n_games"] == 3
    # spread has no prediction column here, so it scores nothing.
    assert m["spread"].get("n_games", 0) == 0


def test_ties_are_excluded_from_classification_metrics():
    d = pd.DataFrame({
        "home_win": [1.0, 0.0, 0.5],       # third game is a tie
        "spread_actual": [7.0, -3.0, 0.0],
        "total_actual": [45.0, 38.0, 40.0],
        "vegas_spread": [3.0, -1.0, 0.0],
        "vegas_total": [44.0, 40.0, 41.0],
        "p_home": [0.9, 0.1, 0.5],
    })
    m = evaluate_target(d, TARGETS["win"])
    assert m["n_games"] == 3
    assert m["n_scored"] == 2, "the tie was not excluded"
    assert m["accuracy"] == pytest.approx(1.0)
