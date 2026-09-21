"""Tests for the append-only prediction log and config loading."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nflpred.config import Config
from nflpred.predict.log import (
    append_predictions,
    logged_predictions_before_kickoff,
    verify_log,
)


@pytest.fixture
def sample():
    return pd.DataFrame({
        "model": ["lightgbm", "lightgbm"],
        "game_id": ["2026_03_LAC_BUF", "2026_03_ATL_GB"],
        "season": [2026, 2026],
        "week": [3, 3],
        "kickoff": ["2026-09-27 13:00:00", "2026-09-24 20:15:00"],
        "home_team": ["BUF", "GB"],
        "away_team": ["LAC", "ATL"],
        "p_home": [0.71, 0.63],
        "pred_spread": [5.2, 3.4],
        "pred_total": [47.1, 44.8],
        "vegas_spread": [7.0, 7.0],
        "vegas_total": [50.5, 44.5],
    })


def test_config_dotted_access_and_defaults():
    c = Config.load("configs/default.yaml")
    assert isinstance(c.get("models.lightgbm.num_leaves"), int)
    assert c.get("does.not.exist", "fallback") == "fallback"
    with pytest.raises(KeyError):
        c.get("does.not.exist")


def test_config_seasons_include_warmup_for_pbp():
    c = Config.load("configs/default.yaml")
    assert min(c.pbp_seasons) < min(c.seasons), "pbp must be warmed up before training starts"


def test_append_is_idempotent_and_never_overwrites(tmp_path, sample):
    p = tmp_path / "log.csv"
    first = append_predictions(sample, p, vintage="v1")
    assert len(first) == 2

    # Same games, different predictions, different vintage: must not overwrite.
    changed = sample.copy()
    changed["p_home"] = [0.01, 0.99]
    second = append_predictions(changed, p, vintage="v2")
    assert len(second) == 0, "existing (model, game) rows were overwritten"

    on_disk = pd.read_csv(p)
    assert len(on_disk) == 2
    assert on_disk["p_home"].tolist() == [0.71, 0.63], "original predictions were mutated"


def test_append_adds_genuinely_new_rows(tmp_path, sample):
    p = tmp_path / "log.csv"
    append_predictions(sample, p, vintage="v1")
    other = sample.copy()
    other["model"] = "elo"
    written = append_predictions(other, p, vintage="v1")
    assert len(written) == 2
    assert len(pd.read_csv(p)) == 4


def test_tampering_is_detected(tmp_path, sample):
    p = tmp_path / "log.csv"
    append_predictions(sample, p, vintage="v1")
    assert verify_log(p).empty

    df = pd.read_csv(p)
    df.loc[0, "p_home"] = 0.999
    df.to_csv(p, index=False)
    bad = verify_log(p)
    assert len(bad) == 1, "an edited prediction went undetected"


def test_predictions_logged_after_kickoff_are_flagged(tmp_path, sample):
    p = tmp_path / "log.csv"
    append_predictions(sample, p, vintage="v1")
    df = logged_predictions_before_kickoff(p)
    # These kickoffs are in 2026; rows logged now are before them.
    assert "logged_before_kickoff" in df.columns

    # Forge a row logged well after its kickoff.
    raw = pd.read_csv(p)
    raw.loc[0, "kickoff"] = "2020-01-01 13:00:00"
    raw.to_csv(p, index=False)
    df2 = logged_predictions_before_kickoff(p)
    assert not df2.loc[0, "logged_before_kickoff"], "a retrodiction was accepted as a prediction"


def test_log_survives_missing_optional_columns(tmp_path):
    minimal = pd.DataFrame({
        "model": ["elo"], "game_id": ["G1"], "season": [2026], "week": [3],
        "kickoff": ["2026-09-27 13:00:00"], "home_team": ["BUF"], "away_team": ["LAC"],
        "p_home": [0.6],
    })
    p = tmp_path / "log.csv"
    written = append_predictions(minimal, p, vintage="v1")
    assert len(written) == 1
    assert pd.read_csv(p)["pred_total"].isna().all()


# --- upcoming-week selection ------------------------------------------------
def _schedule():
    """Week 2 is partly played; week 3 is entirely in the future."""
    return pd.DataFrame({
        "game_id": ["w2_thu", "w2_sun", "w2_mon", "w3_thu", "w3_sun"],
        "season": [2026] * 5,
        "week": [2, 2, 2, 3, 3],
        "kickoff": pd.to_datetime([
            "2026-09-17 20:15", "2026-09-20 13:00", "2026-09-21 20:15",
            "2026-09-24 20:15", "2026-09-27 13:00",
        ]),
        "played": [True, True, False, False, False],
    })


def test_upcoming_games_skips_games_already_kicked_off():
    """Regression: predicting a week that is already underway logged
    retrodictions for games that had finished."""
    from nflpred.predict.week import upcoming_games

    now = pd.Timestamp("2026-09-21 09:00")      # Monday morning
    season, week, slate = upcoming_games(_schedule(), now=now)
    assert (season, week) == (2026, 2)
    assert slate["game_id"].tolist() == ["w2_mon"], "a completed game was selected"
    assert (slate["kickoff"] > now).all()


def test_upcoming_games_rolls_to_next_week_once_current_is_done():
    from nflpred.predict.week import upcoming_games

    now = pd.Timestamp("2026-09-22 09:00")      # after the Monday nighter
    season, week, slate = upcoming_games(_schedule(), now=now)
    assert (season, week) == (2026, 3)
    assert slate["game_id"].tolist() == ["w3_thu", "w3_sun"]


def test_upcoming_games_never_returns_a_played_game():
    from nflpred.predict.week import upcoming_games

    for hour in ["2026-09-16 09:00", "2026-09-19 09:00", "2026-09-21 09:00"]:
        _, _, slate = upcoming_games(_schedule(), now=pd.Timestamp(hour))
        assert not slate["played"].any(), f"returned a completed game at {hour}"


def test_upcoming_games_raises_when_season_is_over():
    from nflpred.predict.week import upcoming_games

    with pytest.raises(RuntimeError, match="no upcoming games"):
        upcoming_games(_schedule(), now=pd.Timestamp("2027-01-01"))


def test_adding_a_prediction_column_does_not_invalidate_old_rows(tmp_path, sample):
    """Regression: the hash is computed over a field list that grows as targets
    are added. Without recording which fields a row was hashed with, adding a
    column retroactively flags every existing row as tampered - a false alarm
    that trains you to ignore the check."""
    import nflpred.predict.log as logmod

    p = tmp_path / "log.csv"
    append_predictions(sample, p, vintage="v1")
    assert verify_log(p).empty

    original = list(logmod.PRED_FIELDS)
    try:
        # Simulate adding a new target's prediction column.
        logmod.PRED_FIELDS = original + ["pred_new_target"]
        flagged = verify_log(p)
        assert flagged.empty, (
            "adding a prediction column falsely flagged existing rows: "
            f"{len(flagged)} row(s)"
        )
        # A genuine edit must still be caught under the new schema.
        df = pd.read_csv(p)
        df.loc[0, "p_home"] = 0.123
        df.to_csv(p, index=False)
        assert len(verify_log(p)) == 1, "a real edit went undetected after a schema change"
    finally:
        logmod.PRED_FIELDS = original


def test_hash_fields_is_recorded_with_each_row(tmp_path, sample):
    p = tmp_path / "log.csv"
    append_predictions(sample, p, vintage="v1")
    df = pd.read_csv(p)
    assert "hash_fields" in df.columns
    assert df["hash_fields"].str.contains("p_home").all()
