"""Derived prediction targets that are not in the schedule file.

Halftime scores have to come from play-by-play. The subtlety is that
``total_home_score`` holds the score **after** the play it sits on, so the
naive "read the first play of the second half" approach silently includes any
points scored on the opening kickoff of the second half - a returned kickoff
touchdown lands in the halftime score. Taking the maximum over first-half
plays avoids that, and is also robust to play ordering, since a score is
monotonically non-decreasing within a game.

Validated against games.csv: the same max-based rule applied to the whole game
reproduces the official final score for every game in the dataset.
"""
from __future__ import annotations

import logging

import pandas as pd

log = logging.getLogger(__name__)

FIRST_HALF = "Half1"


def halftime_scores(pbp: pd.DataFrame) -> pd.DataFrame:
    """Halftime home/away score for every game present in ``pbp``.

    Returns one row per game_id with h1_home_score / h1_away_score.
    """
    needed = {"game_id", "game_half", "total_home_score", "total_away_score"}
    missing = needed - set(pbp.columns)
    if missing:
        raise KeyError(f"play-by-play is missing columns required for halftime: {sorted(missing)}")

    h1 = pbp[pbp["game_half"] == FIRST_HALF]
    if h1.empty:
        return pd.DataFrame(columns=["game_id", "h1_home_score", "h1_away_score"])

    out = (
        h1.groupby("game_id")[["total_home_score", "total_away_score"]]
        .max()
        .rename(columns={"total_home_score": "h1_home_score", "total_away_score": "h1_away_score"})
        .reset_index()
    )
    return out


def final_scores_from_pbp(pbp: pd.DataFrame) -> pd.DataFrame:
    """Final score by the same rule, used to validate the halftime derivation."""
    return (
        pbp.groupby("game_id")[["total_home_score", "total_away_score"]]
        .max()
        .rename(columns={"total_home_score": "pbp_home_score", "total_away_score": "pbp_away_score"})
        .reset_index()
    )


def attach_halftime_targets(games: pd.DataFrame, pbp: pd.DataFrame) -> pd.DataFrame:
    """Add halftime target columns to a game-level frame.

    Adds h1_home_score, h1_away_score, h1_spread_actual (home margin at half),
    h1_total_actual and h1_home_win (0/1, 0.5 if tied at the break - which is
    common, unlike full-game ties).
    """
    ht = halftime_scores(pbp)
    out = games.merge(ht, on="game_id", how="left")
    out["h1_spread_actual"] = out["h1_home_score"] - out["h1_away_score"]
    out["h1_total_actual"] = out["h1_home_score"] + out["h1_away_score"]
    out["h1_home_win"] = (out["h1_spread_actual"] > 0).astype(float)
    out.loc[out["h1_spread_actual"] == 0, "h1_home_win"] = 0.5
    out.loc[out["h1_spread_actual"].isna(), "h1_home_win"] = pd.NA
    return out
