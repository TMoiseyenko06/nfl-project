"""Elo ratings: a Phase-1 baseline and a feature in its own right.

Elo is sequential by construction - a rating entering game *g* is built only
from games that finished before it - so it cannot leak. Ratings are carried
across seasons with partial regression to the mean, which is what makes Elo
competitive in September when rolling windows are still thin.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def _mov_multiplier(margin: float, elo_diff_winner: float) -> float:
    """538's margin-of-victory multiplier, which also damps autocorrelation."""
    return float(np.log(abs(margin) + 1.0) * (2.2 / (elo_diff_winner * 0.001 + 2.2)))


def run_elo(
    games: pd.DataFrame,
    k: float = 20.0,
    home_field: float = 55.0,
    season_regression: float = 0.33,
    initial: float = 1500.0,
    mov_multiplier: bool = True,
) -> pd.DataFrame:
    """Walk the schedule in order, returning pre-game ratings for every game.

    ``games`` must contain game_id, season, kickoff, home_team, away_team,
    result (home margin, NaN if unplayed), location.
    """
    g = games.sort_values("kickoff", kind="mergesort").reset_index(drop=True)
    ratings: dict[str, float] = {}
    last_season: dict[str, int] = {}
    out = []

    for row in g.itertuples(index=False):
        season = int(row.season)
        for team in (row.home_team, row.away_team):
            if team not in ratings:
                ratings[team] = initial
                last_season[team] = season
            elif last_season[team] != season:
                # New season: regress partway to the mean.
                gap = season - last_season[team]
                for _ in range(gap):
                    ratings[team] = initial + (1.0 - season_regression) * (ratings[team] - initial)
                last_season[team] = season

        home_r, away_r = ratings[row.home_team], ratings[row.away_team]
        hfa = 0.0 if getattr(row, "location", "Home") == "Neutral" else home_field
        elo_diff = (home_r + hfa) - away_r
        p_home = expected_score(home_r + hfa, away_r)

        out.append(
            {
                "game_id": row.game_id,
                "elo_home_pre": home_r,
                "elo_away_pre": away_r,
                "elo_diff": elo_diff,
                "elo_prob_home": p_home,
                # 25 Elo points ~ 1 point of spread is the conventional mapping.
                "elo_spread_home": elo_diff / 25.0,
            }
        )

        margin = getattr(row, "result", np.nan)
        if pd.isna(margin):
            continue  # unplayed: rating stays put

        actual = 1.0 if margin > 0 else (0.0 if margin < 0 else 0.5)
        if mov_multiplier and margin != 0:
            winner_diff = elo_diff if margin > 0 else -elo_diff
            mult = _mov_multiplier(margin, winner_diff)
        else:
            mult = 1.0
        delta = k * mult * (actual - p_home)
        ratings[row.home_team] = home_r + delta
        ratings[row.away_team] = away_r - delta

    return pd.DataFrame(out)
