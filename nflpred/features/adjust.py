"""Opponent-adjusted team ratings.

Raw rolling EPA conflates a team's quality with the quality of the defences it
happened to face. We decompose it instead: for every team-game we regress the
observed offensive value on

    offence-team effect + defence-opponent effect + home-field effect

and read the fitted coefficients as opponent-adjusted offensive and defensive
ratings. Ridge shrinkage pulls thin samples toward league average, which is
exactly the behaviour we want in September.

The whole thing is refit once per (season, week) boundary using only games that
kicked off strictly before that week's first kickoff, so a rating attached to a
game never saw that game - or any of its contemporaries.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import Ridge

log = logging.getLogger(__name__)

ADJUST_METRICS = ["off_epa_play", "off_success_rate"]


def _design(
    teams: list[str], off: pd.Series, dfn: pd.Series, home: pd.Series
) -> sparse.csr_matrix:
    index = {t: i for i, t in enumerate(teams)}
    n, k = len(off), len(teams)
    rows = np.repeat(np.arange(n), 2)
    cols = np.empty(2 * n, dtype=int)
    cols[0::2] = [index[t] for t in off]
    cols[1::2] = [index[t] + k for t in dfn]
    data = np.ones(2 * n, dtype=float)
    X = sparse.csr_matrix((data, (rows, cols)), shape=(n, 2 * k))
    return sparse.hstack([X, sparse.csr_matrix(home.to_numpy(dtype=float).reshape(-1, 1))]).tocsr()


def _fit_ratings(
    hist: pd.DataFrame, metric: str, alpha: float, halflife_days: float, asof
) -> tuple[dict[str, float], dict[str, float], float]:
    """Fit one ridge decomposition and return (off_ratings, def_ratings, home_effect)."""
    d = hist[hist[metric].notna()]
    if len(d) < 50:
        return {}, {}, 0.0
    teams = sorted(set(d["team"]) | set(d["opponent"]))
    X = _design(teams, d["team"], d["opponent"], d["is_home"])
    y = d[metric].to_numpy(dtype=float)

    age_days = (asof - d["kickoff"]).dt.total_seconds().to_numpy() / 86400.0
    w = np.exp(-np.log(2.0) * np.clip(age_days, 0, None) / halflife_days)

    model = Ridge(alpha=alpha, fit_intercept=True, solver="sparse_cg", max_iter=5000)
    model.fit(X, y, sample_weight=w)
    k = len(teams)
    coef = model.coef_
    off = {t: float(coef[i]) for i, t in enumerate(teams)}
    dfn = {t: float(coef[i + k]) for i, t in enumerate(teams)}
    return off, dfn, float(coef[2 * k])


def opponent_adjusted_ratings(
    team_games: pd.DataFrame,
    box: pd.DataFrame,
    alpha: float = 12.0,
    halflife_days: float = 400.0,
    min_games: int = 64,
    metrics: list[str] | None = None,
) -> pd.DataFrame:
    """Opponent-adjusted ratings for every team-game, computed as-of its week.

    Returns one row per (game_id, team) with ``adj_<metric>_off`` and
    ``adj_<metric>_def`` columns.
    """
    metrics = metrics or ADJUST_METRICS
    cols = ["game_id", "team", "opponent", "season", "week", "kickoff", "is_home"]
    df = team_games[cols].merge(box, on=["game_id", "team"], how="left")
    df = df.sort_values("kickoff").reset_index(drop=True)

    # One as-of boundary per (season, week): the first kickoff of that week.
    bounds = (
        df.groupby(["season", "week"], sort=True)["kickoff"].min().reset_index(name="asof")
        .sort_values("asof")
    )

    pieces = []
    for _, row in bounds.iterrows():
        asof = row["asof"]
        hist = df[df["kickoff"] < asof]
        target = df[(df["season"] == row["season"]) & (df["week"] == row["week"])]
        if target.empty:
            continue
        rec = target[["game_id", "team"]].copy()
        if len(hist) < min_games:
            for m in metrics:
                rec[f"adj_{m}_off"] = np.nan
                rec[f"adj_{m}_def"] = np.nan
            pieces.append(rec)
            continue
        for m in metrics:
            off, dfn, _ = _fit_ratings(hist, m, alpha, halflife_days, asof)
            rec[f"adj_{m}_off"] = target["team"].map(off).astype(float)
            rec[f"adj_{m}_def"] = target["team"].map(dfn).astype(float)
        pieces.append(rec)

    out = pd.concat(pieces, ignore_index=True)
    # Ridge centres ratings near zero; a team unseen in the training window
    # legitimately gets league average.
    for m in metrics:
        out[f"adj_{m}_net"] = out[f"adj_{m}_off"] - out[f"adj_{m}_def"]
    return out
