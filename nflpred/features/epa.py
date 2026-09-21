"""Team performance features derived from play-by-play.

Two stages, deliberately separated:

1. ``game_box`` turns play-by-play into one row per (game, team) holding what
   that team *did in that game*. These are game outcomes and are never used
   directly as features.
2. ``rolling_features`` converts those outcomes into pre-kickoff features by
   shifting one game back within each team and then aggregating. The
   ``shift(1)`` is the single line that makes the whole feature set leak-free,
   and ``tests/test_leakage.py`` asserts it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Win-probability band that excludes garbage time. Plays outside it are noisy
# and systematically unrepresentative of team strength.
WP_LOW, WP_HIGH = 0.05, 0.95


def _scrimmage(pbp: pd.DataFrame, wp_filter: bool = True) -> pd.DataFrame:
    df = pbp[
        pbp["play_type"].isin(["pass", "run"])
        & pbp["epa"].notna()
        & pbp["posteam"].notna()
        & pbp["defteam"].notna()
    ].copy()
    if wp_filter and "wp" in df.columns:
        wp = pd.to_numeric(df["wp"], errors="coerce")
        df = df[wp.between(WP_LOW, WP_HIGH) | wp.isna()]
    for c in ("pass", "rush", "success", "epa", "down", "ydstogo", "yards_gained",
              "xpass", "sack", "interception", "fumble_lost", "shotgun", "no_huddle"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _agg(df: pd.DataFrame, key: str) -> pd.DataFrame:
    """Aggregate plays to one row per (game_id, team) from ``key``'s perspective."""
    is_pass = df["pass"] == 1
    is_rush = df["rush"] == 1
    early = df["down"].isin([1, 2])
    explosive = ((is_pass) & (df["yards_gained"] >= 20)) | ((is_rush) & (df["yards_gained"] >= 10))

    work = pd.DataFrame(
        {
            "game_id": df["game_id"],
            "team": df[key],
            "epa": df["epa"],
            "success": df["success"],
            "is_pass": is_pass.astype(float),
            "is_rush": is_rush.astype(float),
            "pass_epa": df["epa"].where(is_pass),
            "rush_epa": df["epa"].where(is_rush),
            "pass_success": df["success"].where(is_pass),
            "rush_success": df["success"].where(is_rush),
            "early_epa": df["epa"].where(early),
            "explosive": explosive.astype(float),
            "proe": (df["pass"] - df["xpass"]) if "xpass" in df else np.nan,
            "sack": df.get("sack", pd.Series(np.nan, index=df.index)).where(is_pass),
            "turnover": (
                df.get("interception", pd.Series(0.0, index=df.index)).fillna(0)
                + df.get("fumble_lost", pd.Series(0.0, index=df.index)).fillna(0)
            ),
        }
    )
    out = work.groupby(["game_id", "team"], sort=False).agg(
        epa_play=("epa", "mean"),
        success_rate=("success", "mean"),
        pass_epa=("pass_epa", "mean"),
        rush_epa=("rush_epa", "mean"),
        pass_success=("pass_success", "mean"),
        rush_success=("rush_success", "mean"),
        early_epa=("early_epa", "mean"),
        explosive_rate=("explosive", "mean"),
        pass_rate=("is_pass", "mean"),
        proe=("proe", "mean"),
        sack_rate=("sack", "mean"),
        turnover_rate=("turnover", "mean"),
        plays=("epa", "size"),
    )
    return out.reset_index()


def game_box(pbp: pd.DataFrame, wp_filter: bool = True) -> pd.DataFrame:
    """One row per (game, team): offensive production and defence allowed.

    OUTCOME table - not directly usable as features.
    """
    df = _scrimmage(pbp, wp_filter=wp_filter)
    off = _agg(df, "posteam").add_prefix("off_").rename(
        columns={"off_game_id": "game_id", "off_team": "team"}
    )
    dfn = _agg(df, "defteam").add_prefix("def_").rename(
        columns={"def_game_id": "game_id", "def_team": "team"}
    )
    box = off.merge(dfn, on=["game_id", "team"], how="outer")
    # Net EPA: how much better the offence was than the defence allowed.
    box["net_epa"] = box["off_epa_play"] - box["def_epa_play"]
    return box


# Metrics carried forward into rolling features.
METRICS = [
    "off_epa_play", "off_success_rate", "off_pass_epa", "off_rush_epa",
    "off_early_epa", "off_explosive_rate", "off_proe", "off_sack_rate",
    "off_turnover_rate",
    "def_epa_play", "def_success_rate", "def_pass_epa", "def_rush_epa",
    "def_early_epa", "def_explosive_rate",
    "net_epa",
]


def rolling_features(
    team_games: pd.DataFrame,
    box: pd.DataFrame,
    halflife: float = 6.0,
    windows: tuple[int, ...] = (8,),
) -> pd.DataFrame:
    """Pre-kickoff rolling form for every team-game.

    ``team_games`` must be sorted by kickoff. For each team the series is
    shifted one game back before any aggregation, so the value attached to
    game *g* uses games strictly before *g* and nothing else.
    """
    tg = team_games[["game_id", "team", "season", "week", "kickoff"]].copy()
    df = tg.merge(box, on=["game_id", "team"], how="left")
    df = df.sort_values(["team", "kickoff"], kind="mergesort").reset_index(drop=True)

    out = df[["game_id", "team"]].copy()
    grp = df.groupby("team", sort=False)

    for col in METRICS:
        if col not in df.columns:
            continue
        prior = grp[col].shift(1)                     # <-- the leakage barrier
        pg = prior.groupby(df["team"], sort=False)
        out[f"{col}_ewma"] = pg.transform(
            lambda s: s.ewm(halflife=halflife, min_periods=1, ignore_na=True).mean()
        )
        for w in windows:
            out[f"{col}_r{w}"] = pg.transform(
                lambda s, w=w: s.rolling(w, min_periods=2).mean()
            )

    # Experience counters, also strictly backward-looking.
    out["games_played"] = grp.cumcount()
    prior_season = grp["season"].shift(1)
    same_season = (prior_season == df["season"]).fillna(False)
    out["games_this_season"] = (
        same_season.groupby(df["team"], sort=False).cumsum().astype(float)
    )
    # Recent form: mean scoring margin over the previous 5 games.
    if "margin" in team_games.columns:
        m = df.merge(team_games[["game_id", "team", "margin"]], on=["game_id", "team"], how="left")["margin"]
        pm = m.groupby(df["team"], sort=False).shift(1)
        out["margin_r5"] = pm.groupby(df["team"], sort=False).transform(
            lambda s: s.rolling(5, min_periods=2).mean()
        )
    return out
