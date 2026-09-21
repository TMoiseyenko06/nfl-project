"""Quarterback features.

The single largest gap between this model and the market is roster news, and
the biggest part of that is who is playing quarterback. A backup is worth
several points of spread, and a model built on team rolling averages is blind
to it.

The important design decision: this encodes **change and quality**, not
identity. An earlier attempt using learned QB-identity embeddings made every
headline metric worse - with ~3,000 games and hundreds of quarterbacks, most
embeddings are fit on a handful of starts and become noise. What actually
carries signal is "is the usual starter playing, and how good has he been",
which is a handful of continuous features rather than a large vocabulary.

Leakage note: ``home_qb_id`` in the schedule file is the quarterback who
actually started. That is known before kickoff (inactives are announced ~90
minutes prior) and nflverse populates it for the upcoming week, so it is
legitimately available pre-kickoff. It is NOT the same as the projected
starter on a Tuesday, so a mid-week prediction leans on the projection being
correct. Every rolling QB statistic below is shifted, so a quarterback's
numbers entering a game never include that game.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

MIN_ATTEMPTS = 10  # a "start" for our purposes


def load_qb_weeks(cfg, seasons: list[int] | None = None) -> pd.DataFrame:
    """Per-quarterback, per-week passing lines from the nflverse player stats."""
    seasons = seasons if seasons is not None else cfg.pbp_seasons
    cols = ["player_id", "player_name", "position", "season", "week", "season_type",
            "team", "attempts", "completions", "passing_yards", "passing_epa",
            "passing_tds", "passing_interceptions"]
    frames = []
    for s in seasons:
        path = cfg.raw_cache / f"stats_player_{s}.parquet"
        if not path.exists():
            log.warning("no player stats cached for %s", s)
            continue
        import pyarrow.parquet as pq

        have = set(pq.ParquetFile(path).schema_arrow.names)
        frames.append(pd.read_parquet(path, columns=[c for c in cols if c in have]))
    if not frames:
        return pd.DataFrame()
    d = pd.concat(frames, ignore_index=True)
    d = d[(d["position"] == "QB") & (d["attempts"] >= MIN_ATTEMPTS)]
    return d


def qb_features(team_games: pd.DataFrame, qb_weeks: pd.DataFrame,
                halflife: float = 4.0) -> pd.DataFrame:
    """Pre-kickoff quarterback features for every team-game.

    Returns one row per (game_id, team):
      qb_epa_ewma      the starter's own rolling passing EPA per attempt,
                       from his prior starts only
      qb_starts        how many prior starts he has in the dataset
      qb_is_new        1 if this is a different starter than the team's last game
      qb_is_rookie_ish 1 if he has fewer than 8 prior starts
      qb_epa_delta     how much better/worse he is than the team's usual starter
    """
    tg = team_games[["game_id", "team", "season", "week", "kickoff", "qb_id"]].copy()
    tg = tg.sort_values(["kickoff", "game_id", "team"]).reset_index(drop=True)

    if qb_weeks.empty:
        out = tg[["game_id", "team"]].copy()
        for c in ("qb_epa_ewma", "qb_starts", "qb_is_new", "qb_is_rookie_ish", "qb_epa_delta"):
            out[c] = np.nan
        return out

    # Per-start EPA per attempt, ordered by when it happened.
    q = qb_weeks.copy()
    q["qb_epa_per_att"] = q["passing_epa"] / q["attempts"].clip(lower=1)
    q = q.sort_values(["player_id", "season", "week"]).reset_index(drop=True)
    grp = q.groupby("player_id", sort=False)["qb_epa_per_att"]
    prior = grp.shift(1)                                   # <-- leakage barrier
    pg = prior.groupby(q["player_id"], sort=False)
    q["qb_epa_ewma"] = pg.transform(
        lambda s: s.ewm(halflife=halflife, min_periods=1, ignore_na=True).mean()
    )
    q["qb_starts"] = q.groupby("player_id", sort=False).cumcount()
    qlook = q.set_index(["season", "week", "player_id"])[["qb_epa_ewma", "qb_starts"]]

    keys = list(zip(tg["season"], tg["week"], tg["qb_id"]))
    joined = qlook.reindex(keys)
    out = tg[["game_id", "team"]].copy()
    out["qb_epa_ewma"] = joined["qb_epa_ewma"].to_numpy()
    out["qb_starts"] = joined["qb_starts"].to_numpy()

    # Did the starter change from this team's previous game?
    prev_qb = tg.groupby("team", sort=False)["qb_id"].shift(1)
    out["qb_is_new"] = (
        tg["qb_id"].notna() & prev_qb.notna() & (tg["qb_id"] != prev_qb)
    ).astype(float)
    out["qb_is_rookie_ish"] = (out["qb_starts"].fillna(0) < 8).astype(float)

    # How this starter compares to whoever the team has been using: the team's
    # own trailing QB quality, shifted so it never includes this game.
    team_qb = out["qb_epa_ewma"].groupby(tg["team"], sort=False).shift(1)
    team_base = team_qb.groupby(tg["team"], sort=False).transform(
        lambda s: s.ewm(halflife=8.0, min_periods=1, ignore_na=True).mean()
    )
    out["qb_epa_delta"] = out["qb_epa_ewma"] - team_base
    return out
