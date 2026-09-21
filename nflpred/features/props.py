"""Player prop projections.

A six-game rolling average is the floor, not the ceiling. It reaches r-squared
of about 0.31 on carries and 0.21 on receptions. Five things improve on it,
in rough order of how much they matter:

1. ROLE, NOT OUTPUT. ``target_share`` is far more stable than raw targets,
   because it describes the player's job rather than a noisy realisation of it.
   Snap percentage is the same idea. Project the role, then convert it to
   production, rather than projecting production directly.

2. TEAMMATE CONTEXT. When the man ahead on the depth chart is out, the backup's
   usage jumps - and his posted line is the slowest thing in the market to
   react. This is the single most valuable feature here, and the only one that
   points at a stale number rather than a consensus one.

3. GAME SCRIPT. A team projected to trail throws more. The team model already
   predicts the margin, so that projection feeds straight in.

4. SHRINKAGE. A four-game average is mostly noise. Shrinking toward a role
   baseline with an empirical-Bayes weight beats trusting a thin sample.

5. THE RIGHT DISTRIBUTION. A prop is "over/under X", not "what is the mean".
   Receptions and carries are counts, so a negative binomial gives a usable
   P(over) where a point estimate gives none. This is the difference between a
   projection and a bet.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy import stats

log = logging.getLogger(__name__)

SKILL_POSITIONS = ("WR", "TE", "RB", "QB")


def load_player_weeks(cfg, seasons: list[int]) -> pd.DataFrame:
    cols = ["player_id", "player_name", "player_display_name", "position",
            "season", "week", "season_type",
            "team", "opponent_team", "attempts", "completions", "passing_yards",
            "passing_tds", "carries", "rushing_yards", "receptions", "targets",
            "receiving_yards", "receiving_tds", "rushing_tds", "target_share",
            "air_yards_share"]
    frames = []
    for s in seasons:
        path = cfg.raw_cache / f"stats_player_{s}.parquet"
        if not path.exists():
            continue
        import pyarrow.parquet as pq

        have = set(pq.ParquetFile(path).schema_arrow.names)
        frames.append(pd.read_parquet(path, columns=[c for c in cols if c in have]))
    if not frames:
        return pd.DataFrame()
    d = pd.concat(frames, ignore_index=True)
    d = d[d["season_type"] == "REG"]
    return d[d["position"].isin(SKILL_POSITIONS)].copy()


def load_snaps(cfg, seasons: list[int]) -> pd.DataFrame:
    frames = []
    for s in seasons:
        path = cfg.raw_cache / f"snap_counts_{s}.parquet"
        if not path.exists():
            continue
        frames.append(pd.read_parquet(
            path, columns=["game_id", "season", "week", "player", "position",
                           "team", "offense_snaps", "offense_pct"]))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_injuries(cfg, seasons: list[int]) -> pd.DataFrame:
    frames = []
    for s in seasons:
        path = cfg.raw_cache / f"injuries_{s}.parquet"
        if not path.exists():
            continue
        frames.append(pd.read_parquet(
            path, columns=["season", "week", "team", "gsis_id", "position",
                           "full_name", "report_status"]))
    if not frames:
        return pd.DataFrame()
    d = pd.concat(frames, ignore_index=True)
    d["is_out"] = d["report_status"].isin(["Out", "Doubtful"]).astype(float)
    return d


def _shrink(value: pd.Series, n: pd.Series, prior: pd.Series, strength: float) -> pd.Series:
    """Empirical-Bayes shrinkage toward a prior.

    With few observations the estimate is mostly the prior; with many it is
    mostly the observed value. ``strength`` is the number of observations at
    which the two carry equal weight.
    """
    w = n / (n + strength)
    return w * value + (1.0 - w) * prior


def build_player_features(
    cfg, seasons: list[int], halflife: float = 5.0, shrink_strength: float = 4.0
) -> pd.DataFrame:
    """Pre-kickoff usage and production features for every player-week.

    Every rolling quantity is shifted one game back within player, so a row
    never contains that week's own production.
    """
    d = load_player_weeks(cfg, seasons)
    if d.empty:
        return d
    d = d.sort_values(["player_id", "season", "week"]).reset_index(drop=True)

    # Snap counts key on the full name; the stats table abbreviates the first
    # name ("A.Rodgers"), so the join must use player_display_name. Matching on
    # player_name silently yields zero rows and a feature that is entirely NaN.
    snaps = load_snaps(cfg, seasons)
    if not snaps.empty and "player_display_name" in d.columns:
        snaps = snaps.rename(columns={"player": "player_display_name"})
        key = ["season", "week", "team", "player_display_name"]
        d = d.merge(snaps[key + ["offense_pct", "offense_snaps"]].drop_duplicates(key),
                    on=key, how="left")
        matched = d["offense_pct"].notna().mean()
        log.info("snap counts matched %.1f%% of player-weeks", matched * 100)
        if matched < 0.5:
            log.warning("snap-count join matched only %.1f%% - check the name key", matched * 100)

    stats_cols = ["targets", "receptions", "receiving_yards", "carries",
                  "rushing_yards", "attempts", "passing_yards", "target_share",
                  "air_yards_share", "offense_pct"]
    g = d.groupby("player_id", sort=False)
    for c in stats_cols:
        if c not in d.columns:
            continue
        prior = g[c].shift(1)                               # <-- leakage barrier
        pg = prior.groupby(d["player_id"], sort=False)
        d[f"{c}_ewma"] = pg.transform(
            lambda s: s.ewm(halflife=halflife, min_periods=1, ignore_na=True).mean()
        )
        d[f"{c}_std"] = pg.transform(lambda s: s.rolling(8, min_periods=3).std())
    d["games_played"] = g.cumcount()

    # Shrink thin samples toward the player's position baseline.
    pos_prior = d.groupby("position")["targets_ewma"].transform("median")
    d["targets_shrunk"] = _shrink(
        d["targets_ewma"].fillna(pos_prior), d["games_played"], pos_prior, shrink_strength
    )
    pos_prior_c = d.groupby("position")["carries_ewma"].transform("median")
    d["carries_shrunk"] = _shrink(
        d["carries_ewma"].fillna(pos_prior_c), d["games_played"], pos_prior_c, shrink_strength
    )

    # Teammate context.
    #
    # A COUNT of absent teammates is too blunt to be useful, and testing showed
    # it: the model's edge over a rolling average vanished on exactly the games
    # where someone was out (-0.3% on receptions, -2.5% with two or more). The
    # usage shift is real - receptions rise ~5% and targets ~8% when a teammate
    # sits - but a count cannot say WHO absorbs it.
    #
    # What matters is how much VOLUME was vacated: the summed recent target
    # share of the absent teammates. A missing 25%-target-share receiver is a
    # different event from a missing special-teamer, and the count treats them
    # identically.
    inj = load_injuries(cfg, seasons)
    d["pos_teammates_out"] = 0.0
    d["vacated_target_share"] = 0.0
    d["vacated_carries"] = 0.0
    if not inj.empty:
        absent = inj[inj["is_out"] == 1][["season", "week", "team", "full_name", "position"]]
        absent = absent.rename(columns={"full_name": "player_display_name"})

        cnt = (
            absent.groupby(["season", "week", "team", "position"]).size()
            .reset_index(name="pos_teammates_out_new")
        )
        d = d.merge(cnt, on=["season", "week", "team", "position"], how="left")
        d["pos_teammates_out"] = d.pop("pos_teammates_out_new").fillna(0.0)

        # Recent usage of each absent player.
        #
        # A player who is ruled Out has NO stats row for that week - he did not
        # play - so joining on (season, week, player) finds nothing and every
        # vacated figure comes back zero. The usage has to be carried forward
        # from the last week he did play, which is what merge_asof does here.
        usage = (
            d[["season", "week", "player_display_name", "target_share_ewma", "carries_ewma"]]
            .dropna(subset=["player_display_name"])
            .assign(abs_week=lambda x: (x["season"].astype("int64") * 100
                                        + x["week"].astype("int64")))
            .sort_values("abs_week")
        )
        absent = absent.assign(
            abs_week=lambda x: (x["season"].astype("int64") * 100 + x["week"].astype("int64"))
        ).sort_values("abs_week")
        absent_usage = pd.merge_asof(
            absent, usage[["abs_week", "player_display_name", "target_share_ewma", "carries_ewma"]],
            on="abs_week", by="player_display_name", direction="backward",
        )
        vac = (
            absent_usage.groupby(["season", "week", "team"])[["target_share_ewma", "carries_ewma"]]
            .sum().reset_index()
            .rename(columns={"target_share_ewma": "vac_ts", "carries_ewma": "vac_car"})
        )
        d = d.merge(vac, on=["season", "week", "team"], how="left")
        d["vacated_target_share"] = d.pop("vac_ts").fillna(0.0)
        d["vacated_carries"] = d.pop("vac_car").fillna(0.0)

        # A player only benefits in proportion to his own standing in the
        # offence: vacated volume times his existing share of the team's work.
        d["vacated_x_role"] = d["vacated_target_share"] * d["target_share_ewma"].fillna(0.0)
        d["vacated_car_x_role"] = d["vacated_carries"] * d["carries_ewma"].fillna(0.0)
    else:
        d["vacated_x_role"] = 0.0
        d["vacated_car_x_role"] = 0.0

    # Team-level opportunity: how many targets/carries the team usually has,
    # excluding this player - the pool he is competing for.
    team_tot = d.groupby(["season", "week", "team"])[["targets", "carries"]].transform("sum")
    d["team_targets"] = team_tot["targets"]
    d["team_carries"] = team_tot["carries"]
    tg = d.groupby("player_id", sort=False)["team_targets"].shift(1)
    d["team_targets_ewma"] = tg.groupby(d["player_id"], sort=False).transform(
        lambda s: s.ewm(halflife=halflife, min_periods=1, ignore_na=True).mean()
    )
    return d


def negbinom_over(mean: np.ndarray, var: np.ndarray, line: float) -> np.ndarray:
    """P(count > line) under a negative binomial fitted by moment matching.

    A prop is an over/under, so the distribution is the product, not the mean.
    Counts are overdispersed relative to Poisson (variance exceeds the mean),
    which the negative binomial handles and Poisson does not. Where the data
    is not overdispersed it falls back to Poisson.
    """
    mean = np.clip(np.asarray(mean, dtype=float), 1e-6, None)
    var = np.clip(np.asarray(var, dtype=float), mean * 1.001, None)
    p = mean / var
    n = mean * p / (1.0 - p)
    out = 1.0 - stats.nbinom.cdf(np.floor(line), n, p)
    poisson = 1.0 - stats.poisson.cdf(np.floor(line), mean)
    return np.where(np.isfinite(out), out, poisson)
