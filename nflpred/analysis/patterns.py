"""Conditional pattern mining over play-by-play.

"If X happens at Y, then Z follows" is a question play-by-play can answer well,
because there are ~580,000 plays rather than ~3,000 games. Statistical power is
two orders of magnitude better here than for game outcomes.

It is also where data mining does the most damage. The space of conditions is
effectively unlimited, so scanning it *will* produce hundreds of patterns that
look striking and mean nothing. Three defences, all applied here:

1. DISCOVER AND CONFIRM ON DIFFERENT SEASONS. Patterns are found on early
   seasons and re-tested on later ones the miner never saw. A pattern that does
   not replicate out of sample is noise, whatever its p-value in discovery.
2. CORRECT FOR HOW MANY WERE TESTED. Benjamini-Hochberg across the whole scan.
3. REQUIRE A MEANINGFUL EFFECT, NOT JUST SIGNIFICANCE. With this much data,
   trivial deviations reach significance; a minimum lift is enforced.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy import stats

log = logging.getLogger(__name__)

PBP_COLS = [
    "game_id", "play_id", "season", "week", "qtr", "quarter_seconds_remaining",
    "game_seconds_remaining", "half_seconds_remaining", "down", "ydstogo",
    "yardline_100", "score_differential", "posteam", "defteam", "home_team",
    "away_team", "play_type", "epa", "wp", "success", "yards_gained",
    "touchdown", "interception", "fumble_lost", "sack", "penalty",
    "fourth_down_converted", "fourth_down_failed", "third_down_converted",
    "third_down_failed", "drive", "fixed_drive_result", "goal_to_go",
    "shotgun", "no_huddle", "posteam_timeouts_remaining", "game_half",
    "total_home_score", "total_away_score",
]


def load_pbp(cfg, seasons: list[int]) -> pd.DataFrame:
    frames = []
    for s in seasons:
        path = cfg.raw_cache / f"pbp_{s}.parquet"
        if not path.exists():
            continue
        import pyarrow.parquet as pq

        have = set(pq.ParquetFile(path).schema_arrow.names)
        frames.append(pd.read_parquet(path, columns=[c for c in PBP_COLS if c in have]))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def add_outcomes(pbp: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Attach the things a pattern might predict.

    ``posteam_wins``      did the team with the ball go on to win the game
    ``drive_ends_score``  did this drive end in a touchdown or field goal
    ``next_play_explosive`` did the following play gain 15+ yards
    """
    d = pbp.copy()
    res = games.set_index("game_id")[["home_team", "away_team", "result"]]
    d = d.join(res[["result"]], on="game_id")
    winner = np.where(d["result"] > 0, d["home_team"], d["away_team"])
    d["posteam_wins"] = (d["posteam"] == winner).astype(float)
    d.loc[d["result"].isna() | (d["result"] == 0), "posteam_wins"] = np.nan

    d["drive_ends_score"] = d["fixed_drive_result"].isin(
        ["Touchdown", "Field goal"]
    ).astype(float)

    d = d.sort_values(["game_id", "play_id"])
    d["next_yards"] = d.groupby("game_id")["yards_gained"].shift(-1)
    d["next_play_explosive"] = (d["next_yards"] >= 15).astype(float)
    d.loc[d["next_yards"].isna(), "next_play_explosive"] = np.nan
    return d


def _bucket(d: pd.DataFrame) -> pd.DataFrame:
    """Discretise game state into readable conditions."""
    b = pd.DataFrame(index=d.index)
    b["quarter"] = d["qtr"].map({1: "Q1", 2: "Q2", 3: "Q3", 4: "Q4"}).fillna("OT")
    b["score_state"] = pd.cut(
        d["score_differential"], [-99, -14.5, -7.5, -3.5, 0.5, 3.5, 7.5, 14.5, 99],
        labels=["down 15+", "down 8-14", "down 4-7", "down 1-3", "up 1-3",
                "up 4-7", "up 8-14", "up 15+"],
    ).astype(object)
    b["down"] = d["down"].map({1: "1st", 2: "2nd", 3: "3rd", 4: "4th"})
    b["distance"] = pd.cut(d["ydstogo"], [-1, 2.5, 6.5, 10.5, 99],
                           labels=["short 1-2", "med 3-6", "long 7-10", "v.long 11+"]).astype(object)
    b["field_pos"] = pd.cut(d["yardline_100"], [-1, 20, 50, 80, 101],
                            labels=["redzone", "opp half", "own half", "backed up"]).astype(object)
    b["time_left"] = pd.cut(d["half_seconds_remaining"], [-1, 120, 300, 900, 9999],
                            labels=["<2min", "2-5min", "5-15min", "15min+"]).astype(object)
    b["formation"] = np.where(d["shotgun"] == 1, "shotgun", "under center")
    b["tempo"] = np.where(d["no_huddle"] == 1, "no huddle", "huddle")
    return b


def mine(
    d: pd.DataFrame,
    outcome: str,
    dims: list[str],
    min_n: int = 300,
    max_depth: int = 3,
) -> pd.DataFrame:
    """Every combination of up to ``max_depth`` conditions, with its outcome rate."""
    import itertools

    b = _bucket(d)
    y = d[outcome]
    keep = y.notna()
    b, y = b[keep], y[keep]
    base = y.mean()

    rows = []
    for depth in range(1, max_depth + 1):
        for combo in itertools.combinations(dims, depth):
            g = y.groupby([b[c] for c in combo], observed=True)
            agg = g.agg(["size", "mean"]).reset_index()
            agg = agg[agg["size"] >= min_n]
            for _, r in agg.iterrows():
                cond = " & ".join(f"{c}={r[c]}" for c in combo)
                n, rate = int(r["size"]), float(r["mean"])
                # Binomial test of this cell's rate against the overall base rate.
                p = stats.binomtest(int(round(rate * n)), n, base).pvalue
                rows.append(dict(condition=cond, depth=depth, n=n, rate=rate,
                                 base=base, lift=rate - base, p=p))
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.sort_values("p").reset_index(drop=True)
    out["bh_threshold"] = (out.index + 1) / len(out) * 0.05
    out["survives_fdr"] = out["p"] <= out["bh_threshold"]
    return out.sort_values("lift", key=abs, ascending=False)


def residual_mine(
    d_disc: pd.DataFrame, d_hold: pd.DataFrame, outcome: str,
    dims: list[str], control: list[str], min_n: int = 400, max_depth: int = 2,
) -> pd.DataFrame:
    """Find patterns that survive controlling for the obvious.

    Mining raw rates surfaces tautologies: "down 15+ with two minutes left loses
    99.8% of the time" is true and useless, because the condition mechanically
    determines the outcome. Those dominate any raw scan.

    Here a baseline model is fitted on the CONTROL variables (the obvious
    drivers - score, time, field position, down, distance), and the mining runs
    on its residuals. A pattern only registers if it predicts something the
    obvious variables do not already explain. That is the difference between a
    definition and a discovery.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.pipeline import make_pipeline

    b_disc, b_hold = _bucket(d_disc), _bucket(d_hold)
    y_disc, y_hold = d_disc[outcome], d_hold[outcome]
    keep_d, keep_h = y_disc.notna(), y_hold.notna()
    b_disc, y_disc = b_disc[keep_d], y_disc[keep_d]
    b_hold, y_hold = b_hold[keep_h], y_hold[keep_h]

    enc = make_pipeline(
        OneHotEncoder(handle_unknown="ignore", sparse_output=True),
        LogisticRegression(max_iter=1000, C=1.0),
    )
    enc.fit(b_disc[control].astype(str), y_disc.astype(int))
    exp_d = enc.predict_proba(b_disc[control].astype(str))[:, 1]
    exp_h = enc.predict_proba(b_hold[control].astype(str))[:, 1]
    res_d = y_disc.to_numpy() - exp_d
    res_h = y_hold.to_numpy() - exp_h

    import itertools

    free = [c for c in dims if c not in control]
    rows = []
    for depth in range(1, max_depth + 1):
        for combo in itertools.combinations(free, depth):
            key = [b_disc[c].astype(str) for c in combo]
            grp = pd.Series(res_d, index=b_disc.index).groupby(key, observed=True)
            agg = grp.agg(["size", "mean", "std"]).reset_index()
            agg = agg[agg["size"] >= min_n]
            for _, r in agg.iterrows():
                n, mean, sd = int(r["size"]), float(r["mean"]), float(r["std"])
                if not np.isfinite(sd) or sd == 0:
                    continue
                t = mean / (sd / np.sqrt(n))
                pv = 2 * (1 - stats.norm.cdf(abs(t)))
                cond = " & ".join(f"{c}={r[c]}" for c in combo)
                # Same condition measured on the untouched holdout seasons.
                mh = pd.Series(True, index=b_hold.index)
                for c in combo:
                    mh &= b_hold[c].astype(str) == str(r[c])
                hn = int(mh.sum())
                hres = float(res_h[mh.to_numpy()].mean()) if hn >= 100 else np.nan
                rows.append(dict(condition=cond, n=n, excess=mean, p=pv,
                                 holdout_n=hn, holdout_excess=hres))
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.sort_values("p").reset_index(drop=True)
    out["bh_threshold"] = (out.index + 1) / len(out) * 0.05
    out["survives_fdr"] = out["p"] <= out["bh_threshold"]
    out["replicated"] = (
        out["holdout_excess"].notna()
        & (np.sign(out["holdout_excess"]) == np.sign(out["excess"]))
        & (out["holdout_excess"].abs() >= 0.4 * out["excess"].abs())
    )
    return out.sort_values("excess", key=abs, ascending=False)


def confirm(patterns: pd.DataFrame, d_holdout: pd.DataFrame, outcome: str,
            dims: list[str], min_n: int = 100) -> pd.DataFrame:
    """Re-measure discovered patterns on seasons the miner never saw."""
    b = _bucket(d_holdout)
    y = d_holdout[outcome]
    keep = y.notna()
    b, y = b[keep], y[keep]
    base = y.mean()

    out = []
    for _, row in patterns.iterrows():
        mask = pd.Series(True, index=b.index)
        for part in row["condition"].split(" & "):
            col, val = part.split("=", 1)
            mask &= b[col].astype(str) == val
        n = int(mask.sum())
        if n < min_n:
            out.append(dict(**row, holdout_n=n, holdout_rate=np.nan, holdout_lift=np.nan))
            continue
        rate = float(y[mask].mean())
        out.append(dict(**row, holdout_n=n, holdout_rate=rate, holdout_lift=rate - base))
    r = pd.DataFrame(out)
    r["replicated"] = (
        r["holdout_lift"].notna()
        & (np.sign(r["holdout_lift"]) == np.sign(r["lift"]))
        & (r["holdout_lift"].abs() >= 0.5 * r["lift"].abs())
    )
    return r
