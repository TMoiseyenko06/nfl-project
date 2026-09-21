"""Append-only prediction log.

This is the most important artifact in the project: the only evidence that the
model works on games it had never seen at the time it called them. Three rules
make it credible:

1. Rows are only ever appended, never rewritten. ``append_predictions``
   refuses to log a (model, game_id) that is already present.
2. Every row is stamped with the UTC time it was written and the data vintage
   it was produced from.
3. Every row carries a content hash over the prediction fields. If a row is
   edited after the fact, ``verify_log`` reports it. A log you can quietly
   rewrite is not a track record.
"""
from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

PRED_FIELDS = [
    "logged_at", "vintage", "model", "game_id", "season", "week", "kickoff",
    "home_team", "away_team", "p_home", "pred_spread", "pred_total",
    "vegas_spread", "vegas_total",
]


def _row_hash(row: pd.Series) -> str:
    parts = []
    for f in PRED_FIELDS:
        v = row.get(f)
        if isinstance(v, float) and np.isfinite(v):
            parts.append(f"{f}={v:.10g}")
        else:
            parts.append(f"{f}={v}")
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def append_predictions(preds: pd.DataFrame, path: str | Path, vintage: str = "unknown") -> pd.DataFrame:
    """Append new predictions. Existing (model, game_id) rows are left untouched.

    Returns the rows actually written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    new = preds.copy()
    new["logged_at"] = now
    new["vintage"] = vintage
    for f in PRED_FIELDS:
        if f not in new.columns:
            new[f] = np.nan
    new = new[PRED_FIELDS]
    new["row_hash"] = new.apply(_row_hash, axis=1)

    if path.exists():
        existing = pd.read_csv(path)
        have = set(zip(existing["model"], existing["game_id"]))
        mask = [(m, g) not in have for m, g in zip(new["model"], new["game_id"])]
        skipped = len(new) - sum(mask)
        if skipped:
            log.info("%d prediction(s) already logged; not overwriting", skipped)
        new = new[pd.Series(mask, index=new.index)]
        if new.empty:
            return new
        out = pd.concat([existing, new], ignore_index=True)
    else:
        out = new

    out.to_csv(path, index=False)
    log.info("logged %d prediction(s) -> %s", len(new), path)
    return new


def verify_log(path: str | Path) -> pd.DataFrame:
    """Recompute each row's hash and return any rows that were edited after logging."""
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "row_hash" not in df.columns:
        return df.assign(reason="no hash column")
    recomputed = df.apply(_row_hash, axis=1)
    bad = df[recomputed != df["row_hash"]]
    return bad


def logged_predictions_before_kickoff(path: str | Path) -> pd.DataFrame:
    """Only the rows that were written before the game they predict kicked off.

    Anything logged after kickoff is excluded from the track record. This is
    the difference between a real prediction and a retrodiction.
    """
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if df.empty:
        return df
    logged = pd.to_datetime(df["logged_at"], utc=True, errors="coerce")
    df["logged_before_kickoff"] = logged < _kickoff_utc(df["kickoff"])
    return df


# US/Eastern is UTC-4 (EDT) through early November and UTC-5 (EST) after. If the
# IANA database is unavailable we assume EDT, which places kickoff EARLIER in
# UTC and so is the stricter assumption: it can only reject a borderline row,
# never wave one through.
_FALLBACK_EASTERN_OFFSET_HOURS = 4


def _kickoff_utc(kickoff: pd.Series) -> pd.Series:
    """Kickoff (US/Eastern wall clock) as a UTC timestamp."""
    naive = pd.to_datetime(kickoff, errors="coerce")
    try:
        return naive.dt.tz_localize(
            "US/Eastern", ambiguous="NaT", nonexistent="NaT"
        ).dt.tz_convert("UTC")
    except Exception as exc:  # noqa: BLE001 - missing tzdata must not break scoring
        log.warning(
            "time-zone database unavailable (%s); assuming EDT, which is the "
            "stricter assumption for before-kickoff checks", exc
        )
        return (naive + pd.Timedelta(hours=_FALLBACK_EASTERN_OFFSET_HOURS)).dt.tz_localize("UTC")


def score_logged_predictions(
    log_path: str | Path, features: pd.DataFrame, perf_path: str | Path, cfg
) -> pd.DataFrame:
    """Join logged predictions to finished results and append to the performance log."""
    from nflpred.backtest.metrics import evaluate

    df = logged_predictions_before_kickoff(log_path)
    if df.empty:
        log.warning("no predictions logged yet")
        return pd.DataFrame()

    actual = features[features["played"]][
        ["game_id", "home_win", "spread_actual", "total_actual"]
    ]
    joined = df.merge(actual, on="game_id", how="inner")
    honest = joined[joined["logged_before_kickoff"]]
    if len(honest) < len(joined):
        log.warning(
            "%d logged row(s) were written after kickoff and are excluded",
            len(joined) - len(honest),
        )
    if honest.empty:
        log.warning("no completed games among logged predictions yet")
        return pd.DataFrame()

    rows = []
    scored_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    for (model, season, week), d in honest.groupby(["model", "season", "week"]):
        m = evaluate(d, ats_breakeven=cfg.get("evaluation.ats_breakeven", 0.5238))
        m.update({"scored_at": scored_at, "model": model, "season": int(season), "week": int(week)})
        rows.append(m)
    perf = pd.DataFrame(rows)

    perf_path = Path(perf_path)
    perf_path.parent.mkdir(parents=True, exist_ok=True)
    if perf_path.exists():
        prev = pd.read_csv(perf_path)
        key = ["model", "season", "week"]
        prev = prev[~prev.set_index(key).index.isin(perf.set_index(key).index)]
        perf = pd.concat([prev, perf], ignore_index=True)
    perf = perf.sort_values(["season", "week", "model"])
    perf.to_csv(perf_path, index=False)
    log.info("scored %d model-week(s) -> %s", len(rows), perf_path)
    return perf
