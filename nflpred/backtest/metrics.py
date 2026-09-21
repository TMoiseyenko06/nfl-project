"""Evaluation metrics. Every headline number is reported next to its baseline.

Conventions used throughout:
  spread  = home margin (home_score - away_score); Vegas line is home-perspective,
            positive meaning the home team is favoured by that many points.
  cover   = home team covers when actual home margin > the home-perspective line.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-12


def _clean(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    return y_true[m], y_pred[m]


def accuracy(y_true, p_home) -> float:
    y, p = _clean(y_true, p_home)
    if not len(y):
        return np.nan
    return float(((p > 0.5) == (y > 0.5)).mean())


def log_loss(y_true, p_home) -> float:
    y, p = _clean(y_true, p_home)
    if not len(y):
        return np.nan
    p = np.clip(p, EPS, 1 - EPS)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def brier(y_true, p_home) -> float:
    y, p = _clean(y_true, p_home)
    if not len(y):
        return np.nan
    return float(((p - y) ** 2).mean())


def mae(y_true, y_pred) -> float:
    y, p = _clean(y_true, y_pred)
    return float(np.abs(y - p).mean()) if len(y) else np.nan


def rmse(y_true, y_pred) -> float:
    y, p = _clean(y_true, y_pred)
    return float(np.sqrt(((y - p) ** 2).mean())) if len(y) else np.nan


def ats_record(pred_spread, actual_spread, vegas_spread) -> dict:
    """Against-the-spread record.

    A pick is made only when the model disagrees with the line. Pushes (actual
    margin exactly equal to the line) are excluded from the denominator, which
    is how sportsbooks settle them.
    """
    p = np.asarray(pred_spread, dtype=float)
    a = np.asarray(actual_spread, dtype=float)
    v = np.asarray(vegas_spread, dtype=float)
    m = np.isfinite(p) & np.isfinite(a) & np.isfinite(v)
    p, a, v = p[m], a[m], v[m]
    if not len(p):
        return {"ats_pct": np.nan, "ats_n": 0, "ats_wins": 0, "ats_losses": 0, "ats_pushes": 0}

    pick_home = p > v                       # model thinks home beats the number
    home_covers = a > v
    push = a == v
    decided = ~push
    wins = int((pick_home == home_covers)[decided].sum())
    losses = int(decided.sum() - wins)
    n = wins + losses
    return {
        "ats_pct": (wins / n) if n else np.nan,
        "ats_n": n,
        "ats_wins": wins,
        "ats_losses": losses,
        "ats_pushes": int(push.sum()),
    }


def ats_units(record: dict, juice: float = -110.0) -> float:
    """Net units won risking 1 unit per game at the given American odds."""
    payout = 100.0 / abs(juice)
    return record["ats_wins"] * payout - record["ats_losses"]


def totals_record(pred_total, actual_total, vegas_total) -> dict:
    """Over/under record against the posted total."""
    p = np.asarray(pred_total, dtype=float)
    a = np.asarray(actual_total, dtype=float)
    v = np.asarray(vegas_total, dtype=float)
    m = np.isfinite(p) & np.isfinite(a) & np.isfinite(v)
    p, a, v = p[m], a[m], v[m]
    if not len(p):
        return {"ou_pct": np.nan, "ou_n": 0}
    pick_over = p > v
    went_over = a > v
    decided = a != v
    wins = int((pick_over == went_over)[decided].sum())
    n = int(decided.sum())
    return {"ou_pct": (wins / n) if n else np.nan, "ou_n": n}


def calibration_table(y_true, p_home, bins: int = 10) -> pd.DataFrame:
    """Are 70% predictions right 70% of the time?"""
    y, p = _clean(y_true, p_home)
    if not len(y):
        return pd.DataFrame()
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, bins - 1)
    rows = []
    for b in range(bins):
        m = idx == b
        if not m.any():
            continue
        rows.append(
            {
                "bin": f"{edges[b]:.1f}-{edges[b+1]:.1f}",
                "n": int(m.sum()),
                "mean_pred": float(p[m].mean()),
                "actual": float(y[m].mean()),
                "gap": float(p[m].mean() - y[m].mean()),
            }
        )
    return pd.DataFrame(rows)


def expected_calibration_error(y_true, p_home, bins: int = 10) -> float:
    t = calibration_table(y_true, p_home, bins=bins)
    if t.empty:
        return np.nan
    return float((t["n"] * t["gap"].abs()).sum() / t["n"].sum())


def evaluate(df: pd.DataFrame, ats_breakeven: float = 0.5238) -> dict:
    """Full metric bundle for one model's out-of-sample predictions.

    Expects columns: home_win, spread_actual, total_actual, vegas_spread,
    vegas_total and whichever of p_home / pred_spread / pred_total exist.
    """
    out: dict = {"n_games": int(len(df))}

    if "p_home" in df and df["p_home"].notna().any():
        # Ties (~0.3% of games) have no winner; exclude them from classification.
        d = df[df["home_win"].isin([0.0, 1.0])]
        out.update(
            {
                "n_cls": int(len(d)),
                "accuracy": accuracy(d["home_win"], d["p_home"]),
                "log_loss": log_loss(d["home_win"], d["p_home"]),
                "brier": brier(d["home_win"], d["p_home"]),
                "ece": expected_calibration_error(d["home_win"], d["p_home"]),
            }
        )

    if "pred_spread" in df and df["pred_spread"].notna().any():
        out["spread_mae"] = mae(df["spread_actual"], df["pred_spread"])
        out["spread_rmse"] = rmse(df["spread_actual"], df["pred_spread"])
        rec = ats_record(df["pred_spread"], df["spread_actual"], df["vegas_spread"])
        out.update(rec)
        out["ats_units"] = ats_units(rec)
        out["ats_edge"] = (rec["ats_pct"] - ats_breakeven) if np.isfinite(rec["ats_pct"]) else np.nan

    if "pred_total" in df and df["pred_total"].notna().any():
        out["total_mae"] = mae(df["total_actual"], df["pred_total"])
        out["total_rmse"] = rmse(df["total_actual"], df["pred_total"])
        out.update(totals_record(df["pred_total"], df["total_actual"], df["vegas_total"]))

    return out
