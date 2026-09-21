"""Walk-forward backtesting. No random splits anywhere in this project.

For each (season, week) fold in chronological order:
  train on every completed game that kicked off strictly before the fold's
  first kickoff, predict the fold, record the predictions, roll forward.

That mirrors how the system is actually used on a Tuesday in November, and it
is the only evaluation protocol whose numbers mean anything here.
"""
from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd

from nflpred.backtest.metrics import evaluate_target
from nflpred.config import Config
from nflpred.features.build import select_features
from nflpred.models.targets import resolve

log = logging.getLogger(__name__)


def folds(df: pd.DataFrame, first_season: int, first_week: int) -> list[tuple[int, int]]:
    """Chronologically ordered (season, week) test folds."""
    played = df[df["played"]]
    keys = (
        played[["season", "week"]].drop_duplicates().sort_values(["season", "week"])
    )
    out = []
    for s, w in keys.itertuples(index=False):
        if s > first_season or (s == first_season and w >= first_week):
            out.append((int(s), int(w)))
    return out


def run_backtest(
    df: pd.DataFrame,
    models: list,
    cfg: Config,
    verbose: bool = True,
) -> pd.DataFrame:
    """Return per-game out-of-sample predictions for every model.

    One row per (model, game). Downstream metrics are computed over this frame
    only - never over anything a model saw in training.
    """
    first_season = cfg.get("backtest.first_test_season")
    first_week = cfg.get("backtest.first_test_week", 1)
    min_train = cfg.get("backtest.min_train_games", 500)
    include_post = cfg.get("data.include_postseason_in_training", True)

    df = df.sort_values("kickoff").reset_index(drop=True)
    fold_list = folds(df, first_season, first_week)
    log.info("walk-forward: %d folds from %s week %s", len(fold_list), first_season, first_week)

    trainable = df["played"]
    if not include_post:
        trainable = trainable & (df["game_type"] == "REG")

    records = []
    t0 = time.time()
    for i, (season, week) in enumerate(fold_list):
        test_mask = (df["season"] == season) & (df["week"] == week) & df["played"]
        test = df[test_mask]
        if test.empty:
            continue
        asof = test["kickoff"].min()
        train = df[trainable & (df["kickoff"] < asof)]
        if len(train) < min_train:
            continue

        for model in models:
            feats = select_features(df, model.groups) if model.groups else []
            try:
                model.fit(train, feats)
                preds = model.predict(test, feats)
            except Exception as exc:  # noqa: BLE001 - one bad fold must not kill the run
                log.warning("%s failed on %s wk%s: %s", model.name, season, week, exc)
                continue
            rec = preds.to_frame(test.index)
            rec["model"] = model.name
            rec["game_id"] = test["game_id"].to_numpy()
            rec["season"] = season
            rec["week"] = week
            rec["kickoff"] = test["kickoff"].to_numpy()
            rec["n_train"] = len(train)
            carry = ["home_team", "away_team", "game_type", "vegas_spread", "vegas_total"]
            carry += [s.column for s in resolve(None) if s.column in test.columns]
            for c in carry:
                rec[c] = test[c].to_numpy()
            records.append(rec)

        if verbose and (i + 1) % 25 == 0:
            log.info("  fold %d/%d (%s wk%s) %.1fs", i + 1, len(fold_list), season, week, time.time() - t0)

    if not records:
        raise RuntimeError("backtest produced no predictions - check fold configuration")
    out = pd.concat(records, ignore_index=True)
    log.info("backtest done: %d predictions, %d models, %.1fs",
             len(out), out["model"].nunique(), time.time() - t0)
    return out


def summarize(preds: pd.DataFrame, cfg: Config, model_order: list[str] | None = None,
              targets: list[str] | None = None) -> pd.DataFrame:
    """Comparison table: one row per (target, model), always reported together."""
    breakeven = cfg.get("evaluation.ats_breakeven", 0.5238)
    names = model_order or list(dict.fromkeys(preds["model"]))
    rows = []
    for spec in resolve(targets):
        for name in names:
            d = preds[preds["model"] == name]
            if d.empty:
                continue
            m = evaluate_target(d, spec, breakeven)
            if m.get("n_games", 0) == 0:
                continue
            m["model"] = name
            rows.append(m)
    tab = pd.DataFrame(rows)
    if tab.empty:
        return tab
    cols = ["target", "model", "n_games", "accuracy", "log_loss", "brier", "ece",
            "mae", "rmse", "ats_pct", "ats_n", "ats_units", "ou_pct"]
    return tab[[c for c in cols if c in tab.columns]]


def format_table(tab: pd.DataFrame, breakeven: float = 0.5238) -> str:
    """Human-readable comparison table, grouped by target."""
    from nflpred.models.targets import TARGETS

    if tab.empty:
        return "(no results)"
    fmt = {"accuracy": "{:.4f}", "log_loss": "{:.4f}", "brier": "{:.4f}", "ece": "{:.4f}",
           "mae": "{:.3f}", "rmse": "{:.3f}", "ats_pct": "{:.4f}", "ats_units": "{:+.1f}",
           "ou_pct": "{:.4f}"}
    blocks = []
    for target in tab["target"].drop_duplicates():
        t = tab[tab["target"] == target].drop(columns=["target"]).copy()
        t = t.dropna(axis=1, how="all")
        for c, f in fmt.items():
            if c in t.columns:
                t[c] = t[c].map(lambda v: f.format(v) if pd.notna(v) else "-")
        spec = TARGETS.get(target)
        label = spec.label if spec else target
        market = "" if (spec and spec.market_column) else "   [no market benchmark exists]"
        blocks.append(f"--- {target}: {label}{market} ---\n" + t.to_string(index=False))
    out = "\n\n".join(blocks)
    return out + f"\n\nATS break-even at standard -110 juice: {breakeven:.4f}"


def by_season(preds: pd.DataFrame, cfg: Config, targets: list[str] | None = None) -> pd.DataFrame:
    """Per-season breakdown - a single aggregate hides a lot of variance."""
    breakeven = cfg.get("evaluation.ats_breakeven", 0.5238)
    rows = []
    for spec in resolve(targets):
        for (name, season), d in preds.groupby(["model", "season"]):
            m = evaluate_target(d, spec, breakeven)
            if m.get("n_games", 0) == 0:
                continue
            m["model"] = name
            m["season"] = season
            rows.append(m)
    tab = pd.DataFrame(rows)
    if tab.empty:
        return tab
    cols = ["target", "model", "season", "n_games", "accuracy", "log_loss", "mae", "ats_pct"]
    return tab[[c for c in cols if c in tab.columns]].sort_values(["target", "model", "season"])
