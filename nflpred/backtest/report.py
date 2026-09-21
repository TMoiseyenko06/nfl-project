"""Reporting: calibration, feature importance and SHAP attributions.

SHAP values come from LightGBM's own ``pred_contrib=True``, which is exact
TreeSHAP - no extra dependency, and no sampling approximation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from nflpred.backtest.metrics import calibration_table, expected_calibration_error


def calibration_report(preds: pd.DataFrame, models: list[str] | None = None,
                       bins: int = 10) -> pd.DataFrame:
    """Calibration curve per model: are 70% predictions right 70% of the time?"""
    models = models or list(dict.fromkeys(preds["model"]))
    frames = []
    for name in models:
        d = preds[(preds["model"] == name) & preds["home_win"].isin([0.0, 1.0])]
        if d.empty:
            continue
        t = calibration_table(d["home_win"], d["p_home"], bins=bins)
        if t.empty:
            continue
        t.insert(0, "model", name)
        frames.append(t)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def format_calibration(tab: pd.DataFrame, model: str) -> str:
    d = tab[tab["model"] == model]
    if d.empty:
        return f"(no calibration data for {model})"
    lines = [f"{'bin':<12}{'n':>6}{'predicted':>12}{'actual':>10}{'gap':>9}"]
    for r in d.itertuples(index=False):
        lines.append(
            f"{r.bin:<12}{r.n:>6}{r.mean_pred:>12.3f}{r.actual:>10.3f}{r.gap:>+9.3f}"
        )
    return "\n".join(lines)


def shap_summary(model, X: pd.DataFrame, features: list[str], top: int = 20) -> pd.DataFrame:
    """Mean |SHAP| per feature for a fitted LightGBM classifier."""
    booster = getattr(model, "clf_", None)
    if booster is None or not hasattr(booster, "booster_"):
        return pd.DataFrame()
    contrib = booster.booster_.predict(X[features].to_numpy(), pred_contrib=True)
    vals = np.abs(contrib[:, :-1]).mean(axis=0)     # last column is the base value
    return (
        pd.DataFrame({"feature": features, "mean_abs_shap": vals})
        .sort_values("mean_abs_shap", ascending=False)
        .head(top)
        .reset_index(drop=True)
    )


def importance_report(cfg, df: pd.DataFrame, model_name: str = "lightgbm",
                      top: int = 20) -> dict[str, pd.DataFrame]:
    """Fit the named model on all completed games and report what it leans on."""
    from nflpred.features.build import select_features
    from nflpred.models.registry import default_phase1_models

    models = {m.name: m for m in default_phase1_models(cfg)}
    if model_name not in models:
        raise KeyError(f"unknown model {model_name!r}")
    model = models[model_name]
    feats = select_features(df, model.groups)
    train = df[df["played"]]
    model.fit(train, feats)

    out = {}
    imp = model.feature_importance()
    if imp is not None:
        out["gain"] = imp.head(top).rename("gain").reset_index().rename(columns={"index": "feature"})
    sh = shap_summary(model, train, feats, top=top)
    if not sh.empty:
        out["shap"] = sh
    return out


def weekly_track_record(perf: pd.DataFrame, model: str) -> str:
    """Running performance log for one model, week by week."""
    d = perf[perf["model"] == model].sort_values(["season", "week"])
    if d.empty:
        return f"(no scored predictions for {model})"
    cols = [c for c in ["season", "week", "n_games", "accuracy", "log_loss",
                        "brier", "spread_mae", "ats_pct"] if c in d.columns]
    return d[cols].to_string(index=False)
