"""Generate predictions for an upcoming week and log them before kickoff."""
from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from nflpred.config import Config
from nflpred.features.build import build_features, select_features
from nflpred.ingest.nflverse import cache_vintage
from nflpred.models.registry import default_phase1_models, neural_models
from nflpred.predict.log import append_predictions

log = logging.getLogger(__name__)


def all_models(cfg, targets: list[str] | None = None) -> list:
    """Every model available for prediction, including the neural net when
    torch is installed. torch is optional, so its absence is not an error."""
    models = default_phase1_models(cfg, targets=targets)
    try:
        models += neural_models(cfg, targets=targets)
    except ImportError:
        log.info("torch not installed; skipping neural models "
                 "(install with `make setup-neural`)")
    return models


def eastern_now() -> pd.Timestamp:
    """Current time as a US/Eastern wall clock, matching how kickoffs are stored."""
    now = pd.Timestamp.now(tz="UTC")
    try:
        return now.tz_convert("US/Eastern").tz_localize(None)
    except Exception:  # noqa: BLE001 - missing tzdata must not break prediction
        # Assume EDT. This places "now" LATER in Eastern wall-clock terms than
        # EST would, so a game near the boundary is treated as already started
        # rather than still upcoming - the safe direction.
        log.warning("time-zone database unavailable; assuming EDT")
        return (now - pd.Timedelta(hours=4)).tz_localize(None)


def upcoming_games(df: pd.DataFrame, now: pd.Timestamp | None = None) -> tuple[int, int, pd.DataFrame]:
    """The next slate to predict: (season, week, games).

    "Upcoming" means kickoff is still in the future, not merely "unplayed".
    A week that is already underway - Thursday night done, Sunday still to come -
    yields only its remaining games, so the system can never log a prediction
    for a game that has already started. Predicting a completed game is a
    retrodiction, and retrodictions do not belong in a track record.
    """
    now = now if now is not None else eastern_now()
    future = df[(~df["played"]) & (df["kickoff"] > now)].sort_values("kickoff")
    if future.empty:
        raise RuntimeError("no upcoming games in the schedule - nothing to predict")
    first = future.iloc[0]
    season, week = int(first["season"]), int(first["week"])
    slate = future[(future["season"] == season) & (future["week"] == week)]
    return season, week, slate


def train_production_models(cfg: Config, df: pd.DataFrame, asof: pd.Timestamp,
                            models: list | None = None) -> list:
    """Fit every model on all completed games that kicked off before ``asof``."""
    models = models if models is not None else all_models(cfg)
    include_post = cfg.get("data.include_postseason_in_training", True)
    train = df[df["played"] & (df["kickoff"] < asof)]
    if not include_post:
        train = train[train["game_type"] == "REG"]
    if train.empty:
        raise RuntimeError("no training data available before %s" % asof)
    log.info("training on %d completed games (through %s)", len(train), train["kickoff"].max())
    for m in models:
        feats = select_features(df, m.groups) if m.groups else []
        m.fit(train, feats)
    return models


def predict_week(
    cfg: Config,
    season: int | None = None,
    week: int | None = None,
    models: list | None = None,
    write_log: bool = True,
) -> pd.DataFrame:
    """Predict a week's games and append the predictions to the log."""
    df = build_features(cfg)
    if season is None or week is None:
        season, week, target = upcoming_games(df)
    else:
        target = df[(df["season"] == season) & (df["week"] == week)]
        if target.empty:
            raise RuntimeError(f"no games found for {season} week {week}")

    asof = target["kickoff"].min()
    models = train_production_models(cfg, df, asof, models=models)

    rows = []
    for m in models:
        feats = select_features(df, m.groups) if m.groups else []
        p = m.predict(target, feats)
        rec = pd.DataFrame(
            {
                "model": m.name,
                "game_id": target["game_id"].to_numpy(),
                "season": season,
                "week": week,
                "kickoff": target["kickoff"].to_numpy(),
                "home_team": target["home_team"].to_numpy(),
                "away_team": target["away_team"].to_numpy(),
                "vegas_spread": target["vegas_spread"].to_numpy(),
                "vegas_total": target["vegas_total"].to_numpy(),
            }
        )
        for col, arr in p.values.items():
            rec[col] = arr
        rows.append(rec)

    out = pd.concat(rows, ignore_index=True)
    if write_log:
        append_predictions(out, cfg.get("paths.prediction_log"), vintage=cache_vintage(cfg))
    return out


def save_models(models: list, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(models, fh)
    log.info("saved %d model(s) -> %s", len(models), path)


def load_models(path: str | Path) -> list:
    with open(path, "rb") as fh:
        return pickle.load(fh)


def format_week(preds: pd.DataFrame, model: str) -> str:
    """The prediction: who wins, how confident, and the projected score.

    Team scores are derived from the two things the model actually predicts -
    the margin and the total - since home = (total + margin) / 2 and
    away = (total - margin) / 2.
    """
    d = preds[preds["model"] == model].copy()
    if d.empty:
        return f"(no predictions for model {model!r})"

    total = d["pred_total"]
    margin = d["pred_spread"]
    home_pts = (total + margin) / 2.0
    away_pts = (total - margin) / 2.0

    home_wins = d["p_home"] > 0.5
    out = pd.DataFrame({
        "matchup": d["away_team"] + " @ " + d["home_team"],
        "WINNER": np.where(home_wins, d["home_team"], d["away_team"]),
        "win%": np.maximum(d["p_home"], 1 - d["p_home"]).map("{:.0%}".format),
        "confidence": np.maximum(d["p_home"], 1 - d["p_home"]).map(_confidence_word),
        "projected score": [
            f"{h} {hp:.0f} - {ap:.0f} {a}" if hw else f"{a} {ap:.0f} - {hp:.0f} {h}"
            for h, a, hp, ap, hw in zip(d["home_team"], d["away_team"], home_pts, away_pts, home_wins)
        ],
        "total": total.map("{:.0f}".format),
    })

    h1 = ""
    if "pred_h1_total" in d.columns:
        h1_home = (d["pred_h1_total"] + d["pred_h1_spread"]) / 2.0
        h1_away = (d["pred_h1_total"] - d["pred_h1_spread"]) / 2.0
        out["H1 LEADER"] = np.where(h1_home >= h1_away, d["home_team"], d["away_team"])
        out["H1 win%"] = np.maximum(d["p_home_h1"], 1 - d["p_home_h1"]).map("{:.0%}".format)
        # Leader first, matching the full-game column.
        out["at half"] = [
            f"{h} {hp:.0f} - {ap:.0f} {a}" if hp >= ap else f"{a} {ap:.0f} - {hp:.0f} {h}"
            for h, a, hp, ap in zip(d["home_team"], d["away_team"], h1_home, h1_away)
        ]
    return out.to_string(index=False) + h1


def _confidence_word(p: float) -> str:
    """Plain-language confidence. The model is well calibrated, so these mean
    what they say: 'likely' picks really do win about 65% of the time."""
    if p >= 0.75:
        return "strong"
    if p >= 0.65:
        return "likely"
    if p >= 0.57:
        return "lean"
    return "coin flip"
