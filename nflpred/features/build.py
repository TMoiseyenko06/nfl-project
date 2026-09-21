"""Assemble the modelling table: one row per game, all features pre-kickoff.

Naming convention encodes provenance, which makes the leakage audit tractable:
  ctx_*    game-level context (same for both teams)
  d_*      home-minus-away difference of a team-level feature
  mu_*     explicit matchup terms (one team's offence vs the other's defence)
  elo_*    Elo ratings entering the game
  vegas_*  market inputs (opt-in feature group; always available as a benchmark)
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from nflpred.config import Config
from nflpred.features.adjust import opponent_adjusted_ratings
from nflpred.features.epa import METRICS, game_box, rolling_features
from nflpred.features.schedule import build_team_games, context_features
from nflpred.features.quarterback import load_qb_weeks, qb_features
from nflpred.features.targets import attach_halftime_targets
from nflpred.ingest.nflverse import cache_vintage, load_games, load_pbp
from nflpred.models.elo import run_elo

log = logging.getLogger(__name__)

# Team-level context columns that get differenced home-minus-away.
CTX_DIFF_COLS = [
    "rest_days", "short_week", "long_rest", "off_bye",
    "travel_miles", "tz_delta", "tz_abs", "travel_load_3g",
    "west_team_early_east_kick",
]
# Game-level context columns taken from the home row (identical on both rows).
CTX_GAME_COLS = [
    "div_game", "conference_game", "neutral_site", "is_indoor",
    "is_retractable_open", "is_turf", "primetime", "is_thursday", "is_monday",
    "temp", "wind", "wind_gt15", "temp_freezing",
]
# Rolling metrics differenced home-minus-away. EWMA for all; r8 only for the core few.
ROLL_EWMA = [f"{m}_ewma" for m in METRICS]
ROLL_WINDOW = ["off_epa_play_r8", "def_epa_play_r8", "net_epa_r8", "off_success_rate_r8"]
ROLL_EXTRA = ["games_this_season", "margin_r5"]
# Metrics summed home+away rather than differenced. A difference cancels out
# exactly what a TOTAL needs to know - that both teams score a lot, or that
# both play fast. Every team feature was a difference until this was added,
# which is why the totals model was no better than guessing the league mean.
SUM_COLS = [
    "off_epa_play_ewma", "def_epa_play_ewma",
    "off_success_rate_ewma", "off_explosive_rate_ewma",
    "off_pass_epa_ewma", "off_proe_ewma",
    "off_plays_ewma", "def_plays_ewma",
    "points_for_ewma", "points_against_ewma",
    "points_for_r8", "points_against_r8",
]
QB_COLS = ["qb_epa_ewma", "qb_starts", "qb_is_new", "qb_is_rookie_ish", "qb_epa_delta"]
ADJ_COLS = [
    "adj_off_epa_play_off", "adj_off_epa_play_def", "adj_off_epa_play_net",
    "adj_off_success_rate_net",
]


def _pivot(team_level: pd.DataFrame, tg: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Split a team-level frame into home/away halves keyed by game_id."""
    side = tg[["game_id", "team", "is_home"]].rename(columns={"is_home": "_side"})
    merged = side.merge(team_level, on=["game_id", "team"], how="left")
    present = [c for c in cols if c in merged.columns]
    home = merged[merged["_side"] == 1].set_index("game_id")[present]
    away = merged[merged["_side"] == 0].set_index("game_id")[present]
    return home, away


def build_features(cfg: Config, force: bool = False, asof: pd.Timestamp | str | None = None) -> pd.DataFrame:
    """Build (or load from cache) the full game-level feature table.

    ``asof`` truncates every input (schedule and play-by-play) to games that
    kicked off strictly before that timestamp. It exists for the leakage audit:
    features for a given game must be identical whether or not later games are
    present in the inputs.
    """
    if asof is not None:
        return _build_uncached(cfg, pd.Timestamp(asof))
    cache_dir = cfg.feature_cache
    cache_dir.mkdir(parents=True, exist_ok=True)
    vintage = cache_vintage(cfg)
    mode = cfg.get("features.weather.mode", "climatology")
    cache_path = cache_dir / f"features_{vintage}_{mode}.parquet"
    if cache_path.exists() and not force:
        log.info("loading cached features: %s", cache_path.name)
        return pd.read_parquet(cache_path)

    log.info("building features (vintage=%s, weather=%s)", vintage, mode)
    out = _build_uncached(cfg, None)
    out.attrs["vintage"] = vintage
    out.to_parquet(cache_path, index=False)
    log.info("wrote %s rows=%d cols=%d", cache_path.name, len(out), out.shape[1])
    return out


def _build_uncached(cfg: Config, asof: pd.Timestamp | None) -> pd.DataFrame:
    games_all = load_games(cfg)
    start, end = cfg.get("data.start_season"), cfg.get("data.end_season")
    warm = cfg.get("data.pbp_warmup_seasons", 1)
    games = games_all[
        (games_all["season"] >= start - warm) & (games_all["season"] <= end)
    ].reset_index(drop=True)

    tg = build_team_games(games)
    if asof is not None:
        keep = set(tg.loc[tg["kickoff"] < asof, "game_id"])
        tg = tg[tg["game_id"].isin(keep)].reset_index(drop=True)
        games = games[games["game_id"].isin(keep)].reset_index(drop=True)
    ctx = context_features(tg, cfg)

    pbp = load_pbp(cfg)
    if asof is not None:
        pbp = pbp[pbp["game_id"].isin(keep)]
    box = game_box(pbp, wp_filter=True)
    roll = rolling_features(
        tg, box,
        halflife=cfg.get("features.ewma_halflife", 6.0),
        windows=tuple(w for w in cfg.get("features.rolling_windows", [8]) if w == 8) or (8,),
    )
    roll = roll.merge(tg[["game_id", "team", "margin"]], on=["game_id", "team"], how="left")

    if cfg.get("features.opponent_adjust.enabled", True):
        adj = opponent_adjusted_ratings(
            tg, box,
            alpha=cfg.get("features.opponent_adjust.ridge_alpha"),
            halflife_days=cfg.get("features.opponent_adjust.recency_halflife_days"),
            min_games=cfg.get("features.opponent_adjust.min_games_required"),
        )
    else:
        adj = pd.DataFrame({"game_id": [], "team": []})

    qbf = qb_features(tg, load_qb_weeks(cfg))
    team_level = (
        roll.merge(adj, on=["game_id", "team"], how="left")
        .merge(ctx, on=["game_id", "team"], how="left")
        .merge(qbf, on=["game_id", "team"], how="left")
    )

    # --- game-level frame ---------------------------------------------------
    gk = tg[tg.is_home == 1].set_index("game_id")
    out = pd.DataFrame(index=gk.index)
    out["season"] = gk["season"]
    out["week"] = gk["week"]
    out["game_type"] = gk["game_type"]
    out["kickoff"] = gk["kickoff"]
    out["gameday"] = gk["gameday"]
    out["home_team"] = gk["team"]
    out["away_team"] = gk["opponent"]
    out["home_qb_id"] = gk["qb_id"]
    out["away_qb_id"] = tg[tg.is_home == 0].set_index("game_id")["qb_id"]

    # Targets.
    out["home_score"] = gk["team_score"]
    out["away_score"] = gk["opp_score"]
    out["spread_actual"] = gk["team_score"] - gk["opp_score"]     # home margin
    out["total_actual"] = gk["team_score"] + gk["opp_score"]
    out["home_win"] = np.where(
        out["spread_actual"].isna(), np.nan, (out["spread_actual"] > 0).astype(float)
    )
    out.loc[out["spread_actual"] == 0, "home_win"] = 0.5          # ties, ~0.3% of games
    out["played"] = out["spread_actual"].notna()

    # Halftime targets, derived from play-by-play (see features/targets.py).
    out = out.reset_index()
    out = attach_halftime_targets(out, pbp).set_index("game_id")

    # Market.
    out["vegas_spread"] = gk["spread_line"]
    out["vegas_total"] = gk["total_line"]

    # Context.
    ctx_home, ctx_away = _pivot(ctx, tg, CTX_DIFF_COLS + CTX_GAME_COLS)
    for c in CTX_GAME_COLS:
        if c in ctx_home.columns:
            out[f"ctx_{c}"] = ctx_home[c]
    for c in CTX_DIFF_COLS:
        if c in ctx_home.columns:
            out[f"d_{c}"] = ctx_home[c] - ctx_away[c]
    out["ctx_week"] = out["week"]

    # Rolling form + adjusted ratings, differenced.
    roll_cols = ROLL_EWMA + ROLL_WINDOW + ROLL_EXTRA + ADJ_COLS + QB_COLS
    r_home, r_away = _pivot(team_level, tg, sorted(set(roll_cols + SUM_COLS)))
    for c in roll_cols:
        if c in r_home.columns:
            out[f"d_{c}"] = r_home[c] - r_away[c]

    # Summed features, for the totals target.
    for c in SUM_COLS:
        if c in r_home.columns:
            out[f"s_{c}"] = r_home[c] + r_away[c]

    # Direct estimate of the game total: what each offence usually scores,
    # blended with what the opposing defence usually allows.
    if "points_for_ewma" in r_home.columns:
        home_exp = (r_home["points_for_ewma"] + r_away["points_against_ewma"]) / 2.0
        away_exp = (r_away["points_for_ewma"] + r_home["points_against_ewma"]) / 2.0
        out["s_expected_total"] = home_exp + away_exp
        out["d_expected_margin"] = home_exp - away_exp
    # Expected possessions: pace of both teams combined.
    if "off_plays_ewma" in r_home.columns:
        out["s_expected_plays"] = (
            (r_home["off_plays_ewma"] + r_away["def_plays_ewma"]) / 2.0
            + (r_away["off_plays_ewma"] + r_home["def_plays_ewma"]) / 2.0
        )

    # Explicit matchup terms: each offence against the other defence.
    def mu(off_col, def_col, name):
        if off_col in r_home.columns and def_col in r_away.columns:
            out[f"mu_home_{name}"] = r_home[off_col] - r_away[def_col]
            out[f"mu_away_{name}"] = r_away[off_col] - r_home[def_col]
            out[f"mu_net_{name}"] = out[f"mu_home_{name}"] - out[f"mu_away_{name}"]

    mu("off_epa_play_ewma", "def_epa_play_ewma", "epa")
    mu("off_pass_epa_ewma", "def_pass_epa_ewma", "pass_epa")
    mu("off_rush_epa_ewma", "def_rush_epa_ewma", "rush_epa")
    mu("off_success_rate_ewma", "def_success_rate_ewma", "success")

    # Elo. run_elo needs kickoff to order the schedule; take it from the home
    # rows of the team-game table so both use exactly the same timestamps.
    kickoff_by_game = tg[tg["is_home"] == 1].set_index("game_id")["kickoff"]
    games_for_elo = games.assign(
        kickoff=games["game_id"].map(kickoff_by_game).to_numpy()
    )
    elo = run_elo(
        games_for_elo,
        k=cfg.get("features.elo.k"),
        home_field=cfg.get("features.elo.home_field"),
        season_regression=cfg.get("features.elo.season_regression"),
        initial=cfg.get("features.elo.initial_rating"),
        mov_multiplier=cfg.get("features.elo.mov_multiplier", True),
    ).set_index("game_id")
    elo_cols = ["elo_diff", "elo_prob_home", "elo_spread_home"]
    out = pd.concat([out, elo[elo_cols].reindex(out.index)], axis=1).copy()

    out = out.reset_index().rename(columns={"index": "game_id"})
    out = out[out["season"] >= start].sort_values("kickoff").reset_index(drop=True)
    return out


# --- feature group registry -------------------------------------------------
META_COLS = [
    "game_id", "season", "week", "game_type", "kickoff", "gameday",
    "home_team", "away_team", "home_qb_id", "away_qb_id", "played",
]
TARGET_COLS = [
    "home_score", "away_score", "spread_actual", "total_actual", "home_win",
    "h1_home_score", "h1_away_score", "h1_spread_actual", "h1_total_actual", "h1_home_win",
]


# The lean feature set. Six features that matched or beat the full eighty in
# the overfitting audit (tools/overfit_audit.py). One term each for: team
# strength, opponent-adjusted quality, recent efficiency, rest, expected
# scoring, and recent margin. At ~3,000 games, more than this fits noise.
LEAN_FEATURES = [
    "elo_diff",
    "d_adj_off_epa_play_net",
    "d_net_epa_ewma",
    "d_rest_days",
    "s_expected_total",
    "d_margin_r5",
]


def feature_groups(df: pd.DataFrame) -> dict[str, list[str]]:
    """Named column groups so models and ablations can select without hardcoding."""
    cols = list(df.columns)
    return {
        "context": [c for c in cols if c.startswith("ctx_") or c.startswith("d_rest")
                    or c.startswith("d_short") or c.startswith("d_long") or c.startswith("d_off_bye")
                    or c.startswith("d_travel") or c.startswith("d_tz") or c.startswith("d_west")],
        "form": [c for c in cols if c.startswith("d_off_") or c.startswith("d_def_")
                 or c.startswith("d_net_") or c in ("d_margin_r5", "d_games_this_season")],
        "adjusted": [c for c in cols if c.startswith("d_adj_")],
        "matchup": [c for c in cols if c.startswith("mu_")],
        "elo": [c for c in cols if c.startswith("elo_")],
        # Summed (not differenced) features. Only these carry information about
        # how much scoring a game will contain, as opposed to who wins it.
        "totals": [c for c in cols if c.startswith("s_")],
        "vegas": [c for c in cols if c.startswith("vegas_")],
        "lean": [c for c in LEAN_FEATURES if c in cols],
        # Quarterback change and quality - the biggest single gap between this
        # model and the market. Encodes CHANGE, not identity; see quarterback.py.
        "qb": [f"d_{c}" for c in QB_COLS if f"d_{c}" in cols],
        # The audited six plus quarterback change and quality, kept as its own
        # group so the QB contribution is measured rather than assumed.
        "lean_qb": [c for c in LEAN_FEATURES + ["d_qb_epa_ewma", "d_qb_is_new"] if c in cols],
    }


def select_features(df: pd.DataFrame, groups: list[str]) -> list[str]:
    reg = feature_groups(df)
    out: list[str] = []
    for g in groups:
        if g not in reg:
            raise KeyError(f"unknown feature group {g!r}; have {sorted(reg)}")
        out.extend(reg[g])
    # d_off_bye is context, not form; dedupe preserving order.
    seen, uniq = set(), []
    for c in out:
        if c not in seen:
            seen.add(c)
            uniq.append(c)
    return uniq
