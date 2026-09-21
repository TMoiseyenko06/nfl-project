"""Leakage audit.

Two tests carry the weight, and they catch different directions of leak:

``test_truncation_invariance``
    Features for a game must be bit-identical whether or not LATER games exist
    in the input. Catches season-long aggregates and any statistic pooled over
    the full dataset (a climatology fallback averaged across future seasons was
    caught by exactly this during development).

``test_own_game_outcome_does_not_affect_its_own_features``
    Features for a game must be bit-identical when THAT GAME'S OWN outcome
    changes. Catches an unshifted rolling window. Truncation invariance does
    NOT catch this, because a within-game leak is invariant to what comes
    after; both tests are required.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nflpred.config import Config
from nflpred.features.build import META_COLS, TARGET_COLS, build_features, feature_groups

CUTOFF = "2023-11-01"

# Columns that are known post-kickoff facts and must never reach a model.
BANNED_AS_FEATURES = {
    "home_score", "away_score", "spread_actual", "total_actual", "home_win",
    "result", "total", "overtime", "played",
}


@pytest.fixture(scope="module")
def cfg():
    return Config.load("configs/default.yaml")


@pytest.fixture(scope="module")
def full(cfg):
    return build_features(cfg)


@pytest.fixture(scope="module")
def truncated(cfg):
    return build_features(cfg, asof=CUTOFF)


def _feature_cols(df):
    return [c for g in feature_groups(df).values() for c in g]


def test_truncation_invariance(full, truncated):
    """Features for past games must not change when future games are added.

    This is the strongest available statement of "no data leakage": it tests
    the whole pipeline end to end rather than any single transformation.
    """
    cols = _feature_cols(full)
    a = full[full["kickoff"] < pd.Timestamp(CUTOFF)].set_index("game_id").sort_index()
    b = truncated.set_index("game_id").sort_index()
    shared = a.index.intersection(b.index)
    assert len(shared) > 1500, f"expected a substantial overlap, got {len(shared)}"

    offenders = []
    for c in cols:
        if c not in b.columns:
            continue
        x = pd.to_numeric(a.loc[shared, c], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(b.loc[shared, c], errors="coerce").to_numpy(dtype=float)
        both_nan = np.isnan(x) & np.isnan(y)
        close = np.isclose(x, y, rtol=1e-9, atol=1e-9, equal_nan=True)
        bad = int((~(close | both_nan)).sum())
        if bad:
            worst = float(np.nanmax(np.abs(x - y)))
            offenders.append((c, bad, worst))
    assert not offenders, (
        "features changed when future games were added (LEAKAGE):\n"
        + "\n".join(f"  {c}: {n} rows differ, max delta {d:.6g}" for c, n, d in offenders)
    )


def test_no_outcome_columns_in_feature_set(full):
    cols = set(_feature_cols(full))
    assert not (cols & BANNED_AS_FEATURES), f"outcome columns leaked into features: {cols & BANNED_AS_FEATURES}"


def test_feature_target_correlation_is_plausible(full):
    """No single pre-kickoff feature should explain the result almost perfectly."""
    played = full[full["played"]]
    y = played["spread_actual"].astype(float)
    suspicious = []
    for c in _feature_cols(full):
        x = pd.to_numeric(played[c], errors="coerce")
        if x.notna().sum() < 200 or x.nunique() < 3:
            continue
        r = np.corrcoef(x.fillna(x.median()), y)[0, 1]
        if abs(r) > 0.9:
            suspicious.append((c, float(r)))
    assert not suspicious, f"implausibly predictive features (likely leakage): {suspicious}"


def test_rolling_features_shift_by_one_game(cfg, full):
    """Spot-check the shift directly: a team's rolling EPA entering game g must
    be reconstructible from that team's games strictly before g."""
    from nflpred.features.epa import game_box, rolling_features
    from nflpred.features.schedule import build_team_games
    from nflpred.ingest.nflverse import load_games, load_pbp

    games = load_games(cfg)
    games = games[(games["season"] >= 2021) & (games["season"] <= 2023)].reset_index(drop=True)
    tg = build_team_games(games)
    pbp = load_pbp(cfg, seasons=[2021, 2022, 2023])
    box = game_box(pbp)
    roll = rolling_features(tg, box, halflife=6.0, windows=(8,))

    merged = tg[["game_id", "team", "kickoff"]].merge(roll, on=["game_id", "team"])
    merged = merged.merge(box[["game_id", "team", "off_epa_play"]], on=["game_id", "team"], how="left")

    checked = 0
    for team in ["KC", "BUF", "PHI", "DET"]:
        t = merged[merged["team"] == team].sort_values("kickoff").reset_index(drop=True)
        for i in range(10, min(len(t), 40)):
            expected = t["off_epa_play"].iloc[:i].shift(0).rolling(8, min_periods=2).mean().iloc[-1]
            actual = t["off_epa_play_r8"].iloc[i]
            if pd.isna(expected) and pd.isna(actual):
                continue
            assert np.isclose(expected, actual, atol=1e-9), (
                f"{team} game {i}: rolling window used the current game "
                f"(expected {expected}, got {actual})"
            )
            checked += 1
    assert checked > 50, f"test did not actually check enough rows ({checked})"


def test_elo_is_sequential(cfg):
    """Elo entering a game must not move when later results change."""
    from nflpred.features.schedule import _kickoff
    from nflpred.ingest.nflverse import load_games
    from nflpred.models.elo import run_elo

    games = load_games(cfg)
    games = games[(games["season"] >= 2018) & (games["season"] <= 2022)].copy()
    games["kickoff"] = _kickoff(games)
    games = games.sort_values("kickoff").reset_index(drop=True)

    base = run_elo(games).set_index("game_id")
    perturbed_games = games.copy()
    tail = perturbed_games.index[-200:]
    perturbed_games.loc[tail, "result"] = -perturbed_games.loc[tail, "result"]
    perturbed = run_elo(perturbed_games).set_index("game_id")

    untouched = games.loc[games.index[:-200], "game_id"]
    a = base.loc[untouched, "elo_diff"].to_numpy()
    b = perturbed.loc[untouched, "elo_diff"].to_numpy()
    assert np.allclose(a, b, atol=1e-9), "Elo ratings for earlier games changed when later results changed"


def test_no_future_games_in_training_targets(full):
    """Unplayed games must carry NaN targets so they can never be trained on."""
    unplayed = full[~full["played"]]
    for c in TARGET_COLS:
        assert unplayed[c].isna().all(), f"unplayed games have non-null {c}"


def test_meta_and_target_columns_present(full):
    for c in META_COLS + TARGET_COLS:
        assert c in full.columns, f"missing expected column {c}"


def test_own_game_outcome_does_not_affect_its_own_features(cfg):
    """A game's features must not respond to that game's own result.

    Corrupting a game legitimately changes the features of games that come
    AFTER it, so the corrupted set must contain no game that precedes another
    one being checked. We therefore corrupt only games sharing a single exact
    kickoff timestamp - a Sunday 13:00 slate - which are mutually simultaneous,
    and then assert their own features are untouched.
    """
    from nflpred.features.adjust import opponent_adjusted_ratings
    from nflpred.features.epa import game_box, rolling_features
    from nflpred.features.schedule import build_team_games
    from nflpred.ingest.nflverse import load_games, load_pbp

    seasons = [2021, 2022, 2023]
    games = load_games(cfg)
    games = games[games["season"].isin(seasons)].reset_index(drop=True)
    tg = build_team_games(games)
    pbp = load_pbp(cfg, seasons=seasons)

    def pipeline(frame):
        box = game_box(frame)
        roll = rolling_features(tg, box, halflife=6.0, windows=(8,))
        adj = opponent_adjusted_ratings(tg, box, alpha=12.0, halflife_days=400.0, min_games=64)
        return (
            roll.merge(adj, on=["game_id", "team"], how="left")
            .set_index(["game_id", "team"]).sort_index()
        )

    base = pipeline(pbp)
    total_checked = 0

    for season, week in [(2021, 12), (2022, 9), (2023, 14)]:
        slate = tg[(tg["season"] == season) & (tg["week"] == week)]
        modal_kick = slate["kickoff"].mode().iloc[0]
        target = sorted(set(slate.loc[slate["kickoff"] == modal_kick, "game_id"]))
        assert len(target) >= 5, f"{season} wk{week}: only {len(target)} simultaneous games"

        corrupt = pbp.copy()
        mask = corrupt["game_id"].isin(target)
        assert mask.sum() > 0, "no play-by-play matched the target games"
        corrupt.loc[mask, "epa"] = corrupt.loc[mask, "epa"] * -3.0 + 5.0
        corrupt.loc[mask, "success"] = 1.0 - corrupt.loc[mask, "success"].fillna(0)
        after = pipeline(corrupt)

        rows = [i for i in base.index if i[0] in set(target)]
        assert len(rows) >= 10
        offenders = []
        for c in base.columns:
            x = pd.to_numeric(base.loc[rows, c], errors="coerce").to_numpy(dtype=float)
            y = pd.to_numeric(after.loc[rows, c], errors="coerce").to_numpy(dtype=float)
            both_nan = np.isnan(x) & np.isnan(y)
            ok = np.isclose(x, y, rtol=1e-9, atol=1e-9, equal_nan=True) | both_nan
            if not ok.all():
                offenders.append((c, int((~ok).sum())))
        assert not offenders, (
            f"{season} wk{week}: a game's own outcome changed its own features "
            f"(LEAKAGE): {offenders}"
        )
        total_checked += len(rows)

    assert total_checked > 40, f"test checked too few rows ({total_checked})"
