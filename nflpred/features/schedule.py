"""Schedule-derived context features: rest, travel, time zones, venue, weather.

Everything here is computable from the schedule before kickoff, with one
explicitly-flagged exception (``weather.mode: observed``) that exists so its
cost can be measured rather than assumed. See README.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from nflpred.config import Config
from nflpred.features.reference import (
    DIVISIONS,
    haversine_miles,
    home_venue,
    venue_coords,
)

INDOOR_ROOFS = {"dome", "closed"}


def _kickoff(games: pd.DataFrame) -> pd.Series:
    """Kickoff timestamp (US/Eastern wall clock, naive) used purely for ordering."""
    day = pd.to_datetime(games["gameday"], errors="coerce")
    time = games["gametime"].fillna("13:00").astype(str)
    parsed = pd.to_timedelta(time + ":00", errors="coerce").fillna(pd.Timedelta(hours=13))
    return day + parsed


def _impute_roof(games: pd.DataFrame) -> pd.Series:
    """Fill missing roof from the venue's modal roof across the dataset.

    Roof type is a property of the stadium, not the game, so borrowing the
    venue's usual value introduces no game-specific information.
    """
    roof = games["roof"].copy()
    modal = (
        games.dropna(subset=["roof"])
        .groupby("stadium_id")["roof"]
        .agg(lambda s: s.mode().iloc[0] if len(s.mode()) else np.nan)
    )
    fill = games["stadium_id"].map(modal)
    roof = roof.fillna(fill).fillna("outdoors")
    return roof


def build_team_games(games: pd.DataFrame) -> pd.DataFrame:
    """Long format: one row per (game, team). The backbone for all sequential features.

    Rows are sorted by kickoff so that any ``shift``/``rolling`` on a team group
    is guaranteed to look strictly backwards in time.
    """
    g = games.copy()
    g["kickoff"] = _kickoff(g)
    g["roof_filled"] = _impute_roof(g)
    vc = venue_coords(g)
    g = pd.concat([g, vc], axis=1)

    rows = []
    for side, opp_side in (("home", "away"), ("away", "home")):
        sub = pd.DataFrame(
            {
                "game_id": g["game_id"],
                "season": g["season"],
                "week": g["week"],
                "game_type": g["game_type"],
                "kickoff": g["kickoff"],
                "gameday": g["gameday"],
                "weekday": g["weekday"],
                "gametime": g["gametime"],
                "team": g[f"{side}_team"],
                "opponent": g[f"{opp_side}_team"],
                "is_home": 1 if side == "home" else 0,
                "neutral_site": (g["location"] == "Neutral").astype(int),
                "rest": g[f"{side}_rest"],
                "team_score": g[f"{side}_score"],
                "opp_score": g[f"{opp_side}_score"],
                "roof": g["roof_filled"],
                "surface": g["surface"],
                "temp_observed": g["temp"],
                "wind_observed": g["wind"],
                "div_game": g["div_game"],
                "stadium_id": g["stadium_id"],
                "venue_lat": g["venue_lat"],
                "venue_lon": g["venue_lon"],
                "venue_tz": g["venue_tz"],
                "qb_id": g[f"{side}_qb_id"],
                "qb_name": g[f"{side}_qb_name"],
                "coach": g[f"{side}_coach"],
                "spread_line": g["spread_line"],
                "total_line": g["total_line"],
            }
        )
        rows.append(sub)
    tg = pd.concat(rows, ignore_index=True)

    # Each team's own home venue for this season (origin for travel).
    origins = tg.apply(lambda r: home_venue(r["team"], int(r["season"])), axis=1, result_type="expand")
    origins.columns = ["home_lat", "home_lon", "home_tz"]
    tg = pd.concat([tg, origins], axis=1)

    tg["margin"] = tg["team_score"] - tg["opp_score"]
    tg["played"] = tg["team_score"].notna()
    tg = tg.sort_values(["kickoff", "game_id", "team"]).reset_index(drop=True)
    return tg


def travel_features(tg: pd.DataFrame, trailing_games: int = 3) -> pd.DataFrame:
    """Distance, time-zone shift, direction of travel and trailing travel load."""
    out = pd.DataFrame(index=tg.index)
    out["travel_miles"] = haversine_miles(
        tg["home_lat"], tg["home_lon"], tg["venue_lat"], tg["venue_lon"]
    )
    # A home team at its own venue travels zero; guard against float noise.
    out.loc[out["travel_miles"] < 1.0, "travel_miles"] = 0.0

    # Wrap into [-12, +12]: crossing the date line is a 6-hour shift, not 18.
    raw_tz = tg["venue_tz"] - tg["home_tz"]
    out["tz_delta"] = ((raw_tz + 12.0) % 24.0) - 12.0         # +ve = venue east of home
    out["tz_abs"] = out["tz_delta"].abs()
    out["travel_east"] = (out["tz_delta"] > 0).astype(int)
    out["travel_west"] = (out["tz_delta"] < 0).astype(int)

    # Cumulative travel over the trailing N games, strictly prior (shifted).
    grp = out.groupby(tg["team"], sort=False)["travel_miles"]
    out["travel_load_3g"] = (
        grp.shift(1).groupby(tg["team"], sort=False)
        .rolling(trailing_games, min_periods=1).sum().reset_index(level=0, drop=True)
    )
    out["travel_load_3g"] = out["travel_load_3g"].fillna(0.0)
    return out


def rest_features(tg: pd.DataFrame) -> pd.DataFrame:
    """Rest days plus the categorical patterns that matter (short week, bye, primetime)."""
    out = pd.DataFrame(index=tg.index)
    rest = pd.to_numeric(tg["rest"], errors="coerce")
    # Week 1 "rest" is an artefact of the offseason; cap so it does not dominate.
    rest = rest.where(tg["week"] > 1, 7).clip(upper=21)
    out["rest_days"] = rest
    out["short_week"] = (rest <= 4).astype(int)
    out["long_rest"] = (rest >= 10).astype(int)
    out["off_bye"] = (rest >= 13).astype(int)

    wd = tg["weekday"].astype(str)
    hour = pd.to_numeric(tg["gametime"].astype(str).str.slice(0, 2), errors="coerce").fillna(13)
    out["is_thursday"] = (wd == "Thursday").astype(int)
    out["is_monday"] = (wd == "Monday").astype(int)
    out["is_sunday"] = (wd == "Sunday").astype(int)
    out["primetime"] = (
        ((wd == "Sunday") & (hour >= 19)) | wd.isin(["Thursday", "Monday", "Friday", "Saturday"])
    ).astype(int)
    out["early_kick"] = ((wd == "Sunday") & (hour <= 13)).astype(int)
    # A west-coast body clock playing a 13:00 ET kick is the classic disadvantage.
    out["west_team_early_east_kick"] = (out["early_kick"] & (tg["home_tz"] <= -8)).astype(int)
    return out


def venue_features(tg: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=tg.index)
    out["is_home"] = tg["is_home"].astype(int)
    out["neutral_site"] = tg["neutral_site"].astype(int)
    out["div_game"] = pd.to_numeric(tg["div_game"], errors="coerce").fillna(0).astype(int)
    out["is_indoor"] = tg["roof"].isin(INDOOR_ROOFS).astype(int)
    out["is_retractable_open"] = (tg["roof"] == "open").astype(int)
    surf = tg["surface"].astype(str).str.lower()
    out["is_turf"] = (~surf.isin(["grass", "nan"])).astype(int)
    out["conference_game"] = (
        tg["team"].map(DIVISIONS).str.slice(0, 3) == tg["opponent"].map(DIVISIONS).str.slice(0, 3)
    ).astype(int)
    return out


def weather_features(tg: pd.DataFrame, mode: str = "climatology") -> pd.DataFrame:
    """Temperature and wind.

    ``observed``    uses the recorded game-time conditions. These are only
                    populated after the game and are NOT available pre-kickoff,
                    so this mode is backtest-optimistic. It exists to quantify
                    what weather is worth, not for production use.
    ``climatology`` uses indoor constants for roofed games and, for outdoor
                    games, the venue's historical mean for that month computed
                    from strictly earlier seasons. Fully leak-free and
                    available for future games.
    ``none``        drops weather entirely.
    """
    idx = tg.index
    indoor = tg["roof"].isin(INDOOR_ROOFS)
    if mode == "none":
        return pd.DataFrame(index=idx)

    if mode == "observed":
        temp = pd.to_numeric(tg["temp_observed"], errors="coerce")
        wind = pd.to_numeric(tg["wind_observed"], errors="coerce")
        temp = temp.where(~indoor, 70.0)
        wind = wind.where(~indoor, 0.0)
        return pd.DataFrame(
            {"temp": temp, "wind": wind, "wind_gt15": (wind > 15).astype(float),
             "temp_freezing": (temp <= 32).astype(float)},
            index=idx,
        )

    if mode != "climatology":
        raise ValueError(f"unknown weather mode: {mode!r}")

    # Leak-free climatology: venue x month mean from strictly earlier seasons.
    df = pd.DataFrame(
        {
            "season": tg["season"].astype(int),
            "stadium_id": tg["stadium_id"].astype(str),
            "month": pd.to_datetime(tg["gameday"]).dt.month,
            "temp": pd.to_numeric(tg["temp_observed"], errors="coerce"),
            "wind": pd.to_numeric(tg["wind_observed"], errors="coerce"),
        },
        index=idx,
    )
    obs = df[~indoor & df["temp"].notna()]
    # Every level of this lookup - venue-month, month, global - is built from
    # seasons STRICTLY BEFORE the target season. A fallback that averaged over
    # all seasons would leak future weather into past games, which the
    # truncation test in tests/test_leakage.py catches.
    key = ["stadium_id", "month"]
    seasons = sorted(df["season"].unique())
    clim_t, clim_w, fb_t, fb_w, glob_t, glob_w = {}, {}, {}, {}, {}, {}
    for s in seasons:
        prior = obs[obs["season"] < s]
        if not len(prior):
            continue
        for k, v in prior.groupby(key)["temp"].mean().items():
            clim_t[(s, *k)] = v
        for k, v in prior.groupby(key)["wind"].mean().items():
            clim_w[(s, *k)] = v
        for m, v in prior.groupby("month")["temp"].mean().items():
            fb_t[(s, m)] = v
        for m, v in prior.groupby("month")["wind"].mean().items():
            fb_w[(s, m)] = v
        glob_t[s] = float(prior["temp"].mean())
        glob_w[s] = float(prior["wind"].mean())

    # Seasons with no prior data at all (the first year in the frame) fall back
    # to fixed physical constants rather than to anything data-derived.
    NEUTRAL_TEMP, NEUTRAL_WIND = 60.0, 8.0

    keys = list(zip(df["season"], df["stadium_id"], df["month"]))
    temp = np.array(
        [clim_t.get(k, fb_t.get((k[0], k[2]), glob_t.get(k[0], NEUTRAL_TEMP))) for k in keys],
        dtype=float,
    )
    wind = np.array(
        [clim_w.get(k, fb_w.get((k[0], k[2]), glob_w.get(k[0], NEUTRAL_WIND))) for k in keys],
        dtype=float,
    )
    temp = np.where(indoor.to_numpy(), 70.0, temp)
    wind = np.where(indoor.to_numpy(), 0.0, wind)
    return pd.DataFrame(
        {"temp": temp, "wind": wind,
         "wind_gt15": (wind > 15).astype(float),
         "temp_freezing": (temp <= 32).astype(float)},
        index=idx,
    )


def context_features(tg: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """All schedule-context features for each team-game row."""
    mode = cfg.get("features.weather.mode", "climatology")
    parts = [
        rest_features(tg),
        travel_features(tg),
        venue_features(tg),
        weather_features(tg, mode=mode),
    ]
    out = pd.concat(parts, axis=1)
    out.insert(0, "game_id", tg["game_id"].to_numpy())
    out.insert(1, "team", tg["team"].to_numpy())
    return out
