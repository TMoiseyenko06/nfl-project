"""Unit tests for individual feature computations, on synthetic data where the
correct answer is known by hand."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nflpred.features.epa import game_box, rolling_features
from nflpred.features.reference import haversine_miles, home_venue, venue_coords
from nflpred.features.schedule import (
    build_team_games,
    rest_features,
    travel_features,
    weather_features,
)
from nflpred.models.elo import expected_score, run_elo


# --- geography --------------------------------------------------------------
def test_haversine_known_distances():
    # Published great-circle distances, +-1%.
    sea = home_venue("SEA", 2023)
    mia = home_venue("MIA", 2023)
    d = float(haversine_miles(sea[0], sea[1], mia[0], mia[1]))
    assert 2650 < d < 2800, d

    gb = home_venue("GB", 2023)
    chi = home_venue("CHI", 2023)
    d2 = float(haversine_miles(gb[0], gb[1], chi[0], chi[1]))
    assert 160 < d2 < 210, d2


def test_haversine_is_zero_for_identical_points():
    assert float(haversine_miles(40.0, -75.0, 40.0, -75.0)) == pytest.approx(0.0, abs=1e-9)


def test_shared_stadium_has_zero_distance():
    a, b = home_venue("NYG", 2023), home_venue("NYJ", 2023)
    assert float(haversine_miles(a[0], a[1], b[0], b[1])) == pytest.approx(0.0, abs=1e-6)


def test_relocated_franchise_uses_era_correct_venue():
    """The Rams played in the Coliseum 2016-19 and SoFi from 2020."""
    coliseum = home_venue("LA", 2018)
    sofi = home_venue("LA", 2022)
    assert coliseum != sofi
    assert float(haversine_miles(coliseum[0], coliseum[1], sofi[0], sofi[1])) > 5


# --- travel / time zones ----------------------------------------------------
def _tg(rows):
    return pd.DataFrame(rows)


def test_timezone_delta_wraps_across_the_date_line():
    """A US west-coast team playing in Melbourne shifts 6 hours, not 18."""
    tg = _tg([{"team": "SF", "home_lat": 37.4, "home_lon": -121.9, "home_tz": -8.0,
               "venue_lat": -37.8, "venue_lon": 144.9, "venue_tz": 10.0}])
    out = travel_features(tg)
    assert abs(out["tz_delta"].iloc[0]) <= 12
    assert out["tz_delta"].iloc[0] == pytest.approx(-6.0)


def test_home_team_travels_zero_miles():
    tg = _tg([{"team": "GB", "home_lat": 44.5013, "home_lon": -88.0622, "home_tz": -6.0,
               "venue_lat": 44.5013, "venue_lon": -88.0622, "venue_tz": -6.0}])
    assert travel_features(tg)["travel_miles"].iloc[0] == 0.0


def test_travel_load_excludes_the_current_game():
    """Trailing travel load must be shifted: the current trip does not count."""
    tg = _tg([
        {"team": "SEA", "home_lat": 47.6, "home_lon": -122.3, "home_tz": -8.0,
         "venue_lat": 47.6, "venue_lon": -122.3, "venue_tz": -8.0},
        {"team": "SEA", "home_lat": 47.6, "home_lon": -122.3, "home_tz": -8.0,
         "venue_lat": 25.9, "venue_lon": -80.2, "venue_tz": -5.0},
        {"team": "SEA", "home_lat": 47.6, "home_lon": -122.3, "home_tz": -8.0,
         "venue_lat": 42.1, "venue_lon": -71.3, "venue_tz": -5.0},
    ])
    out = travel_features(tg)
    assert out["travel_load_3g"].iloc[0] == 0.0          # nothing before game 1
    assert out["travel_load_3g"].iloc[1] == 0.0          # game 1 was at home
    # Entering game 3 the team has flown to Miami once and nowhere else.
    assert out["travel_load_3g"].iloc[2] == pytest.approx(out["travel_miles"].iloc[1])


# --- rest -------------------------------------------------------------------
def test_rest_flags():
    tg = _tg([
        {"rest": 4, "week": 5, "weekday": "Thursday", "gametime": "20:15", "home_tz": -5.0},
        {"rest": 7, "week": 5, "weekday": "Sunday", "gametime": "13:00", "home_tz": -5.0},
        {"rest": 14, "week": 9, "weekday": "Sunday", "gametime": "16:25", "home_tz": -5.0},
        {"rest": 99, "week": 1, "weekday": "Sunday", "gametime": "13:00", "home_tz": -8.0},
    ])
    out = rest_features(tg)
    assert out["short_week"].tolist() == [1, 0, 0, 0]
    assert out["off_bye"].tolist() == [0, 0, 1, 0]
    assert out["is_thursday"].tolist() == [1, 0, 0, 0]
    assert out["primetime"].tolist() == [1, 0, 0, 0]
    # Week 1 "rest" is an offseason artefact and is normalised to 7.
    assert out["rest_days"].iloc[3] == 7
    # West-coast team in a 13:00 ET window.
    assert out["west_team_early_east_kick"].iloc[3] == 1


# --- weather ----------------------------------------------------------------
def test_indoor_games_get_fixed_conditions():
    tg = _tg([
        {"roof": "dome", "season": 2022, "stadium_id": "DET00", "gameday": "2022-12-01",
         "temp_observed": np.nan, "wind_observed": np.nan},
        {"roof": "closed", "season": 2022, "stadium_id": "HOU00", "gameday": "2022-12-01",
         "temp_observed": np.nan, "wind_observed": np.nan},
    ])
    for mode in ("observed", "climatology"):
        out = weather_features(tg, mode=mode)
        assert (out["temp"] == 70.0).all(), mode
        assert (out["wind"] == 0.0).all(), mode


def test_weather_mode_none_drops_columns():
    tg = _tg([{"roof": "outdoors", "season": 2022, "stadium_id": "GNB00",
               "gameday": "2022-12-01", "temp_observed": 15.0, "wind_observed": 12.0}])
    assert weather_features(tg, mode="none").shape[1] == 0


def test_unknown_weather_mode_raises():
    tg = _tg([{"roof": "outdoors", "season": 2022, "stadium_id": "GNB00",
               "gameday": "2022-12-01", "temp_observed": 15.0, "wind_observed": 12.0}])
    with pytest.raises(ValueError):
        weather_features(tg, mode="sunshine")


# --- play-by-play aggregation ----------------------------------------------
def _synthetic_pbp():
    """Two teams, one game, hand-checkable EPA."""
    return pd.DataFrame({
        "game_id": ["G1"] * 6,
        "season": [2022] * 6,
        "week": [1] * 6,
        "season_type": ["REG"] * 6,
        "posteam": ["A", "A", "A", "B", "B", "B"],
        "defteam": ["B", "B", "B", "A", "A", "A"],
        "play_type": ["pass", "run", "pass", "run", "pass", "run"],
        "epa": [1.0, -1.0, 3.0, 0.0, 2.0, -2.0],
        "success": [1, 0, 1, 0, 1, 0],
        "pass": [1, 0, 1, 0, 1, 0],
        "rush": [0, 1, 0, 1, 0, 1],
        "down": [1, 2, 3, 1, 2, 1],
        "ydstogo": [10] * 6,
        "yards_gained": [25, -2, 30, 3, 8, 1],
        "xpass": [0.6, 0.5, 0.8, 0.4, 0.5, 0.3],
        "wp": [0.5] * 6,
        "sack": [0, 0, 0, 0, 0, 0],
        "interception": [0] * 6,
        "fumble_lost": [0] * 6,
    })


def test_game_box_offense_and_defense_are_mirrored():
    box = game_box(_synthetic_pbp(), wp_filter=True).set_index("team")
    # A's offence: mean of 1, -1, 3 = 1.0
    assert box.loc["A", "off_epa_play"] == pytest.approx(1.0)
    # B's defence allowed exactly what A's offence produced.
    assert box.loc["B", "def_epa_play"] == pytest.approx(1.0)
    # B's offence: mean of 0, 2, -2 = 0.0
    assert box.loc["B", "off_epa_play"] == pytest.approx(0.0)
    assert box.loc["A", "def_epa_play"] == pytest.approx(0.0)
    assert box.loc["A", "net_epa"] == pytest.approx(1.0)


def test_game_box_explosive_and_rates():
    box = game_box(_synthetic_pbp()).set_index("team")
    # A: 25-yd pass (>=20) and 30-yd pass are explosive; -2 run is not. 2/3.
    assert box.loc["A", "off_explosive_rate"] == pytest.approx(2 / 3)
    assert box.loc["A", "off_pass_rate"] == pytest.approx(2 / 3)
    assert box.loc["A", "off_success_rate"] == pytest.approx(2 / 3)
    # PROE = mean(pass - xpass) over A's plays = ((1-.6)+(0-.5)+(1-.8))/3
    assert box.loc["A", "off_proe"] == pytest.approx(((1 - .6) + (0 - .5) + (1 - .8)) / 3)


def test_win_probability_filter_excludes_garbage_time():
    pbp = _synthetic_pbp()
    pbp.loc[2, "wp"] = 0.99          # A's 3.0-EPA play now garbage time
    box = game_box(pbp, wp_filter=True).set_index("team")
    assert box.loc["A", "off_epa_play"] == pytest.approx(0.0)   # mean of 1, -1
    unfiltered = game_box(pbp, wp_filter=False).set_index("team")
    assert unfiltered.loc["A", "off_epa_play"] == pytest.approx(1.0)


def test_rolling_features_are_nan_for_a_teams_first_game():
    tg = pd.DataFrame({
        "game_id": ["G1", "G2", "G3"],
        "team": ["A", "A", "A"],
        "season": [2022] * 3,
        "week": [1, 2, 3],
        "kickoff": pd.to_datetime(["2022-09-11", "2022-09-18", "2022-09-25"]),
    })
    box = pd.DataFrame({
        "game_id": ["G1", "G2", "G3"],
        "team": ["A", "A", "A"],
        "off_epa_play": [0.2, 0.4, 0.6],
    })
    out = rolling_features(tg, box, halflife=6.0, windows=(8,)).set_index("game_id")
    assert np.isnan(out.loc["G1", "off_epa_play_ewma"])       # nothing before it
    assert out.loc["G2", "off_epa_play_ewma"] == pytest.approx(0.2)
    assert out.loc["G3", "off_epa_play_ewma"] == pytest.approx(
        (0.4 + 0.2 * 0.5 ** (1 / 6)) / (1 + 0.5 ** (1 / 6))
    )
    assert out.loc["G1", "games_played"] == 0
    assert out.loc["G3", "games_played"] == 2


# --- Elo --------------------------------------------------------------------
def test_expected_score_is_symmetric_and_bounded():
    assert expected_score(1500, 1500) == pytest.approx(0.5)
    assert expected_score(1700, 1500) + expected_score(1500, 1700) == pytest.approx(1.0)
    assert 0.0 < expected_score(1000, 2000) < 0.5


def test_elo_is_zero_sum_and_responds_to_results():
    games = pd.DataFrame({
        "game_id": ["G1"],
        "season": [2022],
        "kickoff": pd.to_datetime(["2022-09-11"]),
        "home_team": ["A"],
        "away_team": ["B"],
        "result": [10.0],
        "location": ["Home"],
    })
    out = run_elo(games, k=20.0, home_field=0.0, initial=1500.0)
    assert out["elo_home_pre"].iloc[0] == 1500.0
    assert out["elo_prob_home"].iloc[0] == pytest.approx(0.5)


def test_elo_neutral_site_removes_home_field():
    base = dict(game_id=["G1"], season=[2022], kickoff=pd.to_datetime(["2022-09-11"]),
                home_team=["A"], away_team=["B"], result=[np.nan])
    home = run_elo(pd.DataFrame({**base, "location": ["Home"]}), home_field=55.0)
    neutral = run_elo(pd.DataFrame({**base, "location": ["Neutral"]}), home_field=55.0)
    assert home["elo_diff"].iloc[0] == pytest.approx(55.0)
    assert neutral["elo_diff"].iloc[0] == pytest.approx(0.0)


def test_elo_unplayed_games_do_not_update_ratings():
    games = pd.DataFrame({
        "game_id": ["G1", "G2"],
        "season": [2022, 2022],
        "kickoff": pd.to_datetime(["2022-09-11", "2022-09-18"]),
        "home_team": ["A", "A"],
        "away_team": ["B", "B"],
        "result": [np.nan, np.nan],
        "location": ["Home", "Home"],
    })
    out = run_elo(games, home_field=0.0)
    assert out["elo_home_pre"].tolist() == [1500.0, 1500.0]
