"""Static reference data: stadium locations, time zones, divisions.

Coordinates are of the stadium itself, not the city centre, because travel
distance is measured venue-to-venue. Relocated franchises get their own
entries keyed by the abbreviation nflverse used at the time (STL/LA, SD/LAC,
OAK/LV), so historical travel is computed against the stadium the team
actually played in that season.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EARTH_RADIUS_MI = 3958.7613

# team -> (lat, lon, tz_utc_offset_hours_during_season)
# Arizona does not observe DST; during Sep-Jan it is effectively UTC-7.
TEAM_HOME: dict[str, tuple[float, float, float]] = {
    "ARI": (33.5276, -112.2626, -7.0),
    "ATL": (33.7554, -84.4009, -5.0),
    "BAL": (39.2780, -76.6227, -5.0),
    "BUF": (42.7738, -78.7870, -5.0),
    "CAR": (35.2258, -80.8528, -5.0),
    "CHI": (41.8623, -87.6167, -6.0),
    "CIN": (39.0955, -84.5160, -5.0),
    "CLE": (41.5061, -81.6995, -5.0),
    "DAL": (32.7473, -97.0945, -6.0),
    "DEN": (39.7439, -105.0201, -7.0),
    "DET": (42.3400, -83.0456, -5.0),
    "GB":  (44.5013, -88.0622, -6.0),
    "HOU": (29.6847, -95.4107, -6.0),
    "IND": (39.7601, -86.1639, -5.0),
    "JAX": (30.3239, -81.6373, -5.0),
    "KC":  (39.0489, -94.4839, -6.0),
    "LA":  (33.9535, -118.3392, -8.0),   # Rams (SoFi from 2020; Coliseum 2016-19)
    "LAC": (33.9535, -118.3392, -8.0),   # Chargers (SoFi from 2020; Dignity 2017-19)
    "LV":  (36.0909, -115.1833, -8.0),
    "MIA": (25.9580, -80.2389, -5.0),
    "MIN": (44.9737, -93.2581, -6.0),
    "NE":  (42.0909, -71.2643, -5.0),
    "NO":  (29.9511, -90.0812, -6.0),
    "NYG": (40.8135, -74.0745, -5.0),
    "NYJ": (40.8135, -74.0745, -5.0),
    "OAK": (37.7516, -122.2005, -8.0),
    "PHI": (39.9008, -75.1675, -5.0),
    "PIT": (40.4468, -80.0158, -5.0),
    "SD":  (32.7831, -117.1196, -8.0),
    "SEA": (47.5952, -122.3316, -8.0),
    "SF":  (37.4033, -121.9694, -8.0),
    "STL": (38.6327, -90.1885, -6.0),
    "TB":  (27.9759, -82.5033, -5.0),
    "TEN": (36.1665, -86.7713, -6.0),
    "WAS": (38.9077, -76.8645, -5.0),
}

# Early-period venues for relocated/temporarily-displaced clubs.
# (team, first_season, last_season) -> (lat, lon, tz)
TEAM_HOME_OVERRIDES: dict[tuple[str, int, int], tuple[float, float, float]] = {
    ("LA", 2016, 2019): (34.0141, -118.2879, -8.0),   # LA Memorial Coliseum
    ("LAC", 2017, 2019): (33.8644, -118.2611, -8.0),  # Dignity Health Sports Park
}

# Neutral-site venues (international series, relocations, Super Bowls).
# Keyed by nflverse stadium_id where available.
NEUTRAL_STADIUMS: dict[str, tuple[float, float, float]] = {
    "LON00": (51.5560, -0.2795, 1.0),    # Wembley
    "LON01": (51.4560, -0.3416, 1.0),    # Twickenham
    "LON02": (51.6043, -0.0665, 1.0),    # Tottenham Hotspur
    "MEX00": (19.3029, -99.1505, -6.0),  # Estadio Azteca
    "GER00": (48.2188, 11.6247, 2.0),    # Allianz Arena, Munich
    "MUN01": (48.2188, 11.6247, 2.0),    # Allianz Arena (alt id)
    "FRA00": (50.0686, 8.6455, 2.0),     # Deutsche Bank Park, Frankfurt
    "SAO00": (-23.5453, -46.4739, -3.0), # Arena Corinthians, Sao Paulo
    "RIO00": (-22.9121, -43.2302, -3.0), # Maracana, Rio
    "MAD01": (40.4531, -3.6883, 2.0),    # Bernabeu, Madrid
    "PAR00": (48.9245, 2.3601, 2.0),     # Stade de France
    "MEL00": (-37.8200, 144.9834, 10.0), # Melbourne Cricket Ground
}

# nflverse stadium_id -> the NFL team whose home venue that is. Needed for
# neutral-site games (Super Bowls, displaced home games) whose stadium_id
# prefix is the city code rather than the team abbreviation.
STADIUM_ID_TEAM: dict[str, str] = {
    "PHO00": "ARI", "ATL97": "ATL", "BAL00": "BAL", "BUF00": "BUF",
    "CAR00": "CAR", "CHI98": "CHI", "CIN00": "CIN", "CLE00": "CLE",
    "DAL00": "DAL", "DEN00": "DEN", "DET00": "DET", "GNB00": "GB",
    "HOU00": "HOU", "IND00": "IND", "JAX00": "JAX", "KAN00": "KC",
    "LAX01": "LA", "LAX00": "LA", "LAX97": "LAC", "VEG00": "LV",
    "MIA00": "MIA", "MIN01": "MIN", "NWE00": "NE", "NOR00": "NO",
    "NYC01": "NYG", "OAK00": "OAK", "PHI00": "PHI", "PIT00": "PIT",
    "SDG00": "SD", "SEA00": "SEA", "SFO01": "SF", "STL00": "STL",
    "TAM00": "TB", "NAS00": "TEN", "WAS00": "WAS",
}

DIVISIONS: dict[str, str] = {
    "BUF": "AFC East", "MIA": "AFC East", "NE": "AFC East", "NYJ": "AFC East",
    "BAL": "AFC North", "CIN": "AFC North", "CLE": "AFC North", "PIT": "AFC North",
    "HOU": "AFC South", "IND": "AFC South", "JAX": "AFC South", "TEN": "AFC South",
    "DEN": "AFC West", "KC": "AFC West", "LAC": "AFC West", "LV": "AFC West",
    "OAK": "AFC West", "SD": "AFC West",
    "DAL": "NFC East", "NYG": "NFC East", "PHI": "NFC East", "WAS": "NFC East",
    "CHI": "NFC North", "DET": "NFC North", "GB": "NFC North", "MIN": "NFC North",
    "ATL": "NFC South", "CAR": "NFC South", "NO": "NFC South", "TB": "NFC South",
    "ARI": "NFC West", "LA": "NFC West", "SF": "NFC West", "SEA": "NFC West",
    "STL": "NFC West",
}


def home_venue(team: str, season: int) -> tuple[float, float, float]:
    """Stadium coordinates + season time-zone offset for a team's home venue."""
    for (t, lo, hi), coords in TEAM_HOME_OVERRIDES.items():
        if t == team and lo <= season <= hi:
            return coords
    if team not in TEAM_HOME:
        raise KeyError(f"unknown team abbreviation: {team!r}")
    return TEAM_HOME[team]


def haversine_miles(lat1, lon1, lat2, lon2):
    """Great-circle distance in miles. Vectorised over numpy arrays."""
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(x, dtype=float)) for x in (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_MI * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def venue_coords(games: pd.DataFrame) -> pd.DataFrame:
    """Resolve the actual playing venue for each game.

    Home games use the home team's stadium. Neutral-site games use the mapped
    stadium when we know it, and otherwise fall back to the home team's venue
    with ``venue_known`` set False so the caller can see the approximation.
    """
    lats, lons, tzs, known = [], [], [], []
    for season, home, loc, sid in zip(
        games["season"], games["home_team"], games["location"], games.get("stadium_id", pd.Series([None] * len(games)))
    ):
        if loc == "Neutral" and isinstance(sid, str) and sid in NEUTRAL_STADIUMS:
            la, lo, tz = NEUTRAL_STADIUMS[sid]
            k = True
        elif loc == "Neutral":
            # Domestic neutral site (e.g. a displaced home game). The mapped
            # stadium_id is usually another NFL venue; try that first.
            alt = STADIUM_ID_TEAM.get(sid) if isinstance(sid, str) else None
            if alt:
                la, lo, tz = home_venue(alt, int(season))
                k = True
            else:
                la, lo, tz = home_venue(home, int(season))
                k = False
        else:
            la, lo, tz = home_venue(home, int(season))
            k = True
        lats.append(la)
        lons.append(lo)
        tzs.append(tz)
        known.append(k)
    return pd.DataFrame(
        {"venue_lat": lats, "venue_lon": lons, "venue_tz": tzs, "venue_known": known},
        index=games.index,
    )
