"""nflverse ingest with a versioned local cache.

Every download records url, sha256, byte size, HTTP ETag/Last-Modified and the
fetch timestamp into ``manifest.json``. Backtests can therefore be tied to an
exact data vintage: if the manifest changes, results are allowed to change.
Nothing re-downloads unless the file is missing or ``force`` is set.
"""
from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

from nflpred.config import Config

log = logging.getLogger(__name__)

MANIFEST_NAME = "manifest.json"
_TIMEOUT = 300


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_manifest(cache_dir: Path) -> dict:
    p = cache_dir / MANIFEST_NAME
    if p.exists():
        with open(p) as fh:
            return json.load(fh)
    return {}


def _save_manifest(cache_dir: Path, manifest: dict) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    with open(cache_dir / MANIFEST_NAME, "w") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)


def _download(url: str, dest: Path, retries: int = 4) -> dict:
    """Fetch ``url`` to ``dest``, returning provenance metadata."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            with requests.get(url, stream=True, timeout=_TIMEOUT) as r:
                r.raise_for_status()
                tmp = dest.with_suffix(dest.suffix + ".part")
                with open(tmp, "wb") as fh:
                    for chunk in r.iter_content(1 << 20):
                        fh.write(chunk)
                tmp.replace(dest)
                return {
                    "url": url,
                    "etag": r.headers.get("ETag"),
                    "last_modified": r.headers.get("Last-Modified"),
                    "fetched_at": datetime.now(timezone.utc).isoformat(),
                    "bytes": dest.stat().st_size,
                    "sha256": _sha256(dest),
                }
        except Exception as exc:  # noqa: BLE001 - network errors are retried
            last_err = exc
            wait = 2 ** (attempt + 1)
            log.warning("download failed (%s), retry %d in %ds: %s", url, attempt + 1, wait, exc)
            if attempt < retries - 1:
                import time

                time.sleep(wait)
    raise RuntimeError(f"failed to download {url} after {retries} attempts") from last_err


def fetch(cfg: Config, force: bool = False) -> dict:
    """Download games.csv and the configured play-by-play seasons into the cache."""
    cache = cfg.raw_cache
    cache.mkdir(parents=True, exist_ok=True)
    manifest = _load_manifest(cache)

    targets: list[tuple[str, str, Path]] = [
        ("games", cfg.get("sources.games_url"), cache / "games.csv")
    ]
    tmpl = cfg.get("sources.pbp_url_template")
    for season in cfg.pbp_seasons:
        targets.append((f"pbp_{season}", tmpl.format(season=season), cache / f"pbp_{season}.parquet"))

    for key, url, dest in targets:
        if dest.exists() and dest.stat().st_size > 0 and not force:
            if key not in manifest:
                # File present but unrecorded (e.g. fetched outside this module):
                # register it so the vintage is still pinned.
                manifest[key] = {
                    "url": url,
                    "fetched_at": datetime.now(timezone.utc).isoformat(),
                    "bytes": dest.stat().st_size,
                    "sha256": _sha256(dest),
                    "note": "adopted pre-existing cache file",
                }
            log.info("cached: %s", dest.name)
            continue
        log.info("downloading %s -> %s", url, dest.name)
        try:
            manifest[key] = _download(url, dest)
        except RuntimeError as exc:
            # A future season may not have a release yet; that is not fatal.
            if key.startswith("pbp_"):
                log.warning("skipping %s: %s", key, exc)
                continue
            raise

    _save_manifest(cache, manifest)
    return manifest


# --- readers ---------------------------------------------------------------

# Columns we actually consume. Reading a subset keeps 12 seasons of pbp in RAM.
PBP_COLUMNS = [
    "game_id", "season", "week", "season_type", "posteam", "defteam",
    "play_type", "epa", "success", "pass", "rush", "down", "ydstogo",
    "yardline_100", "qtr", "score_differential", "xpass", "qb_epa", "cpoe",
    "yards_gained", "penalty", "special", "sack", "interception", "fumble_lost",
    "touchdown", "first_down", "air_yards", "series_success", "drive",
    "shotgun", "no_huddle", "wp",
    # Required for the halftime targets (features/targets.py).
    "game_half", "total_home_score", "total_away_score",
]


def load_games(cfg: Config) -> pd.DataFrame:
    """The schedule/results table: one row per game, 1999-present."""
    path = cfg.raw_cache / "games.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing - run `nflpred fetch` first")
    g = pd.read_csv(path, low_memory=False)
    g["gameday"] = pd.to_datetime(g["gameday"], errors="coerce")
    return g


def load_pbp(cfg: Config, seasons: list[int] | None = None) -> pd.DataFrame:
    """Play-by-play for the requested seasons, restricted to the columns we use."""
    seasons = seasons if seasons is not None else cfg.pbp_seasons
    frames = []
    for season in seasons:
        path = cfg.raw_cache / f"pbp_{season}.parquet"
        if not path.exists():
            log.warning("no pbp cached for %s, skipping", season)
            continue
        import pyarrow.parquet as pq

        available = set(pq.ParquetFile(path).schema_arrow.names)
        cols = [c for c in PBP_COLUMNS if c in available]
        missing = set(PBP_COLUMNS) - available
        if missing:
            log.warning("pbp %s missing columns: %s", season, sorted(missing))
        frames.append(pd.read_parquet(path, columns=cols))
    if not frames:
        raise FileNotFoundError("no play-by-play cached - run `nflpred fetch` first")
    return pd.concat(frames, ignore_index=True)


def cache_vintage(cfg: Config) -> str:
    """Short, stable identifier for the current data vintage."""
    manifest = _load_manifest(cfg.raw_cache)
    if not manifest:
        return "unknown"
    blob = json.dumps({k: v.get("sha256") for k, v in sorted(manifest.items())}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]
