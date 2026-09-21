"""Configuration loading. Every season, path and hyperparameter comes from YAML."""
from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path("configs/default.yaml")


class Config:
    """Dict-backed config with dotted-path access.

    Wrapping rather than subclassing dict keeps ``cfg.get("a.b.c")`` unambiguous.
    """

    def __init__(self, data: dict[str, Any], source: Path | None = None):
        self._data = data
        self.source = source

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Config":
        path = Path(path) if path else DEFAULT_CONFIG_PATH
        if not path.exists():
            raise FileNotFoundError(f"config not found: {path}")
        with open(path) as fh:
            data = yaml.safe_load(fh) or {}
        return cls(data, source=path)

    def get(self, dotted: str, default: Any = "__raise__") -> Any:
        node: Any = self._data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                if default == "__raise__":
                    raise KeyError(f"missing config key: {dotted}")
                return default
            node = node[part]
        return copy.deepcopy(node) if isinstance(node, (dict, list)) else node

    def __getitem__(self, key: str) -> Any:
        return self.get(key)

    def as_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self._data)

    # --- convenience accessors used all over the codebase -------------------
    @property
    def raw_cache(self) -> Path:
        return Path(self.get("paths.raw_cache"))

    @property
    def feature_cache(self) -> Path:
        return Path(self.get("paths.feature_cache"))

    @property
    def artifacts(self) -> Path:
        return Path(self.get("paths.artifacts"))

    @property
    def seasons(self) -> list[int]:
        return list(range(self.get("data.start_season"), self.get("data.end_season") + 1))

    @property
    def pbp_seasons(self) -> list[int]:
        """Seasons of play-by-play to pull, including warm-up years before training starts."""
        warm = self.get("data.pbp_warmup_seasons", 1)
        return list(range(self.get("data.start_season") - warm, self.get("data.end_season") + 1))

    @property
    def seed(self) -> int:
        return int(self.get("models.seed", 1729))
