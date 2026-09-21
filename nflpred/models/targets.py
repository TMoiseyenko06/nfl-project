"""The prediction targets the system supports.

Adding a market means adding a TargetSpec here; the models, backtester and
metrics all iterate over this registry rather than hardcoding three targets.

``market_column`` names the Vegas line that serves as the benchmark for a
target, or None where no free market data exists. Halftime lines are not in
nflverse, so the halftime targets have no true market benchmark - see
VegasBaseline, which fits a derived one from the full-game line and is
labelled as derived wherever it is reported.
"""
from __future__ import annotations

from dataclasses import dataclass

BINARY = "binary"
REGRESSION = "regression"


@dataclass(frozen=True)
class TargetSpec:
    name: str
    column: str            # target column in the feature table
    kind: str              # BINARY or REGRESSION
    pred_column: str       # column name for this target's prediction
    market_column: str | None = None   # Vegas benchmark, if one exists
    label: str = ""

    @property
    def is_binary(self) -> bool:
        return self.kind == BINARY


TARGETS: dict[str, TargetSpec] = {
    "win": TargetSpec(
        "win", "home_win", BINARY, "p_home", "vegas_spread",
        "Moneyline / win probability",
    ),
    "spread": TargetSpec(
        "spread", "spread_actual", REGRESSION, "pred_spread", "vegas_spread",
        "Point spread (home margin)",
    ),
    "total": TargetSpec(
        "total", "total_actual", REGRESSION, "pred_total", "vegas_total",
        "Total points",
    ),
    "h1_win": TargetSpec(
        "h1_win", "h1_home_win", BINARY, "p_home_h1", None,
        "Leading at halftime",
    ),
    "h1_spread": TargetSpec(
        "h1_spread", "h1_spread_actual", REGRESSION, "pred_h1_spread", None,
        "Halftime margin",
    ),
    "h1_total": TargetSpec(
        "h1_total", "h1_total_actual", REGRESSION, "pred_h1_total", None,
        "Halftime total points",
    ),
}

# A binary target and its corresponding margin target. Fitting these as two
# independent estimators lets them disagree - the moneyline pick saying one
# team and the spread saying the other - which is incoherent in output a human
# reads. Models can instead derive the probability from the margin, which makes
# the two agree by construction. See LinearModel(derive_binary_from_margin=...).
BINARY_FROM_MARGIN: dict[str, str] = {"win": "spread", "h1_win": "h1_spread"}

FULL_GAME_TARGETS = ["win", "spread", "total"]
HALFTIME_TARGETS = ["h1_win", "h1_spread", "h1_total"]
DEFAULT_TARGETS = FULL_GAME_TARGETS + HALFTIME_TARGETS


def resolve(names: list[str] | None) -> list[TargetSpec]:
    names = names if names is not None else DEFAULT_TARGETS
    out = []
    for n in names:
        if n not in TARGETS:
            raise KeyError(f"unknown target {n!r}; have {sorted(TARGETS)}")
        out.append(TARGETS[n])
    return out


def prediction_columns(names: list[str] | None = None) -> list[str]:
    return [t.pred_column for t in resolve(names)]
