"""Concrete strategies, and the name -> class table the CLI resolves against.

Two of the five classes here are trading ideas and three of them are
measuring instruments, which is roughly the correct ratio. `DonchianBreakout`
and `ZScoreReversion` are the two classic families - trend following and mean
reversion - implemented plainly enough to read in one sitting, because the
point of having them in the package is to have something concrete to run the
validation machinery against, not to have something to trade.

`BuyAndHold`, `RandomEntry` and `AlwaysFlat` are the controls from
`baseline.py`. Every reported result should be quoted next to at least the
random-entry control on identical bars with identical costs; a strategy that
cannot clearly beat a coin flip with the same risk profile has not been shown
to contain a signal.

`REGISTRY` maps the short names the CLI accepts to classes. It is a plain
dict of classes rather than of constructed instances, so the CLI can pass
parameters through and so the same class can be instantiated many times with
different parameters during a sensitivity sweep without sharing state.
"""

from __future__ import annotations

from typing import Dict, Type

from ..strategy import Strategy
from .baseline import AlwaysFlat, BuyAndHold, RandomEntry
from .donchian import DonchianBreakout
from .mean_reversion import ZScoreReversion

__all__ = [
    "DonchianBreakout",
    "ZScoreReversion",
    "BuyAndHold",
    "RandomEntry",
    "AlwaysFlat",
    "REGISTRY",
]

REGISTRY: Dict[str, Type[Strategy]] = {
    "donchian": DonchianBreakout,
    "zscore": ZScoreReversion,
    "buyhold": BuyAndHold,
    "random": RandomEntry,
    "flat": AlwaysFlat,
}
