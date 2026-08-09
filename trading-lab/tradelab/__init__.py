"""tradelab - a backtesting lab built to be hard to fool yourself with.

This file exists to answer one question: what is the package, and what is an
implementation detail? Everything re-exported here is a supported name that
other code may import directly from `tradelab`. Everything else lives behind a
submodule and may move.

The surface is deliberately narrow and deliberately ordered the way a run
actually happens:

    data  ->  strategy  ->  Backtester.run  ->  summarize  ->  render_report
                                            \\-> validation

`data` fabricates or loads bars. A `Strategy` sees them one at a time through a
`Context` that cannot reach past the current bar. `Backtester` turns decisions
into money, paying spread, slippage, commission and swap on the way.
`summarize` reduces the run to numbers, `render_report` prints them, and the
`validation` entry points spend the rest of their time trying to prove those
numbers were luck.

Importing this module runs no code beyond the imports themselves: no global
configuration, no registry side effects, no random seeding. A backtest is a
pure function of (strategy, bars, configuration), and it stays that way.

Nothing exported here predicts anything. Every number this package produces
describes what a fixed rule set did on one sample of history at one set of
assumed costs.
"""

from __future__ import annotations

from . import data, indicators, metrics, report, strategies, validation
from .costs import CostModel, Fill
from .data import (
    DataError,
    load_csv,
    load_mt5_csv,
    resample,
    save_csv,
    synthetic_random_walk,
    synthetic_trending,
)
from .engine import Backtester
from .metrics import Metrics, infer_periods_per_year, summarize, summarize_trades_only
from .report import render_comparison, render_report, render_validation
from .strategies import (
    REGISTRY,
    AlwaysFlat,
    BuyAndHold,
    DonchianBreakout,
    RandomEntry,
    ZScoreReversion,
)
from .strategy import BarWindow, Context, LookAheadError, PositionView, Strategy
from .types import (
    BacktestResult,
    Bar,
    CostConfig,
    Direction,
    EquityPoint,
    ExitReason,
    Instrument,
    OrderIntent,
    RiskConfig,
    Trade,
)
from .validation import (
    MIN_TRADES,
    Evaluation,
    FoldResult,
    MonteCarloResult,
    PermutationResult,
    SensitivityResult,
    WalkForwardResult,
    deflated_sharpe,
    grid,
    monte_carlo_drawdown,
    null_benchmark,
    parameter_sensitivity,
    permutation_test,
    sharpe_objective,
    split_walk_forward,
    surrogate_series,
    walk_forward,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # value types
    "Bar",
    "Instrument",
    "CostConfig",
    "RiskConfig",
    "Direction",
    "ExitReason",
    "OrderIntent",
    "Trade",
    "EquityPoint",
    "BacktestResult",
    # strategy authoring
    "Strategy",
    "Context",
    "BarWindow",
    "PositionView",
    "LookAheadError",
    # execution
    "Backtester",
    "CostModel",
    "Fill",
    # measurement
    "Metrics",
    "summarize",
    "summarize_trades_only",
    "infer_periods_per_year",
    "render_report",
    "render_comparison",
    "render_validation",
    # strategies
    "REGISTRY",
    "DonchianBreakout",
    "ZScoreReversion",
    "BuyAndHold",
    "RandomEntry",
    "AlwaysFlat",
    # data
    "DataError",
    "load_csv",
    "load_mt5_csv",
    "save_csv",
    "resample",
    "synthetic_random_walk",
    "synthetic_trending",
    # validation
    "MIN_TRADES",
    "Evaluation",
    "FoldResult",
    "WalkForwardResult",
    "MonteCarloResult",
    "PermutationResult",
    "SensitivityResult",
    "grid",
    "sharpe_objective",
    "split_walk_forward",
    "walk_forward",
    "permutation_test",
    "monte_carlo_drawdown",
    "parameter_sensitivity",
    "deflated_sharpe",
    "surrogate_series",
    "null_benchmark",
    # submodules
    "data",
    "indicators",
    "metrics",
    "report",
    "strategies",
    "validation",
]
