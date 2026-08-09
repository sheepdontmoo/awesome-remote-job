"""The command line: everything the package does, without writing any Python.

This module exists so that the first thing someone does with tradelab is run a
backtest that argues with them, rather than read an API. `python3 -m tradelab
demo` works on a freshly cloned tree with no data files, no configuration and
no install step, because the alternative - "first obtain five years of tick
data" - is where most people stop, and stopping means never seeing the numbers
that would have talked them out of the strategy.

Three commands, in the order they should be used:

* `demo` runs a breakout, a coin flip and buy-and-hold over the same
  synthetic bars and prints them side by side. The bars are a **random walk
  with no exploitable edge by construction**, so the comparison is not "which
  of these is good" - it is a demonstration that a rule set scores something
  on data that certainly contains nothing.
* `run` backtests one strategy, on your own CSV or on synthetic data, and
  prints the full report including the caveats the numbers themselves justify.
* `validate` spends most of its time trying to destroy the result: a parameter
  sweep, walk-forward, a permutation test against return-shuffled surrogates,
  a bootstrap of the drawdown, a Sharpe deflated by the number of
  configurations tried, and a random-entry baseline on the same bars.

Two decisions worth stating. **Synthetic data is the default everywhere**, and
every command says so in its output; a default that silently invented prices
would be worse than no default at all. And **costs are never defaulted to
zero** - the flags start at the package's conservative `CostConfig` and
`--stress` multiplies them, because the interesting question about a strategy
is not what it made, it is what survives the broker.

Exit codes: 0 on success, 2 on a usage or data error. Nothing here writes
files or mutates state; every command is a pure function of its arguments.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import Optional, Sequence

from .data import DataError, load_csv, load_mt5_csv, summary, synthetic_random_walk, synthetic_trending
from .engine import Backtester
from .metrics import summarize
from .report import render_comparison, render_report, render_validation
from .strategies import REGISTRY
from .strategy import Strategy
from .types import Bar, CostConfig, Instrument, RiskConfig
from .validation import (
    deflated_sharpe,
    grid,
    monte_carlo_drawdown,
    null_benchmark,
    parameter_sensitivity,
    permutation_test,
    walk_forward,
)

__all__ = ["main", "build_parser"]

# Parameter grids the `validate` command sweeps, per strategy. They are small
# on purpose: every extra cell is another draw in a maximum, and the deflated
# Sharpe below charges for all of them. Anything the caller pins with --param
# is removed from the sweep rather than fought over.
_DEFAULT_GRIDS: dict = {
    "donchian": {"entry_lookback": [10, 20, 40], "exit_lookback": [5, 10, 20]},
    "zscore": {"lookback": [10, 20, 40], "entry_z": [1.5, 2.0, 2.5]},
    "random": {"seed": [0, 1, 2]},
}

# Enough hourly bars for roughly a year of a five-day week. Short enough that
# `demo` finishes while somebody is still watching it.
_DEFAULT_BARS = 3000

_SYNTHETIC_NOTE = (
    "This series is synthetic. `walk` is a driftless random walk with "
    "independent returns and therefore no exploitable edge whatsoever; "
    "whatever a strategy scores on it is the score of luck, and that number is "
    "the bar a result on real data has to clear. `trend` embeds a deliberate "
    "lag-1 autocorrelation and exists only as a positive control - an edge that "
    "stable does not occur in markets."
)


# ---------------------------------------------------------------------------
# argument plumbing
# ---------------------------------------------------------------------------


def _coerce(raw: str):
    """Turn a `--param` value into the narrowest type that parses it."""
    text = raw.strip()
    lowered = text.lower()
    if lowered in ("true", "yes", "on"):
        return True
    if lowered in ("false", "no", "off"):
        return False
    if lowered in ("none", "null"):
        return None
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        pass
    return text


def _parse_param(text: str) -> tuple:
    key, separator, raw = text.partition("=")
    if not separator or not key.strip():
        raise argparse.ArgumentTypeError(f"expected key=value, got {text!r}")
    return key.strip(), _coerce(raw)


def _add_common(parser: argparse.ArgumentParser) -> None:
    """Flags shared by `run` and `validate`, in one place so they cannot drift."""
    parser.add_argument("--strategy", default="donchian", choices=sorted(REGISTRY),
                        help="strategy to run (default: donchian)")
    parser.add_argument("--param", action="append", type=_parse_param, default=[],
                        metavar="KEY=VALUE",
                        help="strategy parameter; repeatable, e.g. --param entry_lookback=40")
    parser.add_argument("--csv", default=None, metavar="PATH",
                        help="price file to load; omit to use synthetic data")
    parser.add_argument("--mt5", action="store_true",
                        help="treat --csv as a MetaTrader 5 export: tab separated, "
                             "YYYY.MM.DD dates, and BID prices (sets price_is_mid=False)")
    parser.add_argument("--series", choices=("walk", "trend"), default="walk",
                        help="synthetic generator when no --csv is given (default: walk, "
                             "which has no edge by construction)")
    parser.add_argument("--bars", type=int, default=_DEFAULT_BARS,
                        help=f"synthetic bars to generate (default: {_DEFAULT_BARS})")
    parser.add_argument("--seed", type=int, default=0, help="synthetic data seed (default: 0)")
    parser.add_argument("--equity", type=float, default=10_000.0,
                        help="starting equity (default: 10000)")
    parser.add_argument("--spread", type=float, default=CostConfig().spread_points,
                        help="spread in points (default: %(default)s)")
    parser.add_argument("--slippage", type=float, default=CostConfig().slippage_points,
                        help="slippage in points on market and stop fills (default: %(default)s)")
    parser.add_argument("--commission", type=float, default=Instrument().commission_per_lot_per_side,
                        help="commission per lot per side (default: %(default)s)")
    parser.add_argument("--risk", type=float, default=RiskConfig().risk_per_trade,
                        help="fraction of equity risked per trade (default: %(default)s)")
    parser.add_argument("--stress", type=float, default=1.0,
                        help="multiply spread, slippage and swap by this factor. "
                             "Run every promising result at 2.0 (default: 1.0)")
    parser.add_argument("--width", type=int, default=78, help="report width in columns")


def build_parser() -> argparse.ArgumentParser:
    """The full command line. Exposed so tests can drive it without a shell."""
    parser = argparse.ArgumentParser(
        prog="python3 -m tradelab",
        description="Backtest trading rules against history and against noise.",
        epilog="No command here forecasts anything. Every number describes what a "
               "fixed rule set did on one sample, at one set of assumed costs.",
    )
    parser.add_argument("--version", action="version", version=_version_string())
    subparsers = parser.add_subparsers(dest="command")

    demo = subparsers.add_parser(
        "demo",
        help="run a breakout, a coin flip and buy-and-hold on synthetic data",
        description="Compare a breakout against its own controls on bars that "
                    "contain no edge. Needs no data files.",
    )
    demo.add_argument("--bars", type=int, default=_DEFAULT_BARS,
                      help=f"synthetic bars to generate (default: {_DEFAULT_BARS})")
    demo.add_argument("--seed", type=int, default=0, help="synthetic data seed (default: 0)")
    demo.add_argument("--series", choices=("walk", "trend"), default="walk",
                      help="synthetic generator (default: walk)")
    demo.add_argument("--width", type=int, default=78, help="report width in columns")

    run = subparsers.add_parser(
        "run", help="backtest one strategy and print the full report",
        description="Backtest one strategy on a CSV or on synthetic data.",
    )
    _add_common(run)
    run.add_argument("--baselines", action="store_true",
                     help="also run the random-entry and buy-and-hold controls "
                          "on identical bars and print the comparison")

    validate = subparsers.add_parser(
        "validate", help="try to prove the result was luck",
        description="Parameter sweep, walk-forward, permutation test, drawdown "
                    "bootstrap, deflated Sharpe and a random-entry baseline. "
                    "Slower than `run` by roughly the permutation count.",
    )
    _add_common(validate)
    validate.add_argument("--perms", type=int, default=200,
                          help="return-shuffled surrogate series to test against (default: 200)")
    validate.add_argument("--folds", type=int, default=5,
                          help="walk-forward folds (default: 5)")
    validate.add_argument("--mc-runs", type=int, default=2000, dest="mc_runs",
                          help="trade-order bootstrap samples (default: 2000)")
    validate.add_argument("--null-runs", type=int, default=25, dest="null_runs",
                          help="random-entry baseline runs (default: 25)")
    return parser


def _version_string() -> str:
    from . import __version__

    return f"tradelab {__version__}"


# ---------------------------------------------------------------------------
# shared construction
# ---------------------------------------------------------------------------


class UsageError(Exception):
    """A problem with what the user asked for, not with the package."""


def _bars_from(args: argparse.Namespace) -> tuple:
    """Load or fabricate the bars, and a one-line description of where from."""
    if getattr(args, "csv", None):
        try:
            bars = load_mt5_csv(args.csv) if args.mt5 else load_csv(args.csv)
        except (DataError, OSError) as exc:
            raise UsageError(str(exc)) from exc
        return bars, f"{args.csv}"
    if args.bars < 1:
        raise UsageError("--bars must be at least 1")
    generate = synthetic_trending if args.series == "trend" else synthetic_random_walk
    return generate(args.bars, seed=args.seed), f"synthetic {args.series}, seed {args.seed}"


def _costs_from(args: argparse.Namespace) -> CostConfig:
    base = CostConfig(
        spread_points=args.spread,
        slippage_points=args.slippage,
        # An MT5 export is bid-side, so treating it as mid would add half a
        # spread the file has already taken off.
        price_is_mid=not getattr(args, "mt5", False),
    )
    return base.stress(args.stress) if args.stress != 1.0 else base


def _backtester_from(args: argparse.Namespace) -> Backtester:
    return Backtester(
        instrument=Instrument(commission_per_lot_per_side=args.commission),
        costs=_costs_from(args),
        risk=RiskConfig(risk_per_trade=args.risk),
        initial_equity=args.equity,
    )


def _build_strategy(name: str, params: dict) -> Strategy:
    try:
        return REGISTRY[name](**params)
    except TypeError as exc:
        raise UsageError(f"{name}: {exc}") from exc
    except ValueError as exc:
        raise UsageError(f"{name}: {exc}") from exc


def _fixed_params(args: argparse.Namespace) -> dict:
    return dict(getattr(args, "param", []) or [])


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------


def cmd_demo(args: argparse.Namespace, stream) -> int:
    """Three strategies, identical bars, identical costs, printed side by side."""
    generate = synthetic_trending if args.series == "trend" else synthetic_random_walk
    bars = generate(max(1, args.bars), seed=args.seed)
    width = max(48, args.width)

    print(_banner("DATA", width), file=stream)
    print(summary(bars), file=stream)
    print(file=stream)
    print(_paragraph(_SYNTHETIC_NOTE, width), file=stream)

    runs = []
    for key in ("donchian", "random", "buyhold"):
        result = Backtester().run(REGISTRY[key](), bars)
        runs.append((key, result, summarize(result)))

    print(file=stream)
    print(_banner("COMPARISON", width), file=stream)
    print(render_comparison(runs, width=width), file=stream)

    print(file=stream)
    result, metrics = runs[0][1], runs[0][2]
    print(render_report(result, metrics, width=width), file=stream)

    print(file=stream)
    print(
        _paragraph(
            "That report describes a breakout system trading data generated with "
            "no edge in it. Read the CAVEATS section: everything it says about "
            "this run is also true of a run on real prices that happens to look "
            "good. Next: `python3 -m tradelab run --strategy donchian --csv "
            "YOUR.csv --stress 2.0`, then `python3 -m tradelab validate`.",
            width,
        ),
        file=stream,
    )
    return 0


def cmd_run(args: argparse.Namespace, stream) -> int:
    """One strategy, one set of bars, the whole report."""
    bars, source = _bars_from(args)
    width = max(48, args.width)
    strategy = _build_strategy(args.strategy, _fixed_params(args))

    print(_banner("DATA", width), file=stream)
    print(f"Source    : {source}", file=stream)
    print(summary(bars), file=stream)
    if not args.csv:
        print(file=stream)
        print(_paragraph(_SYNTHETIC_NOTE, width), file=stream)

    result = _backtester_from(args).run(strategy, bars)
    print(file=stream)
    print(render_report(result, summarize(result), width=width), file=stream)

    if args.baselines:
        runs = [(strategy.name, result, summarize(result))]
        for key in ("random", "buyhold"):
            if key == args.strategy:
                continue
            control = _backtester_from(args).run(REGISTRY[key](), bars)
            runs.append((key, control, summarize(control)))
        print(file=stream)
        print(_banner("CONTROLS", width), file=stream)
        print(render_comparison(runs, width=width), file=stream)
    return 0


@dataclass(frozen=True)
class _Deflation:
    """The Sharpe of the best cell, minus what searching alone would produce."""

    observed_sharpe: Optional[float]
    n_trials: int
    n_years: Optional[float]
    deflated_sharpe: Optional[float]
    verdict: str


def cmd_validate(args: argparse.Namespace, stream) -> int:
    """Sweep, walk forward, permute, bootstrap, deflate, and benchmark."""
    bars, source = _bars_from(args)
    width = max(48, args.width)
    if args.perms < 1 or args.folds < 1:
        raise UsageError("--perms and --folds must be at least 1")

    fixed = _fixed_params(args)
    cells = _grid_for(args.strategy, fixed)
    # Fail fast on a bad parameter rather than after the first fold.
    _build_strategy(args.strategy, cells[0])

    def factory(params: dict) -> Strategy:
        return _build_strategy(args.strategy, params)

    def backtester_factory() -> Backtester:
        return _backtester_from(args)

    print(_banner("DATA", width), file=stream)
    print(f"Source    : {source}", file=stream)
    print(summary(bars), file=stream)
    if not args.csv:
        print(file=stream)
        print(_paragraph(_SYNTHETIC_NOTE, width), file=stream)
    print(file=stream)
    print(
        _paragraph(
            f"Sweeping {len(cells)} configuration(s) of {args.strategy}, then "
            f"walk-forward over {args.folds} folds, {args.perms} permutations, "
            f"{args.mc_runs} bootstrap samples and {args.null_runs} random-entry "
            "runs. The permutation test below uses the best cell of the sweep, "
            "which is exactly the multiple-testing problem the deflated Sharpe "
            "then charges for.",
            width,
        ),
        file=stream,
    )

    sensitivity = parameter_sensitivity(factory, cells, bars, backtester_factory)
    best_params = dict(sensitivity.best.params)

    forward = _walk_forward_or_none(factory, cells, bars, backtester_factory, args.folds)
    permutation = permutation_test(
        factory, best_params, bars, backtester_factory, n_perms=args.perms
    )
    best_result = sensitivity.best.result
    monte_carlo = monte_carlo_drawdown(best_result, n_runs=max(1, args.mc_runs))
    baseline = null_benchmark(bars, backtester_factory, n_runs=max(1, args.null_runs))
    deflation = _deflate(best_result, sensitivity.best.metrics, len(cells))

    records = [sensitivity]
    if forward is not None:
        records.append(forward)
    records.extend([permutation, monte_carlo, deflation, baseline])

    print(file=stream)
    print(render_validation(*records, width=width), file=stream)
    return 0


def _grid_for(name: str, fixed: dict) -> list:
    """The sweep for `name`, minus every axis the caller pinned with --param."""
    axes = {key: values for key, values in _DEFAULT_GRIDS.get(name, {}).items() if key not in fixed}
    cells = grid(**axes) if axes else [{}]
    return [dict(fixed, **cell) for cell in cells]


def _walk_forward_or_none(factory, cells, bars, backtester_factory, folds):
    """Walk-forward, or None when the sample cannot be split that many ways.

    A sample too short to fold is a fact about the data, not an error worth
    abandoning the other five tests over.
    """
    try:
        return walk_forward(factory, cells, bars, backtester_factory, n_folds=folds)
    except ValueError:
        return None


def _deflate(result, metrics, n_trials: int) -> _Deflation:
    """Haircut the best cell's Sharpe by the expected maximum of the search.

    Units are the trap here: an *annualised* Sharpe must be deflated against a
    count of *years*, not of bars. Pairing it with a bar count understates the
    haircut by roughly the square root of the bars per year, which is the
    difference between a result that survives the search and one that does not.
    """
    sharpe = metrics.sharpe
    ppy = metrics.periods_per_year
    years = (len(result.equity_curve) / ppy) if ppy else None
    if sharpe is None or years is None or years < 2.0 / 12.0:
        return _Deflation(
            observed_sharpe=sharpe,
            n_trials=n_trials,
            n_years=years,
            deflated_sharpe=None,
            verdict=(
                "not enough of a sample to deflate: an annualised Sharpe has to be "
                "haircut against the number of years of returns behind it, and this "
                "run does not have a usable span. Read the sweep and the walk-forward "
                "instead."
            ),
        )
    value = deflated_sharpe(float(sharpe), n_trials=max(1, n_trials), n_obs=max(2, int(round(years))))
    if value > 0.0:
        tail = (
            "still positive after the haircut, which means the search alone does not "
            "explain it - on this sample, at these costs, with these rules chosen by "
            "somebody who had already seen the data"
        )
    else:
        tail = (
            "at or below zero after the haircut: searching this many configurations "
            "would be expected to produce a result this good on data with no edge in it"
        )
    return _Deflation(
        observed_sharpe=float(sharpe),
        n_trials=n_trials,
        n_years=years,
        deflated_sharpe=value,
        verdict=(
            f"best-cell Sharpe of {float(sharpe):.2f} over ~{years:.2f} years, deflated "
            f"for {n_trials} configuration(s) tried, comes to {value:.2f} - {tail}"
        ),
    )


# ---------------------------------------------------------------------------
# formatting
# ---------------------------------------------------------------------------


def _banner(title: str, width: int) -> str:
    return f"== {title} " + "=" * max(0, width - len(title) - 4)


def _paragraph(text: str, width: int) -> str:
    import textwrap

    return "\n".join(textwrap.wrap(text, width=width))


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


_COMMANDS = {"demo": cmd_demo, "run": cmd_run, "validate": cmd_validate}


def main(argv: Optional[Sequence[str]] = None, stream=None) -> int:
    """Run one command. Returns the process exit code rather than exiting.

    `stream` is where the report goes; it defaults to stdout and exists so a
    test can capture the output without redirecting the process.
    """
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    out = stream if stream is not None else sys.stdout

    if not args.command:
        parser.print_help(out)
        return 0
    try:
        return _COMMANDS[args.command](args, out)
    except UsageError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
