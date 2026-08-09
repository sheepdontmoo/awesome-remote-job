"""Tests that try to prove a strategy has no edge.

Every other module in this package answers "what happened?". This one answers
the only question that matters afterwards: **was any of it real?**

A backtest is a measurement taken with a ruler you also designed. If you tried
forty parameter sets and kept the best, the number you kept is contaminated by
the search itself, and no amount of care inside `engine.py` removes that. The
routines here are the standard ways of putting a figure on the contamination:

* `walk_forward` - choose parameters on data the test segment never saw, then
  score them on the untouched segment. In-sample and out-of-sample side by
  side, with the degradation ratio spelled out.
* `monte_carlo_drawdown` - the drawdown you observed is one draw from a
  distribution of orderings, and orderings are close to arbitrary. Usually it
  is a flattering draw.
* `permutation_test` - destroy the serial structure of the price series, keep
  everything else, and see how often noise reproduces your result.
* `parameter_sensitivity` - a lone profitable cell surrounded by losers is a
  fitted artefact. A broad plateau is at least the right shape for an effect.
* `deflated_sharpe` - subtract what the best of N random tries would have
  scored anyway.
* `null_benchmark` - what a coin flip with a stop-loss scores on *your* data,
  so "impressive" has a local reference point.

None of these can establish that an edge exists. They can only fail to kill
it, which is a weaker and much more useful thing. Passing all six is the
minimum bar for spending money on a demo account, not a green light.

Conventions
-----------
An *objective* is `Callable[[BacktestResult], float]`, larger is better. The
default is the annualised Sharpe with a hard floor on trade count: a
configuration that took fewer than `MIN_TRADES` trades scores `-inf`, because
three lucky trades are not a measurement. Where aggregates are taken over
objectives, `-inf` is folded to `0.0` - a strategy that never traded earned
nothing, and pretending its score is merely "missing" would flatter the
average.
"""

from __future__ import annotations

import itertools
import math
import random
import statistics
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Iterable, Optional, Sequence

from .engine import Backtester
from .metrics import Metrics, summarize
from .strategy import Context, Strategy
from .types import Bar, BacktestResult, Direction, Instrument, OrderIntent

__all__ = [
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
    "monte_carlo_drawdown",
    "permutation_test",
    "parameter_sensitivity",
    "deflated_sharpe",
    "surrogate_series",
    "null_benchmark",
]

# Below this many closed trades, no performance statistic means anything.
# Thirty is the conventional bar for a mean; ten is the bar for "this is not
# purely an artefact of two trades".
MIN_TRADES: int = 10

StrategyFactory = Callable[[dict], Strategy]
BacktesterFactory = Callable[[], Backtester]
Objective = Callable[[BacktestResult], float]


# ---------------------------------------------------------------------------
# objectives and evaluation
# ---------------------------------------------------------------------------


def sharpe_objective(result: BacktestResult, min_trades: int = MIN_TRADES) -> float:
    """Annualised Sharpe, or `-inf` when there were too few trades to judge.

    Rejecting thin samples inside the objective rather than afterwards matters:
    a parameter search left to its own devices will reliably discover the
    configuration that took four trades and won all of them.
    """
    if len(result.trades) < min_trades:
        return float("-inf")
    sharpe = summarize(result).sharpe
    # An undefined Sharpe - a curve with no variance to divide by - is not a
    # score of zero and must not crash a sweep half way through. It is the same
    # statement as "too few trades to judge": unscoreable.
    if sharpe is None:
        return float("-inf")
    return float(sharpe)


@dataclass(frozen=True)
class Evaluation:
    """One strategy configuration run over one segment of bars."""

    params: dict
    objective: float
    result: BacktestResult
    metrics: Metrics

    @property
    def trades(self) -> int:
        return len(self.result.trades)

    @property
    def net_pnl(self) -> float:
        return self.result.net_pnl

    @property
    def scored(self) -> bool:
        """False when the objective refused to score this run at all."""
        return math.isfinite(self.objective)


def _evaluate(
    strategy_factory: StrategyFactory,
    params: dict,
    bars: list[Bar],
    backtester_factory: BacktesterFactory,
    objective: Objective,
) -> Evaluation:
    """Run one configuration once. A fresh strategy and a fresh Backtester
    every time, so no state survives from the previous configuration."""
    strategy = strategy_factory(dict(params))
    result = backtester_factory().run(strategy, bars)
    return Evaluation(
        params=dict(params),
        objective=objective(result),
        result=result,
        metrics=summarize(result),
    )


def _score(objective: float) -> float:
    """Fold an unscoreable objective to zero for aggregation. See module docstring."""
    return objective if math.isfinite(objective) else 0.0


# ---------------------------------------------------------------------------
# walk-forward
# ---------------------------------------------------------------------------


def split_walk_forward(
    bars: list[Bar],
    n_folds: int = 5,
    train_frac: float = 0.6,
) -> list[tuple[list[Bar], list[Bar]]]:
    """Split `bars` into `n_folds` anchored (train, test) pairs, in time order.

    The first `train_frac` of the data is training-only. What remains is cut
    into `n_folds` contiguous, non-overlapping test segments. Fold `k` trains
    on everything that happened before its own test segment starts, so the
    training window is anchored at bar 0 and grows; the test segments never
    overlap each other and never overlap the training window that selected the
    parameters used on them.

    The honest caveat: fold `k`'s test segment *is* part of fold `k+1`'s
    training data. That is what "anchored" means and it matches how a live
    system would be re-fitted, but it does mean the later folds are not
    independent of the earlier ones. Bars are never reordered.

    Returns a list of `(train_bars, test_bars)` tuples.
    """
    if n_folds < 1:
        raise ValueError("n_folds must be at least 1")
    if not 0.0 < train_frac < 1.0:
        raise ValueError("train_frac must lie strictly between 0 and 1")

    n = len(bars)
    first_test = int(n * train_frac)
    if first_test < 1:
        raise ValueError(f"{n} bars with train_frac={train_frac} leaves no training data")
    if n - first_test < n_folds:
        raise ValueError(
            f"{n} bars leaves {n - first_test} bars after the training window, "
            f"which cannot be split into {n_folds} non-empty test segments"
        )

    remaining = n - first_test
    edges = [first_test + round(k * remaining / n_folds) for k in range(n_folds + 1)]
    return [(bars[: edges[k]], bars[edges[k] : edges[k + 1]]) for k in range(n_folds)]


@dataclass(frozen=True)
class FoldResult:
    """One walk-forward fold: what was chosen in-sample, what it then did."""

    index: int
    params: dict
    train_bars: int
    test_bars: int
    test_start: Optional[datetime]
    test_end: Optional[datetime]
    train_objective: float
    test_objective: float
    train_metrics: Metrics
    test_metrics: Metrics
    train_result: BacktestResult
    test_result: BacktestResult
    candidates_scored: int
    candidates_total: int


@dataclass(frozen=True)
class WalkForwardResult:
    """In-sample against out-of-sample, fold by fold."""

    folds: list[FoldResult]
    n_folds: int
    in_sample_objective: float
    out_of_sample_objective: float
    degradation: float
    in_sample_net_pnl: float
    out_of_sample_net_pnl: float
    out_of_sample_trades: int
    folds_positive: int
    verdict: str
    warnings: list[str] = field(default_factory=list)

    @property
    def best_params_per_fold(self) -> list[dict]:
        return [fold.params for fold in self.folds]


def walk_forward(
    strategy_factory: StrategyFactory,
    param_grid: Sequence[dict],
    bars: list[Bar],
    backtester_factory: BacktesterFactory,
    n_folds: int = 5,
    objective: Objective = sharpe_objective,
    train_frac: float = 0.6,
) -> WalkForwardResult:
    """Select parameters in-sample, score them out-of-sample, report the gap.

    For each fold the whole grid is run on the training segment and the best
    scoring configuration is carried forward. That configuration - and only
    that one - is then run on the test segment, which no part of the selection
    ever touched. The out-of-sample number is therefore the closest thing this
    package produces to an honest estimate, and it is still optimistic: the
    strategy code itself, the instrument, the timeframe and the grid bounds
    were all chosen by a human who had already seen this data.

    A fold whose test segment is shorter than the strategy's warmup cannot
    trade at all; that is recorded in `warnings` rather than silently scoring
    zero for a reason nobody can see.
    """
    if not param_grid:
        raise ValueError("param_grid is empty; there is nothing to select from")

    splits = split_walk_forward(bars, n_folds=n_folds, train_frac=train_frac)
    warnings: list[str] = []
    folds: list[FoldResult] = []

    warmup = strategy_factory(dict(param_grid[0])).warmup

    for index, (train_bars, test_bars) in enumerate(splits):
        if len(test_bars) <= warmup:
            warnings.append(
                f"fold {index}: test segment of {len(test_bars)} bars is not longer than "
                f"the strategy warmup of {warmup}; this fold could not trade"
            )

        candidates = [
            _evaluate(strategy_factory, params, train_bars, backtester_factory, objective)
            for params in param_grid
        ]
        scored = [c for c in candidates if c.scored]
        if scored:
            best = max(scored, key=lambda c: c.objective)
        else:
            best = candidates[0]
            warnings.append(
                f"fold {index}: no configuration in the grid took at least {MIN_TRADES} "
                f"trades in training; fell back to {best.params}, which selects nothing"
            )

        test = _evaluate(strategy_factory, best.params, test_bars, backtester_factory, objective)
        folds.append(
            FoldResult(
                index=index,
                params=dict(best.params),
                train_bars=len(train_bars),
                test_bars=len(test_bars),
                test_start=test_bars[0].ts if test_bars else None,
                test_end=test_bars[-1].ts if test_bars else None,
                train_objective=best.objective,
                test_objective=test.objective,
                train_metrics=best.metrics,
                test_metrics=test.metrics,
                train_result=best.result,
                test_result=test.result,
                candidates_scored=len(scored),
                candidates_total=len(candidates),
            )
        )

    in_sample = statistics.fmean(_score(f.train_objective) for f in folds)
    out_sample = statistics.fmean(_score(f.test_objective) for f in folds)
    degradation = out_sample / in_sample if in_sample > 0 else float("nan")
    oos_trades = sum(len(f.test_result.trades) for f in folds)
    positive = sum(1 for f in folds if f.test_result.net_pnl > 0)

    return WalkForwardResult(
        folds=folds,
        n_folds=len(folds),
        in_sample_objective=in_sample,
        out_of_sample_objective=out_sample,
        degradation=degradation,
        in_sample_net_pnl=sum(f.train_result.net_pnl for f in folds),
        out_of_sample_net_pnl=sum(f.test_result.net_pnl for f in folds),
        out_of_sample_trades=oos_trades,
        folds_positive=positive,
        verdict=_walk_forward_verdict(
            in_sample, out_sample, degradation, positive, len(folds), oos_trades
        ),
        warnings=warnings,
    )


def _walk_forward_verdict(
    in_sample: float,
    out_sample: float,
    degradation: float,
    positive: int,
    n_folds: int,
    oos_trades: int,
) -> str:
    tail = f"{positive} of {n_folds} folds profitable out of sample"
    if oos_trades < MIN_TRADES:
        return (
            f"only {oos_trades} out-of-sample trades across {n_folds} folds - too few to "
            f"separate skill from noise, whatever the objective says"
        )
    if in_sample <= 0.0:
        return (
            f"in-sample objective is {in_sample:.2f} - the search never found a profitable "
            f"configuration to begin with, so there is no degradation to measure; {tail}"
        )
    if out_sample <= 0.0:
        return (
            f"out-of-sample objective is {out_sample:.2f} against {in_sample:.2f} in-sample - "
            f"the sign flipped once the data was unseen; {tail}"
        )
    pct = degradation * 100.0
    if degradation < 0.5:
        return (
            f"out-of-sample objective is {pct:.0f}% of in-sample ({out_sample:.2f} vs "
            f"{in_sample:.2f}) - this is what overfitting looks like; {tail}"
        )
    if degradation < 0.8:
        return (
            f"out-of-sample objective is {pct:.0f}% of in-sample ({out_sample:.2f} vs "
            f"{in_sample:.2f}) - decay of this size is normal, and it is still decay; {tail}"
        )
    return (
        f"out-of-sample objective is {pct:.0f}% of in-sample ({out_sample:.2f} vs "
        f"{in_sample:.2f}), {tail} - consistent with a stable effect, which is not "
        f"the same as evidence of one"
    )


# ---------------------------------------------------------------------------
# Monte Carlo on trade order
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MonteCarloResult:
    """The distribution of drawdowns the same trades could have produced."""

    n_runs: int
    n_trades: int
    initial_equity: float
    observed_max_drawdown: float
    curve_max_drawdown: float
    observed_final_equity: float
    observed_drawdown_percentile: float
    median_max_drawdown: float
    p5_max_drawdown: float
    p95_max_drawdown: float
    p99_max_drawdown: float
    median_final_equity: float
    p5_final_equity: float
    p95_final_equity: float
    ruin_threshold: float
    prob_ruin: float
    verdict: str


def monte_carlo_drawdown(
    result: BacktestResult,
    n_runs: int = 5000,
    seed: int = 7,
) -> MonteCarloResult:
    """Bootstrap the realised trade sequence and report the drawdown spread.

    Each run draws `len(trades)` trades with replacement from the trades that
    actually happened, lays them end to end from the starting equity, and
    measures the deepest peak-to-trough fall of that path. The point is not
    the median: it is that the drawdown you observed is a single sample from
    this distribution, and the ordering that produced it was arbitrary. A
    strategy whose realised drawdown sits at the 20th percentile did not have
    a shallow drawdown; it had a lucky order.

    Two limitations worth stating rather than burying. Trade P&L is taken as
    realised, so reordering does not re-size later trades the way percentage
    risk actually would - real compounding makes good sequences better and bad
    ones worse than this shows. And the bootstrap assumes trades are
    independent, which destroys any streakiness the strategy really has; if
    losses cluster in reality, the true tail is worse than the p99 here.
    """
    pnls = [t.net_pnl for t in result.trades]
    start = float(result.initial_equity)
    threshold = float(getattr(result.risk, "ruin_threshold", 0.30))
    curve_dd = max((p.drawdown for p in result.equity_curve), default=0.0)

    if len(pnls) < 2 or n_runs < 1:
        return MonteCarloResult(
            n_runs=0,
            n_trades=len(pnls),
            initial_equity=start,
            observed_max_drawdown=_path_max_drawdown(_equity_path(start, pnls)),
            curve_max_drawdown=curve_dd,
            observed_final_equity=start + sum(pnls),
            observed_drawdown_percentile=float("nan"),
            median_max_drawdown=float("nan"),
            p5_max_drawdown=float("nan"),
            p95_max_drawdown=float("nan"),
            p99_max_drawdown=float("nan"),
            median_final_equity=float("nan"),
            p5_final_equity=float("nan"),
            p95_final_equity=float("nan"),
            ruin_threshold=threshold,
            prob_ruin=float("nan"),
            verdict=(
                f"with {_plural(len(pnls), 'closed trade')} there is no sequence to "
                f"resample - this is an anecdote, not a drawdown distribution"
            ),
        )

    observed_dd = _path_max_drawdown(_equity_path(start, pnls))
    observed_final = start + sum(pnls)

    rng = random.Random(seed)
    n = len(pnls)
    drawdowns: list[float] = []
    finals: list[float] = []
    ruined = 0
    ruin_level = threshold * start

    for _ in range(n_runs):
        sample = [pnls[rng.randrange(n)] for _ in range(n)]
        path = _equity_path(start, sample)
        drawdowns.append(_path_max_drawdown(path))
        finals.append(path[-1])
        if min(path) <= ruin_level:
            ruined += 1

    prob_ruin = ruined / n_runs
    percentile = sum(1 for d in drawdowns if d <= observed_dd) / n_runs
    median_dd = statistics.median(drawdowns)
    p95_dd = _percentile(drawdowns, 0.95)
    p99_dd = _percentile(drawdowns, 0.99)

    return MonteCarloResult(
        n_runs=n_runs,
        n_trades=n,
        initial_equity=start,
        observed_max_drawdown=observed_dd,
        curve_max_drawdown=curve_dd,
        observed_final_equity=observed_final,
        observed_drawdown_percentile=percentile,
        median_max_drawdown=median_dd,
        p5_max_drawdown=_percentile(drawdowns, 0.05),
        p95_max_drawdown=p95_dd,
        p99_max_drawdown=p99_dd,
        median_final_equity=statistics.median(finals),
        p5_final_equity=_percentile(finals, 0.05),
        p95_final_equity=_percentile(finals, 0.95),
        ruin_threshold=threshold,
        prob_ruin=prob_ruin,
        verdict=_monte_carlo_verdict(
            observed_dd, percentile, median_dd, p95_dd, p99_dd, prob_ruin, threshold, n_runs
        ),
    )


def _monte_carlo_verdict(
    observed: float,
    percentile: float,
    median: float,
    p95: float,
    p99: float,
    prob_ruin: float,
    threshold: float,
    n_runs: int,
) -> str:
    head = (
        f"the realised max drawdown of {_pct(observed)} sits at the "
        f"{_ordinal(round(percentile * 100))} percentile of {n_runs} reshuffles "
        f"(median {_pct(median)}, p95 {_pct(p95)}, p99 {_pct(p99)})"
    )
    if percentile <= 0.35:
        head += " - the order in which these trades arrived flattered the equity curve"
    elif percentile >= 0.80:
        head += " - the realised order was on the unlucky side, so the curve understates it"
    else:
        head += " - a typical draw, which means the p95 is the number to plan around"
    if prob_ruin > 0.0:
        head += (
            f"; {prob_ruin * 100:.1f}% of reshuffles fell to {_pct(threshold)} of "
            f"starting equity at some point"
        )
    else:
        head += f"; no reshuffle reached the {_pct(threshold)} ruin level"
    return head


def _equity_path(start: float, pnls: Sequence[float]) -> list[float]:
    equity = start
    path = [start]
    for pnl in pnls:
        equity += pnl
        path.append(equity)
    return path


def _path_max_drawdown(path: Sequence[float]) -> float:
    """Deepest peak-to-trough fall as a fraction of the running peak."""
    peak = path[0]
    worst = 0.0
    for equity in path:
        if equity > peak:
            peak = equity
        fall = (peak - equity) / peak if peak > 0.0 else 1.0
        if fall > worst:
            worst = fall
    return worst


# ---------------------------------------------------------------------------
# permutation test
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PermutationResult:
    """How often noise with the same statistics reproduced the result."""

    n_perms: int
    observed: float
    surrogate_median: float
    surrogate_p95: float
    surrogate_max: float
    n_at_least_observed: int
    n_unscored: int
    p_value: float
    verdict: str


def surrogate_series(bars: list[Bar], rng: random.Random) -> list[Bar]:
    """Rebuild `bars` from a shuffled copy of their own log returns.

    The surrogate keeps the timestamps, the number of bars, the starting
    price, and the exact multiset of bar-to-bar returns - so drift, volatility
    and the fat tails all survive. What it destroys is the *order*: any trend,
    mean reversion, momentum or level that a strategy could trade is gone.
    Intrabar range is carried over by rescaling each original bar's
    open/high/low around its new close, which preserves the size of the bar
    roughly but not the intrabar path.

    That last point is the honest weakness of this test: the surrogate's open
    is derived from its own close rather than from the previous bar's, so gap
    behaviour is not faithfully reproduced. A strategy that lives on gaps will
    be judged against a null that has the wrong kind of gaps.
    """
    n = len(bars)
    if n < 3:
        return list(bars)

    log_returns = []
    for i in range(1, n):
        previous, current = bars[i - 1].close, bars[i].close
        if previous <= 0.0 or current <= 0.0:
            raise ValueError("permutation test needs strictly positive close prices")
        log_returns.append(math.log(current / previous))
    rng.shuffle(log_returns)

    out = [bars[0]]
    close = bars[0].close
    for i in range(1, n):
        close *= math.exp(log_returns[i - 1])
        source = bars[i]
        scale = close / source.close
        open_ = source.open * scale
        high = max(source.high * scale, open_, close)
        low = min(source.low * scale, open_, close)
        out.append(
            Bar(ts=source.ts, open=open_, high=high, low=low, close=close, volume=source.volume)
        )
    return out


def permutation_test(
    strategy_factory: StrategyFactory,
    params: dict,
    bars: list[Bar],
    backtester_factory: BacktesterFactory,
    n_perms: int = 200,
    seed: int = 7,
    objective: Objective = sharpe_objective,
) -> PermutationResult:
    """Test the null hypothesis that the strategy has no edge on this series.

    The strategy is run once on the real bars, then once on each of `n_perms`
    surrogate series built by `surrogate_series`. The one-sided p-value is

        p = (1 + count(surrogate >= observed)) / (1 + n_perms)

    with the `1 +` on both sides so that the observed series counts as one of
    its own permutations - without it, zero surrogates beating the strategy
    would report `p = 0`, which is never a defensible claim from a finite
    sample. The smallest p-value 200 permutations can produce is 0.005.

    Read the result plainly: `p < 0.05` here is **necessary and nowhere near
    sufficient**. It says the result is unlikely to come from a series with the
    ordering destroyed. It says nothing about whether the parameters were
    tuned on this same data (they usually were - use `walk_forward`), whether
    the costs are right, whether the sample spans more than one regime, or
    whether the broker will fill you where the engine assumed. A tuned
    strategy will often clear this bar and still lose money live.
    """
    if n_perms < 1:
        raise ValueError("n_perms must be at least 1")

    observed_eval = _evaluate(strategy_factory, params, bars, backtester_factory, objective)
    observed = observed_eval.objective

    rng = random.Random(seed)
    scores: list[float] = []
    unscored = 0
    at_least = 0
    for _ in range(n_perms):
        fake = surrogate_series(bars, rng)
        value = _evaluate(strategy_factory, params, fake, backtester_factory, objective).objective
        if math.isfinite(value):
            scores.append(value)
        else:
            unscored += 1
        if value >= observed:
            at_least += 1

    p_value = (1 + at_least) / (1 + n_perms)
    return PermutationResult(
        n_perms=n_perms,
        observed=observed,
        surrogate_median=statistics.median(scores) if scores else float("nan"),
        surrogate_p95=_percentile(scores, 0.95) if scores else float("nan"),
        surrogate_max=max(scores) if scores else float("nan"),
        n_at_least_observed=at_least,
        n_unscored=unscored,
        p_value=p_value,
        verdict=_permutation_verdict(observed, at_least, n_perms, p_value, scores, unscored),
    )


def _permutation_verdict(
    observed: float,
    at_least: int,
    n_perms: int,
    p_value: float,
    scores: Sequence[float],
    unscored: int,
) -> str:
    if not math.isfinite(observed):
        return (
            f"the strategy took fewer than {MIN_TRADES} trades on the real series, so it "
            f"has no score to test; the permutation test has nothing to say here"
        )
    if not scores:
        return (
            f"none of the {n_perms} surrogates took enough trades to score - the strategy "
            f"only trades when the series has structure, so this test cannot separate "
            f"edge from the entry condition itself"
        )

    body = (
        f"p = {p_value:.3f}: {at_least} of {n_perms} return-shuffled surrogates matched or "
        f"beat the observed objective of {observed:.2f} "
        f"(surrogate median {statistics.median(scores):.2f}, "
        f"p95 {_percentile(scores, 0.95):.2f})"
    )
    if unscored:
        body += f"; {unscored} surrogates traded too little to score"

    if p_value < 0.05:
        return (
            body + " - the ordering of the real series matters, which is necessary and "
            "nowhere near sufficient: it says nothing about whether these parameters were "
            "fitted on this same data, or whether the costs are right"
        )
    if p_value < 0.20:
        return (
            body + f" - shuffled noise reproduces this result about one time in "
            f"{max(1, round(1 / p_value))}; that is not evidence of anything"
        )
    return (
        body + f" - shuffled noise reproduces this result {p_value * 100:.0f}% of the time; "
        f"the series ordering is doing no work here"
    )


# ---------------------------------------------------------------------------
# parameter sensitivity
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SensitivityResult:
    """The shape of the objective across the whole parameter grid."""

    cells: list[Evaluation]
    best: Evaluation
    best_objective: float
    median_objective: float
    profitable_fraction: float
    plateau_score: float
    n_cells: int
    n_scored: int
    n_neighbours: int
    verdict: str


def parameter_sensitivity(
    strategy_factory: StrategyFactory,
    param_grid: Sequence[dict],
    bars: list[Bar],
    backtester_factory: BacktesterFactory,
    objective: Objective = sharpe_objective,
) -> SensitivityResult:
    """Run the whole grid and describe the surface, not just its peak.

    Three numbers do the work. The **median** cell says what a randomly chosen
    configuration scores, which is the number you would have got without the
    search. The **profitable fraction** says how much of the grid works at all.
    The **plateau score** is the mean objective of the best cell's immediate
    neighbours divided by the best cell's own objective: parameters are
    continuous knobs on a market that does not know their values, so a real
    effect should degrade gently as you turn them. A plateau score near or
    below zero means the winner is a spike standing alone in a field of losers
    - that is a fitted artefact, and moving one parameter by one step in live
    trading would have destroyed it.

    Neighbours are defined on the implied grid axes: every distinct value of
    each key, sorted, with a neighbour being any cell within one step on every
    axis. Unscoreable neighbours count as zero, not as missing, for the reason
    in the module docstring.
    """
    if not param_grid:
        raise ValueError("param_grid is empty; there is nothing to evaluate")

    cells = [
        _evaluate(strategy_factory, params, bars, backtester_factory, objective)
        for params in param_grid
    ]
    scored = [c for c in cells if c.scored]
    best = max(cells, key=lambda c: (c.objective, c.net_pnl))
    profitable = sum(1 for c in cells if c.net_pnl > 0.0) / len(cells)
    median = statistics.median([c.objective for c in scored]) if scored else float("nan")

    neighbours = _neighbours(cells, best)
    if neighbours and math.isfinite(best.objective) and best.objective > 0.0:
        plateau = statistics.fmean(_score(c.objective) for c in neighbours) / best.objective
    else:
        plateau = float("nan")

    return SensitivityResult(
        cells=cells,
        best=best,
        best_objective=best.objective,
        median_objective=median,
        profitable_fraction=profitable,
        plateau_score=plateau,
        n_cells=len(cells),
        n_scored=len(scored),
        n_neighbours=len(neighbours),
        verdict=_sensitivity_verdict(best, median, profitable, plateau, len(cells), len(neighbours)),
    )


def _sensitivity_verdict(
    best: Evaluation,
    median: float,
    profitable: float,
    plateau: float,
    n_cells: int,
    n_neighbours: int,
) -> str:
    if not math.isfinite(best.objective):
        return (
            f"no cell in the {n_cells}-cell grid took at least {MIN_TRADES} trades - "
            f"the grid does not describe a tradable strategy on this data"
        )

    head = (
        f"best cell scores {best.objective:.2f} at {_format_params(best.params)}, grid median "
        f"{median:.2f}, {profitable * 100:.0f}% of {_plural(n_cells, 'cell')} profitable"
    )
    if best.objective <= 0.0:
        return head + " - the best configuration in the grid still loses money"
    if not math.isfinite(plateau) or n_neighbours == 0:
        return (
            head + " - the grid has no neighbouring cells to compare against, so nothing "
            "here distinguishes a plateau from a spike; widen the grid"
        )

    tail = (
        f"; its {_plural(n_neighbours, 'neighbour')} average {plateau * 100:.0f}% of its score"
    )
    if plateau <= 0.0:
        return head + tail + " - the winner is a lone spike surrounded by losers, which is curve-fitting"
    if plateau < 0.3:
        return head + tail + " - that is a spike, not a plateau; one step in any direction and the edge is gone"
    if plateau < 0.6:
        return head + tail + " - a narrow ridge; the result depends on the parameters more than is comfortable"
    return (
        head + tail + " - a broad plateau, which is the right shape for a real effect "
        "and is not by itself evidence of one"
    )


def _neighbours(cells: Sequence[Evaluation], best: Evaluation) -> list[Evaluation]:
    """Cells within one grid step of `best` on every axis."""
    keys = sorted(best.params)
    if not keys or any(sorted(c.params) != keys for c in cells):
        return []

    axes: dict = {}
    for key in keys:
        values = {c.params[key] for c in cells}
        axes[key] = _sorted_values(values)
    if all(len(axes[key]) < 2 for key in keys):
        return []

    index_of = {key: {value: i for i, value in enumerate(axes[key])} for key in keys}
    target = {key: index_of[key][best.params[key]] for key in keys}

    out = []
    for cell in cells:
        if cell is best:
            continue
        distance = max(abs(index_of[key][cell.params[key]] - target[key]) for key in keys)
        if distance == 1:
            out.append(cell)
    return out


def _sorted_values(values: Iterable) -> list:
    """Sort grid axis values, falling back to text order for mixed types."""
    try:
        return sorted(values)
    except TypeError:
        return sorted(values, key=repr)


# ---------------------------------------------------------------------------
# deflated Sharpe
# ---------------------------------------------------------------------------


_EULER_MASCHERONI = 0.5772156649015329


def deflated_sharpe(
    observed_sharpe: float,
    n_trials: int,
    n_obs: int,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """Subtract from a Sharpe ratio what the best of `n_trials` tries scores anyway.

    **The multiple-testing warning, stated plainly: testing 200 parameter sets
    guarantees an impressive-looking winner even on pure noise.** The maximum
    of 200 independent standard normals averages about 2.75 standard errors
    above zero. On five years of hourly data that is a comfortably positive
    annualised Sharpe produced entirely by searching. Every backtester finds
    that winner; almost none of them subtract it back out.

    This function does the subtraction. The expected maximum of `n_trials`
    standard normals is approximated by the standard extreme-value expression

        E[max] ~ (1 - g) * Phi^-1(1 - 1/N) + g * Phi^-1(1 - 1/(N e))

    with `g` the Euler-Mascheroni constant. That is scaled by the standard
    error of a Sharpe estimator with the supplied higher moments (Mertens; the
    same variance term Bailey and Lopez de Prado use for the Deflated Sharpe
    Ratio),

        se = sqrt((1 - skew * SR + (kurtosis - 1) / 4 * SR^2) / (n_obs - 1))

    and the product is subtracted from `observed_sharpe`. Fat tails and
    negative skew widen `se` and so deepen the haircut, which is correct:
    strategies that sell tails have noisier Sharpe estimates than their
    smooth-looking equity curves suggest.

    **Units matter and getting them wrong flatters you.** `n_obs` must be the
    number of return observations at the frequency `observed_sharpe` is
    expressed in: an annualised Sharpe goes with the number of *years*, a
    per-bar Sharpe with the number of *bars*. Pairing an annualised Sharpe
    with a bar count understates the haircut by roughly the square root of the
    bars per year.

    `n_trials` must count every configuration you evaluated, including the
    ones you abandoned early and the ones you tried before you started writing
    them down. Undercounting here is the whole failure mode.

    Returns the haircut Sharpe in the same units as `observed_sharpe`. It can
    be negative; a negative value means the observed result is below what the
    search alone would be expected to produce.
    """
    if n_trials < 1:
        raise ValueError("n_trials must be at least 1")
    if n_obs < 2:
        raise ValueError("n_obs must be at least 2 for a Sharpe standard error to exist")

    variance_term = 1.0 - skew * observed_sharpe + (kurtosis - 1.0) / 4.0 * observed_sharpe ** 2
    # The term is positive for any plausible moment set; clamp rather than
    # raise so a caller passing a wild sample kurtosis still gets a number.
    variance_term = max(variance_term, 1e-12)
    standard_error = math.sqrt(variance_term / (n_obs - 1))

    return observed_sharpe - _expected_max_normal(n_trials) * standard_error


def _expected_max_normal(n_trials: int) -> float:
    """Expected maximum of `n_trials` independent standard normal draws."""
    if n_trials <= 1:
        return 0.0
    upper = _norm_ppf(1.0 - 1.0 / n_trials)
    lower = _norm_ppf(1.0 - 1.0 / (n_trials * math.e))
    return (1.0 - _EULER_MASCHERONI) * upper + _EULER_MASCHERONI * lower


def _norm_cdf(x: float) -> float:
    """Standard normal CDF, from the error function in the stdlib."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


# Beasley-Springer-Moro coefficients for the inverse normal CDF.
_BSM_A = (2.50662823884, -18.61500062529, 41.39119773534, -25.44106049637)
_BSM_B = (-8.47351093090, 23.08336743743, -21.06224101826, 3.13082909833)
_BSM_C = (
    0.3374754822726147, 0.9761690190917186, 0.1607979714918209, 0.0276438810333863,
    0.0038405729373609, 0.0003951896511919, 0.0000321767881768, 0.0000002888167364,
    0.0000003960315187,
)


def _norm_ppf(p: float) -> float:
    """Inverse standard normal CDF: Beasley-Springer-Moro, polished by Newton.

    BSM alone is accurate to roughly 1e-9 in the body and degrades in the far
    tail, which is exactly where `_expected_max_normal` evaluates it for large
    trial counts. Two Newton steps against `_norm_cdf` cost nothing and remove
    the doubt.
    """
    if not 0.0 < p < 1.0:
        raise ValueError("p must lie strictly between 0 and 1")

    y = p - 0.5
    if abs(y) < 0.42:
        r = y * y
        numerator = (((_BSM_A[3] * r + _BSM_A[2]) * r + _BSM_A[1]) * r + _BSM_A[0])
        denominator = ((((_BSM_B[3] * r + _BSM_B[2]) * r + _BSM_B[1]) * r + _BSM_B[0]) * r + 1.0)
        x = y * numerator / denominator
    else:
        r = p if y < 0.0 else 1.0 - p
        r = math.log(-math.log(r))
        x = _BSM_C[0]
        power = 1.0
        for coefficient in _BSM_C[1:]:
            power *= r
            x += coefficient * power
        if y < 0.0:
            x = -x

    for _ in range(2):
        density = _norm_pdf(x)
        if density < 1e-300:
            break
        x -= (_norm_cdf(x) - p) / density
    return x


# ---------------------------------------------------------------------------
# grid helper and the no-skill baseline
# ---------------------------------------------------------------------------


def grid(**kwargs) -> list[dict]:
    """Cartesian product of keyword arguments, as a list of parameter dicts.

    >>> grid(fast=[5, 10], slow=[50])
    [{'fast': 5, 'slow': 50}, {'fast': 10, 'slow': 50}]

    The length of the returned list is the number you owe `deflated_sharpe`.
    """
    keys = list(kwargs)
    values = [list(kwargs[key]) for key in keys]
    return [dict(zip(keys, combination)) for combination in itertools.product(*values)]


class _CoinFlipEntry(Strategy):
    """Enter at random, size and exit like a real strategy, have no edge.

    This lives in `validation.py` rather than in `strategies/` on purpose: the
    no-skill baseline is part of the measuring apparatus, and `null_benchmark`
    must not depend on which strategies happen to be installed.

    The point of the baseline is that it is not a straw man. It pays the same
    spread, commission, slippage and swap, it sizes from the same
    `RiskConfig`, and it uses an ATR-scaled stop and target so its trade
    durations are comparable. The only thing it lacks is a reason to be in the
    market.
    """

    name = "coin_flip"

    def __init__(
        self,
        seed: int = 0,
        entry_prob: float = 0.02,
        atr_period: int = 14,
        stop_mult: float = 2.0,
        target_mult: float = 2.0,
    ) -> None:
        super().__init__(
            seed=seed,
            entry_prob=entry_prob,
            atr_period=atr_period,
            stop_mult=stop_mult,
            target_mult=target_mult,
        )
        self._rng = random.Random(seed)

    @property
    def warmup(self) -> int:
        return int(self.atr_period) + 1

    def on_start(self, instrument: Instrument) -> None:
        # Reseed so a reused instance is reproducible run to run.
        self._rng = random.Random(self.seed)

    def on_bar(self, ctx: Context) -> Optional[OrderIntent]:
        if ctx.position is not None:
            return None
        if self._rng.random() >= self.entry_prob:
            return None

        window = ctx.bars[-(int(self.atr_period) + 1) :]
        average_range = _mean_true_range(window)
        if average_range <= 0.0:
            return None

        price = ctx.price
        direction = Direction.LONG if self._rng.random() < 0.5 else Direction.SHORT
        sign = int(direction)
        return OrderIntent(
            direction=direction,
            stop_loss=price - sign * self.stop_mult * average_range,
            take_profit=price + sign * self.target_mult * average_range,
            tag="coin_flip",
        )


def _mean_true_range(window: Sequence[Bar]) -> float:
    """Mean true range over `window`, using only bars inside it."""
    if len(window) < 2:
        return 0.0
    total = 0.0
    for i in range(1, len(window)):
        bar = window[i]
        previous_close = window[i - 1].close
        total += max(
            bar.high - bar.low,
            abs(bar.high - previous_close),
            abs(bar.low - previous_close),
        )
    return total / (len(window) - 1)


def null_benchmark(
    bars: list[Bar],
    backtester_factory: BacktesterFactory,
    n_runs: int = 50,
    seed: int = 7,
    objective: Objective = sharpe_objective,
) -> dict:
    """Score a coin flip with a stop-loss on the caller's own data.

    "A Sharpe of 0.8" means nothing in the abstract. It means something once
    you know that fifty random-entry runs on the same bars, paying the same
    costs, produced a median of -0.35 and a best of 0.9. This function
    produces that reference distribution, so "impressive" has a local
    definition instead of an imported one.

    Read the `best` figure especially. It is the score of the luckiest of
    `n_runs` monkeys, and it is the number your strategy has to clear before
    the word "edge" is doing any work.

    Returns a dict with the run distribution and a `verdict` string.
    """
    if n_runs < 1:
        raise ValueError("n_runs must be at least 1")

    scores: list[float] = []
    net_pnls: list[float] = []
    trade_counts: list[int] = []
    for run in range(n_runs):
        strategy = _CoinFlipEntry(seed=seed + run)
        result = backtester_factory().run(strategy, bars)
        value = objective(result)
        if math.isfinite(value):
            scores.append(value)
        net_pnls.append(result.net_pnl)
        trade_counts.append(len(result.trades))

    profitable = sum(1 for pnl in net_pnls if pnl > 0.0) / n_runs
    summary = {
        "strategy": "coin flip entry, ATR stop and target, same costs and sizing",
        "n_runs": n_runs,
        "n_scored": len(scores),
        "median_objective": statistics.median(scores) if scores else float("nan"),
        "p5_objective": _percentile(scores, 0.05) if scores else float("nan"),
        "p95_objective": _percentile(scores, 0.95) if scores else float("nan"),
        "best_objective": max(scores) if scores else float("nan"),
        "median_net_pnl": statistics.median(net_pnls),
        "median_trades": statistics.median(trade_counts),
        "profitable_fraction": profitable,
    }
    summary["verdict"] = _null_benchmark_verdict(summary)
    return summary


def _null_benchmark_verdict(summary: dict) -> str:
    if summary["n_scored"] == 0:
        return (
            f"none of the {summary['n_runs']} random-entry runs took at least {MIN_TRADES} "
            f"trades on this data - the baseline could not be measured here"
        )
    return (
        f"{summary['n_runs']} random-entry runs on this data scored a median objective of "
        f"{summary['median_objective']:.2f} (p95 {summary['p95_objective']:.2f}, best "
        f"{summary['best_objective']:.2f}) on a median of {summary['median_trades']:.0f} "
        f"trades, with {summary['profitable_fraction'] * 100:.0f}% of runs profitable after "
        f"costs - any strategy scoring below {summary['best_objective']:.2f} has not yet "
        f"beaten the luckiest monkey in this sample"
    )


# ---------------------------------------------------------------------------
# small shared helpers
# ---------------------------------------------------------------------------


def _percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile. `q` is a fraction in [0, 1]."""
    if not values:
        return float("nan")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * min(max(q, 0.0), 1.0)
    low = int(math.floor(position))
    high = min(low + 1, len(ordered) - 1)
    fraction = position - low
    return ordered[low] * (1.0 - fraction) + ordered[high] * fraction


def _pct(fraction: float) -> str:
    return f"{fraction * 100:.1f}%"


def _plural(n: int, noun: str) -> str:
    return f"{n} {noun}" if n == 1 else f"{n} {noun}s"


def _ordinal(n: int) -> str:
    """1 -> '1st', 41 -> '41st', 11 -> '11th'."""
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _format_params(params: dict) -> str:
    if not params:
        return "(no parameters)"
    return ", ".join(f"{key}={value}" for key, value in sorted(params.items()))
