"""Performance statistics, and the warnings that belong next to them.

A backtest produces two things: an equity curve and a list of trades. Turning
those into a table of ratios is easy, and that is exactly the problem. Sharpe,
profit factor and CAGR are all trivially computable from thirty lucky trades,
and they look identical whether they came from an edge or from a coin that
happened to land the right way. Most published results are not fraudulent; they
are a real calculation performed on a sample far too small to support it.

So this module does two jobs, and the second one matters more:

1. Compute the numbers, carefully, with every ratio returning ``None`` rather
   than ``inf`` or a crash when it is undefined. A strategy with no losing
   trades has no profit factor - that is information, and printing ``inf`` or
   ``999.0`` throws it away at precisely the moment it is most interesting.
2. Argue with them. `Metrics.caveats()` reads the same numbers back and states,
   in plain English, what they are not entitled to claim. Fewer than thirty
   trades cannot separate skill from noise. Costs eating half the gross P&L
   means the model of the broker is now the dominant term in the result. A run
   of eight losers is arithmetically survivable and psychologically not.

Conventions
-----------
* **The risk-free rate is assumed to be zero.** Sharpe and Sortino here are
  mean return over volatility, not excess return over volatility. With cash
  paying a non-trivial yield this flatters an intraday strategy that holds no
  overnight risk, and it understates nothing. Adjust mentally, or subtract the
  bill rate from the returns before you compare against a benchmark.
* Annualisation assumes a **5-day forex week**: 252 trading days of 24 hours,
  not 365 calendar days. Hourly bars therefore imply ~6048 periods per year.
  Annualising 24-hour bars as if there were 365 of them a year overstates the
  period count by ~45% and the Sharpe by ~20%.
* Everything derived from an equity curve is derived from *equity*, not
  balance, so open-position mark-to-market counts against the drawdown. That is
  what an account statement and a margin call both use.
* Drawdown fractions are fractions (0.25 = 25%). The Ulcer Index is the one
  exception and is in percent, because that is its published definition.

Nothing here forecasts anything. Every number describes what a specific rule
set did on one specific sample of history, at one specific set of assumed
costs, and the caveats exist to keep that sentence attached to it.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Sequence

from .types import BacktestResult, EquityPoint, Instrument, Trade

# A forex week is five days long and each of those days is 24 hours. 252 * 24
# is the hour count that follows; it is also within 3% of the 260-weekday
# convention used by the synthetic data generators, which is close enough that
# the two never disagree about anything that matters.
_TRADING_DAYS_PER_YEAR = 252.0
_SECONDS_PER_TRADING_YEAR = _TRADING_DAYS_PER_YEAR * 24.0 * 3600.0
_SECONDS_PER_DAY = 86400.0
# Wall-clock spans include weekends; trading time does not. 5/7 converts one to
# the other when only timestamps are available (the trades-only path).
_TRADING_TIME_FRACTION = 5.0 / 7.0

# Thresholds used by `caveats()`. They are judgement calls, not laws, and they
# are named here so that they can be argued with instead of hidden in an `if`.
_MIN_CREDIBLE_TRADES = 30
_MAX_TOLERABLE_COST_DRAG = 0.5
_MIN_MEANINGFUL_EXPOSURE = 0.05
_PAINFUL_LOSING_STREAK = 6
_BUSY_TRADES_PER_YEAR = 500.0
_SEVERE_DRAWDOWN = 0.30

_EPS = 1e-12


@dataclass(frozen=True)
class Metrics:
    """Everything worth saying about a run, in numbers.

    Every ratio is `Optional`. `None` means "undefined on this sample", which
    is a different statement from zero and a very different statement from a
    large number. Read `caveats()` before reading anything else here.
    """

    # -- returns ----------------------------------------------------------
    total_return: Optional[float] = None      # fraction, final/initial - 1
    cagr: Optional[float] = None              # annualised, trading-time years
    final_equity: float = 0.0
    net_pnl: float = 0.0                      # after every cost
    gross_pnl: float = 0.0                    # what a naive backtest reports

    # -- risk -------------------------------------------------------------
    max_drawdown: float = 0.0                 # fraction below the running peak
    max_drawdown_cash: float = 0.0            # account currency
    longest_drawdown_bars: int = 0
    longest_drawdown_days: float = 0.0
    ulcer_index: float = 0.0                  # RMS drawdown, in PERCENT

    # -- ratios -----------------------------------------------------------
    sharpe: Optional[float] = None            # annualised, risk-free = 0
    sortino: Optional[float] = None           # annualised, risk-free = 0
    calmar: Optional[float] = None            # cagr / max_drawdown
    profit_factor: Optional[float] = None     # gross wins / gross losses (net)
    expectancy_r: Optional[float] = None      # mean R per trade
    tail_ratio: Optional[float] = None        # p95 / |p05| of bar returns

    # -- trades -----------------------------------------------------------
    n_trades: int = 0
    win_rate: Optional[float] = None
    avg_win: Optional[float] = None           # account currency, positive
    avg_loss: Optional[float] = None          # account currency, NEGATIVE
    payoff_ratio: Optional[float] = None      # avg_win / |avg_loss|
    avg_r: Optional[float] = None
    best_r: Optional[float] = None
    worst_r: Optional[float] = None
    max_consecutive_losses: int = 0           # non-winning trades in a row
    avg_bars_held: Optional[float] = None
    trades_per_year: Optional[float] = None

    # -- costs ------------------------------------------------------------
    total_costs: float = 0.0                  # positive when they hurt
    commission_total: float = 0.0
    spread_total: float = 0.0
    slippage_total: float = 0.0
    swap_total: float = 0.0                   # signed; negative = paid away
    cost_drag: Optional[float] = None         # total_costs / |gross_pnl|
    break_even_edge_points: Optional[float] = None  # points per round turn

    # -- exposure ---------------------------------------------------------
    time_in_market: Optional[float] = None    # fraction of bars holding

    # -- meta -------------------------------------------------------------
    periods_per_year: Optional[float] = None  # basis used for annualisation
    halted_early: bool = False
    n_warnings: int = 0

    def caveats(self) -> list[str]:
        """Plain-English warnings that these numbers themselves justify.

        This is the most important method in the module. Every other function
        here helps a strategy look like something; this one is the only part
        that pushes back. Each entry is triggered by an explicit threshold on a
        computed value - none of it is opinion about the strategy, only about
        what a sample of this size and shape can support.

        Returns at least one string, always. Silence would read as approval.
        """
        out: list[str] = []

        if self.n_trades == 0:
            out.append(
                "No trades were taken. Every number below is an artefact of an "
                "idle account, not a result."
            )
        elif self.n_trades < _MIN_CREDIBLE_TRADES:
            plural = "trade" if self.n_trades == 1 else "trades"
            out.append(
                f"Only {self.n_trades} closed {plural}. This sample cannot "
                "distinguish skill from noise: with this few observations the "
                "confidence interval around every ratio below spans both "
                "'profitable' and 'not'. Treat it as a smoke test, not evidence."
            )

        if self.cost_drag is not None and self.cost_drag > _MAX_TOLERABLE_COST_DRAG:
            out.append(
                f"Costs came to {self.cost_drag * 100:.0f}% of gross P&L in "
                "magnitude. The result is now mostly a statement about the "
                "assumed spread, commission and slippage rather than about the "
                "rules. If those assumptions are optimistic by a third, the edge "
                "is gone. Re-run at stress(2.0)."
            )

        if self.break_even_edge_points is not None and self.trades_per_year is not None:
            annual_points = self.break_even_edge_points * self.trades_per_year
            if self.trades_per_year > _BUSY_TRADES_PER_YEAR:
                out.append(
                    f"{self.trades_per_year:.0f} trades a year, each needing "
                    f"{self.break_even_edge_points:.1f} points just to break "
                    f"even: about {annual_points:.0f} points a year paid to the "
                    "broker before any profit. At this turnover the modelled "
                    "slippage constant, not the signal, decides the outcome, "
                    "and real slippage is worse than a constant during news."
                )

        if (
            self.n_trades > 0
            and self.time_in_market is not None
            and self.time_in_market < _MIN_MEANINGFUL_EXPOSURE
        ):
            out.append(
                f"Positions were open for only {self.time_in_market * 100:.1f}% "
                "of bars. The equity curve is mostly flat line, which suppresses "
                "measured volatility and inflates Sharpe. Judge this one on "
                "expectancy per trade, not on annualised ratios."
            )

        if self.max_consecutive_losses >= _PAINFUL_LOSING_STREAK:
            out.append(
                f"{self.max_consecutive_losses} losing trades in a row occurred "
                "in this backtest, so a longer run is entirely possible live. "
                "Most people abandon a system somewhere around the fifth or "
                "sixth loss. A system you cannot sit through has no expectancy, "
                "whatever the arithmetic says."
            )

        if self.max_drawdown > _SEVERE_DRAWDOWN:
            out.append(
                f"Peak-to-trough drawdown of {self.max_drawdown * 100:.0f}% "
                f"({self.max_drawdown_cash:,.0f} in account currency). Recovering "
                "from that requires a "
                f"{(1.0 / (1.0 - self.max_drawdown) - 1.0) * 100:.0f}% gain, and "
                "it happened on the sample the rules were chosen against."
            )

        if self.n_trades > 0 and self.profit_factor is None:
            out.append(
                "No losing trades at all, so profit factor is undefined. On real "
                "data this is almost always a look-ahead bug, an unrealistically "
                "generous fill assumption, or a sample too short to have met a "
                "bad week yet."
            )

        if self.n_trades >= 2 and self.sharpe is None:
            out.append(
                "Sharpe is undefined: the equity return series has no variance "
                "to divide by. Check that the curve is actually moving."
            )

        # One trade carrying the whole result is the classic curve-fit
        # signature: remove it and the strategy is flat or negative.
        if (
            self.best_r is not None
            and self.avg_r is not None
            and self.n_trades >= 2
        ):
            total_r = self.avg_r * self.n_trades
            if total_r > _EPS and self.best_r > 0.5 * total_r:
                out.append(
                    f"The single best trade is worth {self.best_r:.1f}R against "
                    f"{total_r:.1f}R from all {self.n_trades} trades combined. "
                    "Delete that one trade and there is little or no strategy "
                    "left, only a lottery ticket that happened to hit."
                )

        if self.halted_early:
            out.append(
                "The run halted early on the ruin threshold. Whatever the "
                "surviving statistics say, this configuration lost enough of "
                "the account to stop the test."
            )

        if self.n_warnings:
            out.append(
                f"The engine emitted {self.n_warnings} warning(s) during the run "
                "(skipped trades, sizing failures, data problems). Read them; "
                "a strategy that mostly fails to size is not the strategy tested."
            )

        if not out:
            out.append(
                "No automatic caveats triggered. This is not an endorsement."
            )
        return out


# ---------------------------------------------------------------------------
# periodicity
# ---------------------------------------------------------------------------


def infer_periods_per_year(equity_curve: Sequence[EquityPoint]) -> float:
    """Periods per year implied by the spacing of the equity curve.

    Uses the **median** gap, not the mean. Forex data has a 48-hour hole every
    weekend and assorted holiday gaps; the mean gap is dragged upward by them
    and would under-count the periods in a year, quietly deflating Sharpe. The
    median reports the bar size the data was actually sampled at.

    A trading year here is 252 days of 24 hours - a 5-day week - so hourly bars
    give ~6048 and daily bars give 252. Falls back to 252 when there is not
    enough of a curve to measure.
    """
    if len(equity_curve) < 2:
        return _TRADING_DAYS_PER_YEAR
    gaps = [
        (equity_curve[i].ts - equity_curve[i - 1].ts).total_seconds()
        for i in range(1, len(equity_curve))
    ]
    positive = [g for g in gaps if g > 0.0]
    if not positive:
        return _TRADING_DAYS_PER_YEAR
    median_gap = statistics.median(positive)
    if median_gap <= 0.0:
        return _TRADING_DAYS_PER_YEAR
    return _SECONDS_PER_TRADING_YEAR / median_gap


# ---------------------------------------------------------------------------
# small numeric helpers - all of them return None rather than lying
# ---------------------------------------------------------------------------


def _safe_div(numerator: float, denominator: Optional[float]) -> Optional[float]:
    """Divide, or return None if the answer would be infinite or undefined."""
    if denominator is None or abs(denominator) < _EPS:
        return None
    return numerator / denominator


def _mean(values: Sequence[float]) -> Optional[float]:
    return statistics.fmean(values) if values else None


def _percentile(sorted_values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile of an already-sorted sequence."""
    n = len(sorted_values)
    if n == 1:
        return sorted_values[0]
    pos = q * (n - 1)
    low = int(math.floor(pos))
    high = min(low + 1, n - 1)
    frac = pos - low
    return sorted_values[low] * (1.0 - frac) + sorted_values[high] * frac


@dataclass(frozen=True)
class _Underwater:
    """Drawdown statistics for one equity path."""

    max_fraction: float = 0.0
    max_cash: float = 0.0
    longest_bars: int = 0
    longest_days: float = 0.0
    ulcer_index: float = 0.0


def _underwater(
    series: Sequence[tuple[Optional[datetime], float]],
    start_equity: float,
) -> _Underwater:
    """Walk an equity path once and collect every drawdown statistic.

    The running peak is seeded with the starting equity rather than the first
    observation, so a strategy that loses money immediately is correctly shown
    as underwater from the deposit rather than from its first bad mark.

    The longest stretch in bars and the longest stretch in days are tracked
    independently: with weekend gaps they are not always the same stretch, and
    reporting one measured over the other's window would be wrong.
    """
    if not series:
        return _Underwater()

    peak = start_equity
    peak_ts: Optional[datetime] = series[0][0]
    run_bars = 0
    max_fraction = 0.0
    max_cash = 0.0
    longest_bars = 0
    longest_days = 0.0
    sum_squared_pct = 0.0

    for ts, equity in series:
        if equity >= peak:
            peak = equity
            peak_ts = ts
            run_bars = 0
            drawdown = 0.0
        else:
            run_bars += 1
            cash = peak - equity
            drawdown = cash / peak if peak > _EPS else 0.0
            max_cash = max(max_cash, cash)
            max_fraction = max(max_fraction, drawdown)
            longest_bars = max(longest_bars, run_bars)
            if ts is not None and peak_ts is not None:
                span_days = (ts - peak_ts).total_seconds() / _SECONDS_PER_DAY
                longest_days = max(longest_days, span_days)
        sum_squared_pct += (drawdown * 100.0) ** 2

    return _Underwater(
        max_fraction=max_fraction,
        max_cash=max_cash,
        longest_bars=longest_bars,
        longest_days=longest_days,
        ulcer_index=math.sqrt(sum_squared_pct / len(series)),
    )


def _annualised_sharpe(returns: Sequence[float], periods_per_year: float) -> Optional[float]:
    """Mean over standard deviation, annualised. Risk-free rate assumed zero.

    Needs at least two returns for a sample standard deviation, and returns
    None when that deviation is zero - a flat curve has no Sharpe, and pouring
    a zero into the denominator to get `inf` would be the single most
    misleading number this package could print.
    """
    if len(returns) < 2:
        return None
    deviation = statistics.stdev(returns)
    if deviation < _EPS:
        return None
    return statistics.fmean(returns) / deviation * math.sqrt(periods_per_year)


def _annualised_sortino(returns: Sequence[float], periods_per_year: float) -> Optional[float]:
    """As Sharpe, but dividing by downside deviation only.

    Downside deviation is the root mean square of the negative returns taken
    over *all* periods, not only the negative ones: penalising a strategy for
    having few losing bars would reward doing nothing. Risk-free rate is zero,
    so the target return is zero.
    """
    if len(returns) < 2:
        return None
    downside = math.sqrt(sum(min(r, 0.0) ** 2 for r in returns) / len(returns))
    if downside < _EPS:
        return None
    return statistics.fmean(returns) / downside * math.sqrt(periods_per_year)


def _tail_ratio(returns: Sequence[float]) -> Optional[float]:
    """95th percentile over the absolute 5th percentile of period returns.

    Above 1.0 the right tail is fatter than the left. Undefined on fewer than
    20 observations, because a 5th percentile estimated from a handful of
    numbers is just the minimum wearing a statistical hat.
    """
    if len(returns) < 20:
        return None
    ordered = sorted(returns)
    upper = _percentile(ordered, 0.95)
    lower = _percentile(ordered, 0.05)
    if lower >= 0.0 or abs(lower) < _EPS:
        return None
    return upper / abs(lower)


def _max_consecutive_losses(trades: Sequence[Trade]) -> int:
    """Longest run of non-winning trades. A scratch trade breaks no streak -
    it does not feel like a win while it is happening."""
    worst = 0
    run = 0
    for trade in trades:
        if trade.is_win:
            run = 0
        else:
            run += 1
            worst = max(worst, run)
    return worst


def _break_even_edge_points(
    trades: Sequence[Trade], instrument: Optional[Instrument]
) -> Optional[float]:
    """Average round-turn cost, expressed as a price distance in points.

    This is the concrete question a strategy author should be asked first: how
    far does price have to move in your favour before the trade is worth doing
    at all? Cash costs are divided back out by position value, so the answer is
    comparable across account sizes and lot sizes.
    """
    if instrument is None or instrument.point <= 0.0:
        return None
    per_trade: list[float] = []
    for trade in trades:
        notional = trade.lots * trade.contract_size
        if notional <= _EPS:
            continue
        per_trade.append(trade.total_costs / notional / instrument.point)
    return _mean(per_trade)


def _trade_block(
    trades: Sequence[Trade],
    instrument: Optional[Instrument],
    years: Optional[float],
) -> dict:
    """Every trade-derived and cost-derived field, as a kwargs dict."""
    wins = [t.net_pnl for t in trades if t.is_win]
    losses = [t.net_pnl for t in trades if not t.is_win and t.net_pnl < 0.0]
    r_values = [r for r in (t.r_multiple for t in trades) if r is not None]

    avg_win = _mean(wins)
    avg_loss = _mean(losses)
    gross_pnl = math.fsum(t.gross_pnl for t in trades)
    total_costs = math.fsum(t.total_costs for t in trades)

    return {
        "n_trades": len(trades),
        "win_rate": _safe_div(float(len(wins)), float(len(trades))),
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "payoff_ratio": (
            _safe_div(avg_win, abs(avg_loss)) if avg_win is not None and avg_loss is not None else None
        ),
        "avg_r": _mean(r_values),
        "best_r": max(r_values) if r_values else None,
        "worst_r": min(r_values) if r_values else None,
        "max_consecutive_losses": _max_consecutive_losses(trades),
        "avg_bars_held": _mean([float(t.bars_held) for t in trades]),
        "trades_per_year": _safe_div(float(len(trades)), years),
        "profit_factor": _safe_div(math.fsum(wins), abs(math.fsum(losses)) if losses else None),
        "expectancy_r": _mean(r_values),
        "gross_pnl": gross_pnl,
        "net_pnl": math.fsum(t.net_pnl for t in trades),
        "total_costs": total_costs,
        "commission_total": math.fsum(t.commission for t in trades),
        "spread_total": math.fsum(t.spread_cost for t in trades),
        "slippage_total": math.fsum(t.slippage_cost for t in trades),
        "swap_total": math.fsum(t.swap for t in trades),
        # Magnitude, not sign: a strategy that lost money gross still has a
        # meaningful ratio of costs to trading result, and dividing by a
        # negative gross would print a negative drag, which reads as a discount.
        "cost_drag": _safe_div(total_costs, abs(gross_pnl) if gross_pnl != 0.0 else None),
        "break_even_edge_points": _break_even_edge_points(trades, instrument),
    }


def _cagr(initial_equity: float, final_equity: float, years: Optional[float]) -> Optional[float]:
    """Compound annual growth rate, or None where the maths does not exist.

    An account that ended at or below zero has no growth *rate* - there is no
    real root to take - and one measured over a few days has a rate so
    leveraged by the exponent that quoting it would be dishonest. Both return
    None instead of a large number.
    """
    if years is None or years <= 0.02:  # under ~5 trading days: not annualisable
        return None
    if initial_equity <= 0.0 or final_equity <= 0.0:
        return None
    return (final_equity / initial_equity) ** (1.0 / years) - 1.0


# ---------------------------------------------------------------------------
# public entry points
# ---------------------------------------------------------------------------


def summarize(result: BacktestResult, periods_per_year: Optional[float] = None) -> Metrics:
    """Reduce a completed run to a `Metrics`.

    `periods_per_year` overrides the inference from equity-curve timestamps;
    pass it when the curve is short or irregular enough that the median bar
    spacing is not the sampling interval you intend to annualise by.

    Risk-free rate is zero throughout - see the module docstring.
    """
    curve = result.equity_curve
    ppy = float(periods_per_year) if periods_per_year else infer_periods_per_year(curve)

    initial = result.initial_equity
    final = result.final_equity

    # Returns are seeded from the initial deposit so the first bar's P&L is not
    # silently dropped, and so a curve of length one still has one observation.
    returns: list[float] = []
    previous = initial
    for point in curve:
        if previous > _EPS:
            returns.append((point.equity - previous) / previous)
        previous = point.equity

    path = [(p.ts, p.equity) for p in curve]
    under = _underwater(path, initial)

    years: Optional[float] = (len(curve) / ppy) if curve and ppy > 0.0 else None
    cagr = _cagr(initial, final, years)

    fields = _trade_block(result.trades, result.instrument, years)
    fields.update(
        total_return=_safe_div(final - initial, initial if initial > 0.0 else None),
        cagr=cagr,
        final_equity=final,
        max_drawdown=under.max_fraction,
        max_drawdown_cash=under.max_cash,
        longest_drawdown_bars=under.longest_bars,
        longest_drawdown_days=under.longest_days,
        ulcer_index=under.ulcer_index,
        sharpe=_annualised_sharpe(returns, ppy),
        sortino=_annualised_sortino(returns, ppy),
        calmar=_safe_div(cagr, under.max_fraction) if cagr is not None else None,
        tail_ratio=_tail_ratio(returns),
        time_in_market=(
            sum(1 for p in curve if abs(p.exposure_lots) > _EPS) / len(curve) if curve else None
        ),
        periods_per_year=ppy,
        halted_early=result.halted_early,
        n_warnings=len(result.warnings),
    )
    return Metrics(**fields)


def summarize_trades_only(
    trades: Sequence[Trade],
    initial_equity: float = 10_000.0,
    instrument: Optional[Instrument] = None,
) -> Metrics:
    """Summarise a bare list of trades, with no equity curve behind it.

    Used by the validation module, which resamples and reorders trades to ask
    what the drawdown could have looked like had they arrived in a different
    sequence. There is no bar-by-bar mark-to-market in that world, so the
    equity path is reconstructed as a ladder of realised P&L: one step per
    closed trade.

    That ladder is strictly *kinder* than the real curve, because it never
    marks an open position to market and therefore never sees the intra-trade
    excursion that a real account's drawdown includes. Drawdowns computed here
    are a floor, not an estimate. Bar-based fields (`time_in_market`,
    `longest_drawdown_bars` in bar units) are left unset rather than faked, and
    annualisation is done in trades per year rather than bars per year.
    """
    trades = list(trades)

    equity = initial_equity
    path: list[tuple[Optional[datetime], float]] = []
    returns: list[float] = []
    for trade in trades:
        if equity > _EPS:
            returns.append(trade.net_pnl / equity)
        equity += trade.net_pnl
        path.append((trade.exit_ts, equity))

    under = _underwater(path, initial_equity)
    years = _span_years(trades)

    fields = _trade_block(trades, instrument, years)
    trades_per_year = fields.get("trades_per_year")

    fields.update(
        total_return=_safe_div(equity - initial_equity, initial_equity if initial_equity > 0.0 else None),
        cagr=_cagr(initial_equity, equity, years),
        final_equity=equity,
        max_drawdown=under.max_fraction,
        max_drawdown_cash=under.max_cash,
        longest_drawdown_bars=under.longest_bars,  # trades, not bars: see docstring
        longest_drawdown_days=under.longest_days,
        ulcer_index=under.ulcer_index,
        sharpe=_annualised_sharpe(returns, trades_per_year) if trades_per_year else None,
        sortino=_annualised_sortino(returns, trades_per_year) if trades_per_year else None,
        tail_ratio=_tail_ratio(returns),
        periods_per_year=trades_per_year,
    )
    cagr = fields["cagr"]
    fields["calmar"] = _safe_div(cagr, under.max_fraction) if cagr is not None else None
    return Metrics(**fields)


def _span_years(trades: Sequence[Trade]) -> Optional[float]:
    """Trading-time years spanned by a list of trades.

    The span between two timestamps is wall clock and includes weekends the
    market was shut for; scaling by 5/7 converts it to the same trading-time
    year the bar-based path uses, so `trades_per_year` means the same thing in
    both. None when the trades carry no usable span.
    """
    if len(trades) < 2:
        return None
    start = min(t.entry_ts for t in trades)
    end = max(t.exit_ts for t in trades)
    seconds = (end - start).total_seconds()
    if seconds <= 0.0:
        return None
    days = seconds / _SECONDS_PER_DAY * _TRADING_TIME_FRACTION
    return days / _TRADING_DAYS_PER_YEAR
