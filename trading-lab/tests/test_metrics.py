"""Metrics are tested on curves small enough to check on paper, and on inputs
degenerate enough to break a careless implementation.

Two things matter here and neither of them is precision.

The first is that the arithmetic is the arithmetic everybody means. Max
drawdown is measured from the running peak seeded at the *deposit*, not at the
first observation, so an account that loses money immediately is underwater
from the start rather than from its first bad mark. Profit factor is gross wins
over gross losses computed on **net** P&L, so costs are inside it.

The second, and the one this file spends most of its length on, is that every
ratio returns `None` instead of raising or printing `inf`. A strategy with no
losing trades has no profit factor; a flat curve has no Sharpe. Those are
findings, and a package that reports `999.0` or crashes has thrown away the
most interesting result it will ever produce.
"""

from __future__ import annotations

import math
import unittest
from datetime import datetime, timedelta, timezone

from tradelab.metrics import Metrics, infer_periods_per_year, summarize, summarize_trades_only
from tradelab.types import (
    BacktestResult,
    Direction,
    EquityPoint,
    ExitReason,
    Instrument,
    Trade,
)

START = datetime(2021, 1, 4, tzinfo=timezone.utc)


def curve(equities, start: datetime = START, minutes: int = 60) -> list:
    points = []
    peak = equities[0]
    for index, equity in enumerate(equities):
        peak = max(peak, equity)
        points.append(
            EquityPoint(
                ts=start + timedelta(minutes=minutes * index),
                equity=equity,
                balance=equity,
                exposure_lots=0.0,
                drawdown=max(0.0, (peak - equity) / peak),
            )
        )
    return points


def trade(net: float, index: int = 0, risk: float = 100.0) -> Trade:
    """A trade whose net P&L is exactly `net`, with no costs in the way."""
    entry = START + timedelta(hours=index)
    return Trade(
        direction=Direction.LONG,
        lots=0.1,
        entry_ts=entry,
        entry_price=1.10000,
        exit_ts=entry + timedelta(hours=1),
        exit_price=1.10000 + net / (0.1 * 100_000.0),
        exit_reason=ExitReason.SIGNAL,
        initial_stop=1.10000 - risk / (0.1 * 100_000.0),
        gross_pnl=net,
        bars_held=1,
        contract_size=100_000.0,
    )


def result_from(equities, trades, initial: float = 10_000.0) -> BacktestResult:
    return BacktestResult(
        trades=list(trades),
        equity_curve=curve(equities),
        initial_equity=initial,
        instrument=Instrument(),
    )


class DrawdownTests(unittest.TestCase):
    def test_max_drawdown_is_hand_computable(self) -> None:
        # peak 12,000 then 9,000: 3,000 of 12,000 is 25%.
        metrics = summarize(result_from([10_000, 12_000, 9_000, 11_000], []))
        self.assertAlmostEqual(metrics.max_drawdown, 0.25, places=12)
        self.assertAlmostEqual(metrics.max_drawdown_cash, 3_000.0, places=9)

    def test_the_peak_is_seeded_at_the_deposit(self) -> None:
        # Never above 10,000, so the drawdown is measured from the deposit.
        metrics = summarize(result_from([9_000, 8_000, 8_500], []))
        self.assertAlmostEqual(metrics.max_drawdown, 0.20, places=12)
        self.assertAlmostEqual(metrics.max_drawdown_cash, 2_000.0, places=9)

    def test_a_monotone_curve_has_no_drawdown(self) -> None:
        metrics = summarize(result_from([10_000, 11_000, 12_000], []))
        self.assertEqual(metrics.max_drawdown, 0.0)
        self.assertEqual(metrics.max_drawdown_cash, 0.0)
        self.assertEqual(metrics.ulcer_index, 0.0)

    def test_the_longest_underwater_stretch_is_counted_in_bars(self) -> None:
        metrics = summarize(result_from([10_000, 12_000, 11_000, 11_500, 11_800, 13_000], []))
        self.assertEqual(metrics.longest_drawdown_bars, 3)

    def test_the_ulcer_index_is_quoted_in_percent(self) -> None:
        # Two points at 0% and two at 50% -> rms = sqrt((0+0+2500+2500)/4) = 35.36
        metrics = summarize(result_from([10_000, 10_000, 5_000, 5_000], []))
        self.assertAlmostEqual(metrics.ulcer_index, math.sqrt(5000.0 / 4.0), places=9)


class TradeStatisticTests(unittest.TestCase):
    def setUp(self) -> None:
        self.trades = [trade(100.0, 0), trade(200.0, 1), trade(-50.0, 2), trade(-100.0, 3)]
        self.metrics = summarize(result_from([10_000, 10_100, 10_300, 10_250, 10_150], self.trades))

    def test_profit_factor_is_gross_wins_over_gross_losses(self) -> None:
        self.assertAlmostEqual(self.metrics.profit_factor, 300.0 / 150.0, places=12)

    def test_win_rate_and_averages(self) -> None:
        self.assertAlmostEqual(self.metrics.win_rate, 0.5, places=12)
        self.assertAlmostEqual(self.metrics.avg_win, 150.0, places=9)
        self.assertAlmostEqual(self.metrics.avg_loss, -75.0, places=9)
        self.assertAlmostEqual(self.metrics.payoff_ratio, 2.0, places=12)

    def test_r_multiples_use_the_recorded_contract_size(self) -> None:
        # Risk is 100 of account currency by construction.
        self.assertAlmostEqual(self.metrics.best_r, 2.0, places=9)
        self.assertAlmostEqual(self.metrics.worst_r, -1.0, places=9)
        self.assertAlmostEqual(self.metrics.avg_r, 0.375, places=9)

    def test_max_consecutive_losses(self) -> None:
        self.assertEqual(self.metrics.max_consecutive_losses, 2)

    def test_net_and_gross_are_both_reported(self) -> None:
        self.assertAlmostEqual(self.metrics.net_pnl, 150.0, places=9)
        self.assertAlmostEqual(self.metrics.gross_pnl, 150.0, places=9)


class UndefinedRatherThanWrongTests(unittest.TestCase):
    """Every ratio returns None on degenerate input. None of them raise."""

    def test_an_empty_result_summarises_without_raising(self) -> None:
        metrics = summarize(BacktestResult())
        self.assertEqual(metrics.n_trades, 0)
        for name in ("sharpe", "sortino", "calmar", "profit_factor", "cagr",
                     "win_rate", "avg_win", "avg_loss", "payoff_ratio", "avg_r",
                     "best_r", "worst_r", "tail_ratio", "trades_per_year",
                     "cost_drag", "time_in_market"):
            with self.subTest(field=name):
                self.assertIsNone(getattr(metrics, name))

    def test_a_single_equity_point_has_no_sharpe(self) -> None:
        metrics = summarize(result_from([10_000], []))
        self.assertIsNone(metrics.sharpe)
        self.assertIsNone(metrics.sortino)

    def test_a_flat_curve_has_no_sharpe(self) -> None:
        metrics = summarize(result_from([10_000] * 50, []))
        self.assertIsNone(metrics.sharpe)
        self.assertIsNone(metrics.sortino)

    def test_no_losing_trades_means_no_profit_factor(self) -> None:
        metrics = summarize(result_from([10_000, 10_100, 10_300], [trade(100.0, 0), trade(200.0, 1)]))
        self.assertIsNone(metrics.profit_factor)
        self.assertTrue(any("look-ahead" in c for c in metrics.caveats()))

    def test_a_ruined_account_has_no_cagr(self) -> None:
        metrics = summarize(result_from([10_000] + [0.0] * 500, []))
        self.assertIsNone(metrics.cagr)
        self.assertIsNone(metrics.calmar)

    def test_tail_ratio_needs_twenty_observations(self) -> None:
        self.assertIsNone(summarize(result_from([10_000, 10_100, 9_900], [])).tail_ratio)

    def test_no_ratio_is_ever_infinite(self) -> None:
        metrics = summarize(result_from([10_000, 20_000] + [10_000] * 30, [trade(1000.0)]))
        for name in ("sharpe", "sortino", "calmar", "profit_factor", "cagr",
                     "tail_ratio", "cost_drag", "expectancy_r"):
            value = getattr(metrics, name)
            with self.subTest(field=name):
                self.assertTrue(value is None or math.isfinite(value), f"{name} = {value}")

    def test_summarize_trades_only_handles_an_empty_list(self) -> None:
        metrics = summarize_trades_only([], initial_equity=5_000.0)
        self.assertEqual(metrics.n_trades, 0)
        self.assertEqual(metrics.final_equity, 5_000.0)
        self.assertIsNone(metrics.sharpe)


class PeriodicityTests(unittest.TestCase):
    def test_hourly_bars_imply_a_five_day_trading_year(self) -> None:
        self.assertAlmostEqual(infer_periods_per_year(curve([10_000] * 50)), 252.0 * 24.0, places=6)

    def test_daily_bars_imply_252_periods(self) -> None:
        points = curve([10_000] * 50, minutes=60 * 24)
        self.assertAlmostEqual(infer_periods_per_year(points), 252.0, places=6)

    def test_the_median_gap_is_used_so_weekends_do_not_deflate_it(self) -> None:
        points = list(curve([10_000] * 40))
        # Insert a 48-hour hole; the median hourly spacing must survive it.
        tail = [
            EquityPoint(ts=p.ts + timedelta(hours=48), equity=p.equity, balance=p.balance,
                        exposure_lots=0.0, drawdown=0.0)
            for p in curve([10_000] * 10, start=points[-1].ts + timedelta(hours=1))
        ]
        self.assertAlmostEqual(infer_periods_per_year(points + tail), 252.0 * 24.0, places=6)

    def test_an_empty_curve_falls_back_rather_than_dividing_by_zero(self) -> None:
        self.assertEqual(infer_periods_per_year([]), 252.0)

    def test_an_explicit_basis_overrides_the_inference(self) -> None:
        metrics = summarize(result_from([10_000, 10_100, 9_950, 10_200], []), periods_per_year=252.0)
        self.assertEqual(metrics.periods_per_year, 252.0)


class CaveatTests(unittest.TestCase):
    def test_caveats_are_never_silent(self) -> None:
        self.assertTrue(summarize(BacktestResult()).caveats())
        self.assertTrue(Metrics().caveats())

    def test_an_idle_account_says_so_first(self) -> None:
        caveats = summarize(result_from([10_000] * 10, [])).caveats()
        self.assertIn("No trades were taken", caveats[0])

    def test_a_thin_sample_is_called_out(self) -> None:
        caveats = summarize(result_from([10_000, 10_100], [trade(100.0)])).caveats()
        self.assertTrue(any("cannot" in c and "noise" in c for c in caveats))

    def test_a_long_losing_streak_is_called_out(self) -> None:
        trades = [trade(-10.0, i) for i in range(8)]
        caveats = summarize(result_from([10_000] * 9, trades)).caveats()
        self.assertTrue(any("losing trades in a row" in c for c in caveats))

    def test_an_early_halt_is_called_out(self) -> None:
        run = result_from([10_000, 3_000], [trade(-7000.0)])
        run.halted_early = True
        self.assertTrue(any("ruin threshold" in c for c in summarize(run).caveats()))


if __name__ == "__main__":
    unittest.main()
