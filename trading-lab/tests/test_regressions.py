"""Regression tests, one per defect found in the adversarial audit.

Every test in this file failed before its fix, and every one of the defects
below shipped a *working* backtest that simply reported a better number than
the account would have seen. That is the failure mode this package exists to
prevent, so each one gets a named test that pins the corrected behaviour and a
comment describing what went wrong, rather than being quietly folded into the
suite elsewhere.

The pattern worth noticing: none of these were crashes. Three of them were
found only by asking "is this number too good?" of code whose 228 tests were
green.
"""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from tradelab.costs import CostModel
from tradelab.engine import Backtester
from tradelab.strategy import Context, Strategy
from tradelab.types import (
    Bar,
    CostConfig,
    Direction,
    ExitReason,
    Instrument,
    OrderIntent,
    RiskConfig,
)

START = datetime(2021, 3, 2, 8, 0, tzinfo=timezone.utc)  # a Tuesday morning
NO_COMMISSION = Instrument(commission_per_lot_per_side=0.0)


def bars_from(rows, start: datetime = START, minutes: int = 60) -> list:
    return [
        Bar(ts=start + timedelta(minutes=minutes * i), open=o, high=h, low=lo, close=c)
        for i, (o, h, lo, c) in enumerate(rows)
    ]


def flat(price: float):
    return (price, price, price, price)


class Scripted(Strategy):
    """Emits pre-set intents on pre-set bar indices. No indicators, no opinions."""

    name = "scripted"

    def __init__(self, script: dict) -> None:
        super().__init__()
        self.script = script
        self._i = -1

    def on_bar(self, ctx: Context):
        self._i += 1
        return self.script.get(self._i)


class BidPricedSpreadTests(unittest.TestCase):
    """A round turn on bid-priced bars must still pay a spread.

    `CostConfig.half_spread()` returned `spread / 2` on mid data and **zero**
    on bid data, and `fill()` applied that same symmetric number to both legs.
    On bid-priced bars - which is what MetaTrader exports, and therefore what
    anyone testing their own broker's data uses - neither leg paid anything.
    `stress()` could not restore it either, because doubling zero is zero.
    """

    def _round_turn(self, costs: CostConfig) -> float:
        bars = bars_from([flat(1.10000)] * 3)
        script = {0: OrderIntent(direction=Direction.LONG, lots=1.0),
                  1: OrderIntent(direction=Direction.FLAT)}
        result = Backtester(instrument=NO_COMMISSION, costs=costs).run(Scripted(script), bars)
        self.assertEqual(len(result.trades), 1)
        return result.trades[0].spread_cost

    def test_bid_priced_bars_are_charged_the_same_round_turn_as_mid(self) -> None:
        mid = CostConfig(spread_points=2.0, slippage_points=0.0, price_is_mid=True)
        bid = CostConfig(spread_points=2.0, slippage_points=0.0, price_is_mid=False)
        self.assertAlmostEqual(self._round_turn(mid), 2.00, places=9)
        self.assertAlmostEqual(self._round_turn(bid), 2.00, places=9)

    def test_stressing_costs_still_bites_on_bid_priced_bars(self) -> None:
        bid = CostConfig(spread_points=2.0, slippage_points=0.0, price_is_mid=False)
        self.assertAlmostEqual(self._round_turn(bid.stress(5.0)), 10.00, places=9)


class CostsChargedOnceTests(unittest.TestCase):
    """Spread and slippage were subtracted twice.

    `gross_pnl` was measured between the two *filled* prices, which already
    carry both costs, and `net_pnl` then subtracted them again. Net P&L and
    the equity curve came out understated, every reported cost figure came out
    understated - the flattering half - and a freshly opened position was
    instantly marked down by twice its entry costs, which fed drawdown, Ulcer
    and Sharpe.
    """

    def setUp(self) -> None:
        # Mid moves 1.10000 -> 1.10100: a clean 100.00 on 1.0 lot before costs.
        bars = bars_from([flat(1.10000), flat(1.10000), flat(1.10100)])
        script = {0: OrderIntent(direction=Direction.LONG, lots=1.0),
                  1: OrderIntent(direction=Direction.FLAT)}
        costs = CostConfig(spread_points=1.0, slippage_points=2.0)
        self.result = Backtester(costs=costs).run(Scripted(script), bars)
        self.trade = self.result.trades[0]

    def test_gross_is_the_costless_move(self) -> None:
        self.assertAlmostEqual(self.trade.gross_pnl, 100.00, places=6)

    def test_net_is_gross_less_each_cost_exactly_once(self) -> None:
        self.assertAlmostEqual(self.trade.net_pnl, 88.00, places=6)
        self.assertAlmostEqual(
            self.trade.net_pnl,
            self.trade.gross_pnl - self.trade.total_costs,
            places=9,
        )

    def test_the_realised_fills_agree_with_the_cost_breakdown(self) -> None:
        realised = (self.trade.exit_price - self.trade.entry_price) * 100_000.0
        self.assertAlmostEqual(
            realised,
            self.trade.gross_pnl - self.trade.spread_cost - self.trade.slippage_cost,
            places=6,
        )

    def test_a_new_position_is_marked_down_by_its_entry_costs_only(self) -> None:
        # Entry leg: 0.5 pt spread + 2.0 pt slippage = 2.50, plus 3.50
        # commission = 6.00. Double-counting made this 8.50.
        opened = self.result.equity_curve[1]
        self.assertAlmostEqual(opened.equity, 10_000.00 - 6.00, places=6)


class WeekendFinancingTests(unittest.TestCase):
    """Swap accrued nine nights per calendar week instead of seven.

    Rollovers were emitted every calendar day *including* Saturday and Sunday,
    and Wednesday was tripled on top. Real FX skips the weekend rollovers
    precisely because the Wednesday triple settles the weekend value date, so
    the weekend was being paid for twice - a 29% error on the financing line,
    which for a slow carry position is the entire result.
    """

    def setUp(self) -> None:
        self.model = CostModel(CostConfig(), Instrument())

    def test_a_week_costs_seven_nights(self) -> None:
        monday = datetime(2021, 3, 1, 0, 0, tzinfo=timezone.utc)
        self.assertEqual(monday.weekday(), 0)
        crossings = self.model.rollovers_between(monday, monday + timedelta(days=7))
        one_night = self.model.swap(Direction.LONG, 1.0, monday)
        total = sum(self.model.swap(Direction.LONG, 1.0, c) for c in crossings)
        self.assertAlmostEqual(total / one_night, 7.0, places=9)

    def test_a_positive_carry_short_is_not_over_credited(self) -> None:
        monday = datetime(2021, 3, 1, 0, 0, tzinfo=timezone.utc)
        crossings = self.model.rollovers_between(monday, monday + timedelta(days=7))
        total = sum(self.model.swap(Direction.SHORT, 1.0, c) for c in crossings)
        self.assertAlmostEqual(total, 7 * 0.20, places=9)  # was 9 * 0.20 = 1.80


class ExcursionOnTheExitBarTests(unittest.TestCase):
    """MAE and MFE were never recorded on the bar a trade exited on.

    Excursions were updated at the mark, under a "position is still open"
    guard, but every exit path clears the position earlier in the same bar. So
    the bar that stopped a trade out - by construction its worst bar - counted
    for nothing, and a trade stopped out at maximum heat reported an MAE of
    zero.
    """

    def test_a_stopped_out_trade_records_the_heat_that_stopped_it(self) -> None:
        bars = bars_from([
            flat(1.10000),
            flat(1.10000),
            (1.10000, 1.10000, 1.08000, 1.08000),  # dives through the stop
        ])
        script = {0: OrderIntent(direction=Direction.LONG, lots=1.0, stop_loss=1.09000)}
        result = Backtester(
            instrument=NO_COMMISSION,
            costs=CostConfig(spread_points=0.0, slippage_points=0.0),
            risk=RiskConfig(max_daily_loss=None),
        ).run(Scripted(script), bars)
        trade = result.trades[0]
        self.assertEqual(trade.exit_reason, ExitReason.STOP_LOSS)
        # Low of 1.08000 against a 1.10000 entry on 1.0 lot = 2,000.00 of heat.
        self.assertAlmostEqual(trade.mae, 2_000.00, places=6)
        self.assertGreaterEqual(trade.mae, abs(trade.net_pnl))


class LotRoundingTests(unittest.TestCase):
    """`max_lot` was clamped after the step flooring, so the cap itself could
    be returned even when it was not a whole number of lot steps - a size the
    broker rejects."""

    def test_the_cap_is_still_a_whole_number_of_steps(self) -> None:
        inst = Instrument(min_lot=0.01, lot_step=0.01, max_lot=0.055)
        self.assertAlmostEqual(inst.round_lots(1.0), 0.05, places=9)

    def test_every_result_is_a_multiple_of_the_step(self) -> None:
        inst = Instrument(min_lot=0.01, lot_step=0.03, max_lot=0.29)
        for raw in (0.001, 0.03, 0.05, 0.29, 1.0, 12.5):
            lots = inst.round_lots(raw)
            if lots > 0.0:
                self.assertAlmostEqual(lots / 0.03, round(lots / 0.03), places=6)
                self.assertLessEqual(lots, 0.29)


class StopDistanceGuardTests(unittest.TestCase):
    """The `min_stop_points` guard measured to the filled price, which sits a
    spread and a slippage further from the stop than the quoted price does. A
    stop inside the configured minimum could pass it."""

    def test_a_stop_inside_the_minimum_is_refused_at_the_quoted_price(self) -> None:
        # Stop is 47.5 points below the bar's open, inside a 50-point minimum,
        # but 50.0 points from the fill - which used to let it through.
        bars = bars_from([flat(1.10000), flat(1.10000), flat(1.10000)])
        script = {0: OrderIntent(direction=Direction.LONG, stop_loss=1.10000 - 47.5e-5)}
        result = Backtester(
            costs=CostConfig(spread_points=1.0, slippage_points=2.0),
            risk=RiskConfig(min_stop_points=50.0),
        ).run(Scripted(script), bars)
        self.assertEqual(result.trades, [])
        self.assertTrue(any("inside the" in w for w in result.warnings), result.warnings)


class EntryBarProtectionTests(unittest.TestCase):
    """A position was never checked against the stop on its own entry bar.

    Protective exits ran before the pending intent filled, so a position opened
    at bar i's open only became protected at bar i+1. Any strategy whose stop
    sits inside a typical bar range was effectively backtested without a stop -
    at `atr_stop_mult=0.5` the audit measured 468 of 1183 trades breaching an
    unfilled stop on their entry bar, manufacturing the equity curve out of the
    hole. The engine even recorded the excursion it declined to act on.
    """

    def test_a_stop_inside_the_entry_bar_is_filled_on_that_bar(self) -> None:
        bars = bars_from([
            flat(1.10000),
            (1.10000, 1.10100, 1.05000, 1.10000),  # entry bar, dives past the stop
            flat(1.10000),
        ])
        script = {0: OrderIntent(direction=Direction.LONG, stop_loss=1.09000)}
        result = Backtester(
            instrument=NO_COMMISSION,
            costs=CostConfig(spread_points=0.0, slippage_points=0.0,
                             swap_long_points=0.0, swap_short_points=0.0),
            risk=RiskConfig(risk_per_trade=0.005, max_daily_loss=None),
        ).run(Scripted(script), bars)
        trade = result.trades[0]
        self.assertEqual(trade.exit_reason, ExitReason.STOP_LOSS)
        self.assertAlmostEqual(trade.exit_price, 1.09000, places=9)
        # Sized to risk 0.5% of 10,000 over a 100-point stop.
        self.assertAlmostEqual(trade.net_pnl, -50.00, places=2)

    def test_the_entry_bar_check_does_not_use_the_open_gap_branch(self) -> None:
        # The account just dealt at this open, so it cannot also have gapped
        # through a level on the way to it. A stop above the open of a long -
        # only reachable with min_stop_points relaxed - must not fill at entry.
        bars = bars_from([flat(1.10000), (1.10000, 1.10000, 1.09990, 1.10000), flat(1.10000)])
        script = {0: OrderIntent(direction=Direction.LONG, lots=1.0, stop_loss=1.09995)}
        result = Backtester(
            instrument=NO_COMMISSION,
            costs=CostConfig(spread_points=0.0, slippage_points=0.0),
            risk=RiskConfig(min_stop_points=0.0, max_daily_loss=None),
        ).run(Scripted(script), bars)
        self.assertEqual(result.trades[0].exit_reason, ExitReason.STOP_LOSS)
        self.assertAlmostEqual(result.trades[0].exit_price, 1.09995, places=9)


class SurrogateGeometryTests(unittest.TestCase):
    """The permutation surrogate corrupted every bar's geometry.

    Each surrogate bar was built by scaling the *original* bar's open/high/low
    around a close taken from an unrelated point in the shuffled path. That
    carried the high/close and low/close ratios over verbatim while the close
    followed a different walk, inflating mean true range ~38% and manufacturing
    an opening gap on every bar. The null became "a materially more hostile
    market" rather than "the same market, unordered", so surrogates lost by
    construction and the test reported significance for strategies with no edge
    - it fired at p<0.05 on 4 of 14 no-edge seeds.
    """

    def setUp(self) -> None:
        import random as _random

        from tradelab.data import synthetic_random_walk
        from tradelab.validation import surrogate_series

        self.real = synthetic_random_walk(1200, seed=3)
        self.fake = surrogate_series(self.real, _random.Random(11))

    @staticmethod
    def _mean_true_range(bars) -> float:
        return sum(b.high - b.low for b in bars) / len(bars)

    @staticmethod
    def _mean_gap(bars) -> float:
        return sum(abs(bars[i].open - bars[i - 1].close) for i in range(1, len(bars))) / (len(bars) - 1)

    def test_intrabar_range_is_preserved_not_inflated(self) -> None:
        real, fake = self._mean_true_range(self.real), self._mean_true_range(self.fake)
        self.assertLess(abs(fake - real) / real, 0.10, f"real {real:.7f} vs surrogate {fake:.7f}")

    def test_no_gaps_are_manufactured(self) -> None:
        # This generator emits bars whose open is exactly the previous close,
        # so the surrogate must too. The old construction gapped every bar by
        # about a full bar range.
        self.assertAlmostEqual(self._mean_gap(self.real), 0.0, places=12)
        self.assertAlmostEqual(self._mean_gap(self.fake), 0.0, places=12)

    def test_the_multiset_of_returns_survives(self) -> None:
        import math

        def returns(bars):
            return sorted(round(math.log(bars[i].close / bars[i - 1].close), 12)
                          for i in range(1, len(bars)))

        self.assertEqual(returns(self.real), returns(self.fake))

    def test_every_surrogate_bar_is_geometrically_valid(self) -> None:
        for bar in self.fake:
            self.assertLessEqual(bar.low, min(bar.open, bar.close))
            self.assertGreaterEqual(bar.high, max(bar.open, bar.close))

    def test_a_no_edge_strategy_is_not_called_significant(self) -> None:
        from tradelab.data import synthetic_random_walk
        from tradelab.strategies import DonchianBreakout
        from tradelab.validation import permutation_test

        result = permutation_test(
            lambda p: DonchianBreakout(**p),
            {"entry_lookback": 20, "exit_lookback": 10},
            synthetic_random_walk(900, seed=5),
            lambda: Backtester(),
            n_perms=25,
            seed=17,
        )
        # Data with no edge by construction must not clear the bar.
        self.assertGreater(result.p_value, 0.05)


class UnscoredSurrogateTests(unittest.TestCase):
    """Surrogates that could not be scored stayed in the p-value denominator,
    silently counted as "did not beat the strategy" and biasing p downward."""

    def test_unscored_surrogates_leave_the_denominator(self) -> None:
        from tradelab.data import synthetic_random_walk
        from tradelab.strategies import DonchianBreakout
        from tradelab.validation import permutation_test

        result = permutation_test(
            lambda p: DonchianBreakout(**p),
            {"entry_lookback": 20, "exit_lookback": 10},
            synthetic_random_walk(900, seed=5),
            lambda: Backtester(),
            n_perms=25,
            seed=17,
        )
        scored = result.n_perms - result.n_unscored
        self.assertAlmostEqual(
            result.p_value, (1 + result.n_at_least_observed) / (1 + scored), places=12
        )


class WidenedStopTests(unittest.TestCase):
    """Re-emitting an entry intent with a looser stop rewrote the live stop but
    left `initial_stop` alone, so every R-multiple was measured against risk the
    trade no longer carried - silently, with no warning."""

    def test_widening_a_stop_moves_the_risk_it_is_measured_against(self) -> None:
        bars = bars_from([flat(1.10000)] * 4)
        script = {
            0: OrderIntent(direction=Direction.LONG, lots=1.0, stop_loss=1.09000),
            1: OrderIntent(direction=Direction.LONG, lots=1.0, stop_loss=1.07000),
        }
        result = Backtester(
            instrument=NO_COMMISSION,
            costs=CostConfig(spread_points=0.0, slippage_points=0.0),
            risk=RiskConfig(max_daily_loss=None),
        ).run(Scripted(script), bars)
        trade = result.trades[0]
        self.assertAlmostEqual(trade.initial_stop, 1.07000, places=9)
        self.assertAlmostEqual(trade.risk_amount, 3_000.00, places=4)
        self.assertTrue(any("widened" in w for w in result.warnings), result.warnings)

    def test_tightening_a_stop_leaves_the_original_risk_alone(self) -> None:
        bars = bars_from([flat(1.10000)] * 4)
        script = {
            0: OrderIntent(direction=Direction.LONG, lots=1.0, stop_loss=1.09000),
            1: OrderIntent(direction=Direction.LONG, lots=1.0, stop_loss=1.09500),
        }
        result = Backtester(
            instrument=NO_COMMISSION,
            costs=CostConfig(spread_points=0.0, slippage_points=0.0),
            risk=RiskConfig(max_daily_loss=None),
        ).run(Scripted(script), bars)
        self.assertAlmostEqual(result.trades[0].initial_stop, 1.09000, places=9)
        self.assertFalse(any("widened" in w for w in result.warnings))


if __name__ == "__main__":
    unittest.main()
