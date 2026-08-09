"""Costs are the difference between a backtest and a business, so they are
checked against arithmetic done by hand rather than against the code that
produces them.

The centrepiece is `RoundTripTests`: one forced long, held for a single bar,
with every component of its cost worked out on paper in the docstring and
asserted individually. A test that only checks the total would pass with the
spread attributed to slippage, and the attribution is the part a report reader
actually uses to decide which assumption to argue with.

The rest of the file covers the three ways cost models usually go wrong:
charging something for doing nothing, being insensitive to the parameter the
whole result depends on, and forgetting that a position held over a weekend is
financed for every night inside it rather than for the one bar boundary the
data happens to show.
"""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from typing import Optional

from tradelab.costs import CostModel
from tradelab.data import synthetic_trending
from tradelab.engine import Backtester
from tradelab.strategies import AlwaysFlat
from tradelab.strategy import Context, Strategy
from tradelab.types import Bar, CostConfig, Direction, Instrument, OrderIntent, RiskConfig

INSTRUMENT = Instrument()          # 5-digit EURUSD, 3.50 per lot per side
COSTS = CostConfig()               # 1.0 point spread, 2.0 points slippage
POINT = INSTRUMENT.point


def bars_at(times, rows) -> list:
    return [
        Bar(ts=ts, open=o, high=h, low=lo, close=c)
        for ts, (o, h, lo, c) in zip(times, rows)
    ]


def hourly(start: datetime, count: int) -> list:
    return [start + timedelta(hours=i) for i in range(count)]


class Scripted(Strategy):
    """Emits fixed intents at fixed bar indices."""

    name = "scripted"

    def __init__(self, script: dict) -> None:
        super().__init__()
        self.script = dict(script)

    def on_bar(self, ctx: Context) -> Optional[OrderIntent]:
        return self.script.get(len(ctx.bars) - 1)


class FillPricingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = CostModel(COSTS, INSTRUMENT)

    def test_buying_lifts_the_offer_and_pays_slippage(self) -> None:
        fill = self.model.fill(1.10000, buying=True)
        self.assertAlmostEqual(fill.price, 1.10000 + 2.5 * POINT, places=10)
        self.assertEqual(fill.spread_points, 0.5)
        self.assertEqual(fill.slippage_points, 2.0)

    def test_selling_hits_the_bid_and_pays_slippage(self) -> None:
        fill = self.model.fill(1.10000, buying=False)
        self.assertAlmostEqual(fill.price, 1.10000 - 2.5 * POINT, places=10)

    def test_a_limit_pays_the_half_spread_but_no_slippage(self) -> None:
        fill = self.model.fill(1.10000, buying=False, is_limit=True)
        self.assertAlmostEqual(fill.price, 1.10000 - 0.5 * POINT, places=10)
        self.assertEqual(fill.slippage_points, 0.0)

    def test_bid_priced_data_pays_no_half_spread(self) -> None:
        model = CostModel(CostConfig(price_is_mid=False), INSTRUMENT)
        fill = model.fill(1.10000, buying=True)
        self.assertEqual(fill.spread_points, 0.0)
        self.assertAlmostEqual(fill.price, 1.10000 + 2.0 * POINT, places=10)

    def test_commission_is_one_side_only(self) -> None:
        self.assertAlmostEqual(self.model.commission(2.0), 7.00, places=10)

    def test_points_convert_to_cash_through_one_function(self) -> None:
        # 1 point on 1 lot of a 100,000-unit contract at 1e-5 per point = 1.00
        self.assertAlmostEqual(self.model.cash(1.0, 1.0), 1.00, places=10)
        self.assertAlmostEqual(self.model.cash(-0.8, 0.5), -0.40, places=10)


class SwapTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = CostModel(COSTS, INSTRUMENT)

    def test_a_long_pays_and_a_short_receives_on_these_defaults(self) -> None:
        monday = datetime(2021, 3, 1, 21, 0, tzinfo=timezone.utc)
        self.assertAlmostEqual(self.model.swap(Direction.LONG, 1.0, monday), -0.80, places=10)
        self.assertAlmostEqual(self.model.swap(Direction.SHORT, 1.0, monday), 0.20, places=10)
        self.assertEqual(self.model.swap(Direction.FLAT, 1.0, monday), 0.0)

    def test_the_triple_swap_weekday_is_charged_three_times(self) -> None:
        wednesday = datetime(2021, 3, 3, 21, 0, tzinfo=timezone.utc)
        self.assertEqual(wednesday.weekday(), COSTS.triple_swap_weekday)
        tuesday = wednesday - timedelta(days=1)
        single = self.model.swap(Direction.LONG, 1.0, tuesday)
        triple = self.model.swap(Direction.LONG, 1.0, wednesday)
        self.assertAlmostEqual(triple, 3.0 * single, places=10)
        self.assertAlmostEqual(triple, -2.40, places=10)

    def test_a_friday_to_monday_hold_crosses_three_rollovers(self) -> None:
        friday = datetime(2021, 3, 5, 20, 0, tzinfo=timezone.utc)
        monday = datetime(2021, 3, 8, 20, 0, tzinfo=timezone.utc)
        self.assertEqual(friday.weekday(), 4)
        crossings = self.model.rollovers_between(friday, monday)
        self.assertEqual([c.strftime("%a %d %H:%M") for c in crossings],
                         ["Fri 05 21:00", "Sat 06 21:00", "Sun 07 21:00"])

    def test_one_bar_inside_a_day_crosses_nothing(self) -> None:
        start = datetime(2021, 3, 2, 10, 0, tzinfo=timezone.utc)
        self.assertEqual(self.model.rollovers_between(start, start + timedelta(hours=1)), [])

    def test_the_rollover_instant_itself_is_crossed_once(self) -> None:
        start = datetime(2021, 3, 2, 20, 30, tzinfo=timezone.utc)
        self.assertEqual(len(self.model.rollovers_between(start, start + timedelta(hours=1))), 1)

    def test_time_running_backwards_crosses_nothing(self) -> None:
        later = datetime(2021, 3, 5, 20, 0, tzinfo=timezone.utc)
        self.assertEqual(self.model.rollovers_between(later, later - timedelta(days=2)), [])


class RoundTripTests(unittest.TestCase):
    """One long, 1.0 lot, entered at 1.10000 mid and closed at 1.10100 mid.

    Worked by hand, on the package defaults (point 1e-5, contract 100,000,
    spread 1.0 point, slippage 2.0 points, commission 3.50 per lot per side):

        entry fill  = 1.10000 + (0.5 + 2.0) * 1e-5 = 1.1000250
        exit  fill  = 1.10100 - (0.5 + 2.0) * 1e-5 = 1.1009750
        gross       = (1.1009750 - 1.1000250) * 1.0 * 100,000 =  95.00
        commission  = 3.50 * 2 sides                          =   7.00
        spread      = 0.5 points * 2 legs * 1.00 per point    =   1.00
        slippage    = 2.0 points * 2 legs * 1.00 per point    =   4.00
        swap        = no rollover crossed inside one morning  =   0.00
        net         = 95.00 - 7.00 - 1.00 - 4.00              =  83.00

    Every one of those numbers is asserted separately below. The bars sit
    between 10:00 and 12:00 UTC so that the 21:00 rollover is never crossed.
    """

    def setUp(self) -> None:
        times = hourly(datetime(2021, 3, 2, 10, 0, tzinfo=timezone.utc), 3)
        rows = [
            (1.10000, 1.10000, 1.10000, 1.10000),
            (1.10000, 1.10100, 1.09990, 1.10050),
            (1.10100, 1.10100, 1.10100, 1.10100),
        ]
        script = {
            0: OrderIntent(direction=Direction.LONG, lots=1.0, tag="hand_checked"),
            1: OrderIntent(direction=Direction.FLAT),
        }
        self.result = Backtester(instrument=INSTRUMENT, costs=COSTS).run(
            Scripted(script), bars_at(times, rows)
        )
        self.assertEqual(len(self.result.trades), 1)
        self.trade = self.result.trades[0]

    def test_fill_prices(self) -> None:
        self.assertAlmostEqual(self.trade.entry_price, 1.1000250, places=9)
        self.assertAlmostEqual(self.trade.exit_price, 1.1009750, places=9)

    def test_gross_pnl(self) -> None:
        self.assertAlmostEqual(self.trade.gross_pnl, 95.00, places=6)

    def test_commission_covers_both_sides(self) -> None:
        self.assertAlmostEqual(self.trade.commission, 7.00, places=10)

    def test_spread_cost(self) -> None:
        self.assertAlmostEqual(self.trade.spread_cost, 1.00, places=10)

    def test_slippage_cost(self) -> None:
        self.assertAlmostEqual(self.trade.slippage_cost, 4.00, places=10)

    def test_no_swap_inside_a_single_morning(self) -> None:
        self.assertEqual(self.trade.swap, 0.0)

    def test_total_costs_and_net_pnl(self) -> None:
        self.assertAlmostEqual(self.trade.total_costs, 12.00, places=6)
        self.assertAlmostEqual(self.trade.net_pnl, 83.00, places=6)

    def test_the_account_moved_by_exactly_the_net(self) -> None:
        self.assertAlmostEqual(self.result.final_equity, 10_083.00, places=6)


class ChargedEveryNightTests(unittest.TestCase):
    """Financing is charged per night held, not per bar boundary observed."""

    def _hold(self, entry: datetime, exit_: datetime, direction=Direction.LONG):
        # Four bars: decide, enter, hold across the gap, close.
        times = [
            entry - timedelta(hours=1),
            entry,
            exit_,
            exit_ + timedelta(hours=1),
        ]
        rows = [(1.10000, 1.10000, 1.10000, 1.10000)] * 4
        # Formed on bar 0, filled at bar 1; the exit is formed on bar 1 and
        # filled at bar 2, so the position is open exactly from `entry` to
        # `exit_` and financing is charged for the nights strictly between.
        script = {
            0: OrderIntent(direction=direction, lots=1.0),
            1: OrderIntent(direction=Direction.FLAT),
        }
        result = Backtester(
            instrument=Instrument(commission_per_lot_per_side=0.0),
            costs=COSTS,
            risk=RiskConfig(max_daily_loss=None),
        ).run(Scripted(script), bars_at(times, rows))
        self.assertEqual(len(result.trades), 1)
        return result.trades[0]

    def test_a_friday_to_monday_hold_is_charged_for_every_intervening_night(self) -> None:
        friday = datetime(2021, 3, 5, 20, 0, tzinfo=timezone.utc)
        monday = datetime(2021, 3, 8, 20, 0, tzinfo=timezone.utc)
        trade = self._hold(friday, monday)
        # Friday, Saturday and Sunday nights, none of them the triple weekday.
        self.assertAlmostEqual(trade.swap, 3.0 * -0.80, places=9)

    def test_the_triple_swap_weekday_is_charged_three_times_by_the_engine(self) -> None:
        tuesday = datetime(2021, 3, 2, 20, 0, tzinfo=timezone.utc)
        wednesday = datetime(2021, 3, 3, 22, 0, tzinfo=timezone.utc)
        trade = self._hold(tuesday, wednesday)
        # Tuesday night at 1x plus Wednesday night at 3x.
        self.assertAlmostEqual(trade.swap, (-0.80) + 3.0 * (-0.80), places=9)

    def test_a_short_is_credited_rather_than_charged(self) -> None:
        tuesday = datetime(2021, 3, 2, 20, 0, tzinfo=timezone.utc)
        trade = self._hold(tuesday, tuesday + timedelta(hours=2), direction=Direction.SHORT)
        self.assertAlmostEqual(trade.swap, 0.20, places=9)


class NothingCostsNothingTests(unittest.TestCase):
    def test_always_flat_incurs_exactly_zero_cost(self) -> None:
        bars = synthetic_trending(500, seed=12)
        for label, costs in (
            ("default", CostConfig()),
            ("stressed", CostConfig().stress(5.0)),
        ):
            with self.subTest(costs=label):
                result = Backtester(costs=costs).run(AlwaysFlat(), bars)
                self.assertEqual(result.total_costs, 0.0)
                self.assertEqual(result.net_pnl, 0.0)
                self.assertEqual(result.final_equity, result.initial_equity)


class SpreadSensitivityTests(unittest.TestCase):
    """Doubling the spread must strictly reduce net P&L for anything that trades."""

    def _script(self, count: int) -> dict:
        """Alternate in and out so the trade set is identical at any spread."""
        script: dict = {}
        for index in range(0, count - 3, 6):
            script[index] = OrderIntent(direction=Direction.LONG, lots=0.5)
            script[index + 3] = OrderIntent(direction=Direction.FLAT)
        return script

    def test_wider_spreads_cost_more_on_an_identical_trade_set(self) -> None:
        bars = synthetic_trending(400, seed=21)
        script = self._script(len(bars))
        previous_net = None
        previous_spread = None
        for spread in (0.0, 1.0, 2.0, 4.0):
            costs = CostConfig(spread_points=spread, slippage_points=0.0)
            result = Backtester(costs=costs, risk=RiskConfig(max_daily_loss=None)).run(
                Scripted(script), bars
            )
            self.assertGreater(len(result.trades), 5)
            spread_total = sum(t.spread_cost for t in result.trades)
            if previous_net is not None:
                self.assertLess(result.net_pnl, previous_net)
                self.assertGreater(spread_total, previous_spread)
            previous_net, previous_spread = result.net_pnl, spread_total

    def test_doubling_the_spread_costs_exactly_twice_as_much_spread(self) -> None:
        bars = synthetic_trending(400, seed=21)
        script = self._script(len(bars))
        single = Backtester(
            costs=CostConfig(spread_points=1.0, slippage_points=0.0),
            risk=RiskConfig(max_daily_loss=None),
        ).run(Scripted(script), bars)
        double = Backtester(
            costs=CostConfig(spread_points=2.0, slippage_points=0.0),
            risk=RiskConfig(max_daily_loss=None),
        ).run(Scripted(script), bars)

        self.assertEqual(len(single.trades), len(double.trades))
        self.assertAlmostEqual(
            sum(t.spread_cost for t in double.trades),
            2.0 * sum(t.spread_cost for t in single.trades),
            places=6,
        )

    def test_stressing_costs_hurts_a_strategy_that_trades(self) -> None:
        from tradelab.strategies import DonchianBreakout

        bars = synthetic_trending(1500, seed=13)
        cheap = Backtester(costs=CostConfig()).run(DonchianBreakout(), bars)
        dear = Backtester(costs=CostConfig().stress(2.0)).run(DonchianBreakout(), bars)
        self.assertGreater(len(cheap.trades), 10)
        self.assertGreater(dear.total_costs / max(len(dear.trades), 1),
                           cheap.total_costs / len(cheap.trades))


if __name__ == "__main__":
    unittest.main()
