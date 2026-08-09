"""The engine is the only module where being wrong costs money, so it gets the
most hostile tests in the suite.

Four properties are worth more than every performance number the package can
print, and each of them is checked here against hand-built bars rather than
against synthetic data, because a hand-built bar has a known right answer:

* **Causality.** A strategy cannot read past the bar it is on, and the decisions
  it makes on bars 0..i cannot depend on anything after bar i. The second is
  the stronger claim and the one a `BarWindow` alone does not prove - it is
  checked by running the same strategy over two series that agree up to bar i
  and diverge afterwards.
* **The accounting identity.** `final_equity == initial_equity + sum(net_pnl)`.
  If that ever fails, every metric downstream is describing a different account
  from the one the trades happened in.
* **Execution.** An intent formed on bar i fills at the open of bar i+1, and
  never at bar i's close. Gaps fill at the gap. A bar that could have hit both
  the stop and the target fills the stop.
* **Sizing.** A stopped-out trade loses the risk budget it was sized for, a
  stop too tight to size is refused loudly, and nothing exceeds `max_lots`.
"""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from typing import Optional

from tradelab.data import synthetic_random_walk, synthetic_trending
from tradelab.engine import Backtester
from tradelab.strategies import REGISTRY, AlwaysFlat, BuyAndHold, DonchianBreakout, RandomEntry
from tradelab.strategy import Context, LookAheadError, Strategy
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

FREE = CostConfig(
    spread_points=0.0,
    slippage_points=0.0,
    swap_long_points=0.0,
    swap_short_points=0.0,
)
NO_COMMISSION = Instrument(commission_per_lot_per_side=0.0)


def bars_from(rows, start: datetime = START, minutes: int = 60) -> list:
    """Build hourly bars from `(open, high, low, close)` tuples."""
    return [
        Bar(ts=start + timedelta(minutes=minutes * i), open=o, high=h, low=lo, close=c)
        for i, (o, h, lo, c) in enumerate(rows)
    ]


def flat_bar(price: float) -> tuple:
    return (price, price, price, price)


class Scripted(Strategy):
    """Emits a fixed intent on a fixed bar index. No indicators, no judgement.

    Every execution test needs to control exactly one thing - when the order is
    formed - and hold everything else still. A strategy that computes anything
    would make the test about the computation.
    """

    name = "scripted"

    def __init__(self, script: dict) -> None:
        super().__init__()
        self.script = dict(script)

    def on_bar(self, ctx: Context) -> Optional[OrderIntent]:
        return self.script.get(len(ctx.bars) - 1)


class Recorder(Strategy):
    """Records what it was shown and what it decided, bar by bar."""

    name = "recorder"

    def __init__(self, inner: Strategy) -> None:
        super().__init__()
        self.inner = inner
        self.seen: list = []

    @property
    def warmup(self) -> int:
        return self.inner.warmup

    def on_start(self, instrument) -> None:
        self.seen = []
        self.inner.on_start(instrument)

    def on_bar(self, ctx: Context) -> Optional[OrderIntent]:
        index = len(ctx.bars) - 1
        intent = self.inner.on_bar(ctx)
        self.seen.append((index, round(ctx.equity, 9), _describe(intent)))
        return intent


def _describe(intent: Optional[OrderIntent]) -> str:
    if intent is None:
        return "none"
    return (
        f"{int(intent.direction)}|{intent.stop_loss}|{intent.take_profit}|"
        f"{intent.lots}|{intent.tag}"
    )


class ReachesPastTheEnd(Strategy):
    name = "reaches_past_the_end"

    def on_bar(self, ctx: Context) -> Optional[OrderIntent]:
        return ctx.bars[len(ctx.bars) + 3] and None


class SlicesPastTheEnd(Strategy):
    name = "slices_past_the_end"

    def on_bar(self, ctx: Context) -> Optional[OrderIntent]:
        return ctx.bars[0 : len(ctx.bars) + 5] and None


class LookAheadTests(unittest.TestCase):
    def setUp(self) -> None:
        self.bars = synthetic_random_walk(60, seed=4)

    def test_indexing_past_the_current_bar_raises(self) -> None:
        with self.assertRaises(LookAheadError):
            Backtester().run(ReachesPastTheEnd(), self.bars)

    def test_slicing_past_the_current_bar_raises(self) -> None:
        with self.assertRaises(LookAheadError):
            Backtester().run(SlicesPastTheEnd(), self.bars)

    def test_window_length_is_the_number_of_closed_bars(self) -> None:
        lengths: list = []

        class Counter(Strategy):
            name = "counter"

            def on_bar(self, ctx: Context) -> Optional[OrderIntent]:
                lengths.append(len(ctx.bars))
                self_ts = ctx.bars.current.ts
                assert ctx.bars[-1].ts == self_ts
                return None

        Backtester().run(Counter(), self.bars)
        self.assertEqual(lengths, list(range(1, len(self.bars) + 1)))

    def test_decisions_do_not_change_when_the_future_changes(self) -> None:
        """Two series identical up to bar `split` must produce identical
        decisions up to bar `split`, whatever happens afterwards."""
        split = 300
        original = synthetic_random_walk(500, seed=11)
        diverged = list(original[: split + 1]) + list(
            synthetic_random_walk(500 - split - 1, seed=99, start=original[split].close)
        )
        self.assertEqual(len(diverged), len(original))
        self.assertNotEqual(
            [b.close for b in original[split + 1 :]],
            [b.close for b in diverged[split + 1 :]],
        )

        for strategy in (DonchianBreakout(entry_lookback=10, exit_lookback=5), RandomEntry(seed=3)):
            with self.subTest(strategy=strategy.name):
                first = Recorder(strategy)
                second = Recorder(strategy)
                Backtester().run(first, original)
                Backtester().run(second, diverged)
                before = [row for row in first.seen if row[0] <= split]
                after = [row for row in second.seen if row[0] <= split]
                self.assertTrue(before)
                self.assertEqual(before, after)


class AccountingIdentityTests(unittest.TestCase):
    """`final_equity == initial_equity + sum(net_pnl)`, or nothing else matters."""

    def test_identity_holds_across_strategies_and_cost_models(self) -> None:
        bars = synthetic_trending(1200, seed=6)
        cost_models = {
            "default": CostConfig(),
            "free": FREE,
            "stressed": CostConfig().stress(3.0),
            "bid_prices": CostConfig(price_is_mid=False),
        }
        for name, factory in REGISTRY.items():
            for label, costs in cost_models.items():
                with self.subTest(strategy=name, costs=label):
                    result = Backtester(costs=costs, initial_equity=25_000.0).run(factory(), bars)
                    self.assertAlmostEqual(
                        result.final_equity,
                        result.initial_equity + sum(t.net_pnl for t in result.trades),
                        delta=1e-6,
                    )

    def test_equity_equals_balance_whenever_the_account_is_flat(self) -> None:
        bars = synthetic_trending(600, seed=8)
        result = Backtester().run(DonchianBreakout(), bars)
        flat_points = [p for p in result.equity_curve if p.exposure_lots == 0.0]
        self.assertTrue(flat_points)
        for point in flat_points:
            self.assertAlmostEqual(point.equity, point.balance, delta=1e-9)

    def test_run_always_ends_flat(self) -> None:
        bars = synthetic_trending(400, seed=2)
        for name, factory in REGISTRY.items():
            with self.subTest(strategy=name):
                result = Backtester().run(factory(), bars)
                self.assertEqual(result.equity_curve[-1].exposure_lots, 0.0)

    def test_doing_nothing_costs_nothing(self) -> None:
        bars = synthetic_random_walk(300, seed=1)
        result = Backtester().run(AlwaysFlat(), bars)
        self.assertEqual(result.trades, [])
        self.assertEqual(result.final_equity, result.initial_equity)
        self.assertTrue(all(p.equity == 10_000.0 for p in result.equity_curve))


class ExecutionTests(unittest.TestCase):
    """Where an order fills, and which side of a fork the engine takes."""

    def test_intent_fills_at_the_next_open_not_this_close(self) -> None:
        bars = bars_from(
            [
                (1.10000, 1.10000, 1.10000, 1.10000),  # decision bar
                (1.12000, 1.12100, 1.11900, 1.12000),  # gapped open: the fill
                (1.12000, 1.12100, 1.11900, 1.12000),
            ]
        )
        script = {0: OrderIntent(direction=Direction.LONG, lots=1.0, tag="entry")}
        result = Backtester(costs=CostConfig()).run(Scripted(script), bars)

        self.assertEqual(len(result.trades), 1)
        trade = result.trades[0]
        half_and_slip = (0.5 + 2.0) * Instrument().point
        self.assertAlmostEqual(trade.entry_price, 1.12000 + half_and_slip, places=9)
        self.assertEqual(trade.entry_ts, bars[1].ts)
        # The signal bar's close was 1.10000; filling there would be a 200-point
        # gift, which is precisely the bug this asserts against.
        self.assertGreater(trade.entry_price, 1.11)

    def test_intent_on_the_final_bar_is_never_executed(self) -> None:
        bars = bars_from([flat_bar(1.1), flat_bar(1.1), flat_bar(1.1)])
        script = {2: OrderIntent(direction=Direction.LONG, lots=1.0)}
        result = Backtester().run(Scripted(script), bars)
        self.assertEqual(result.trades, [])

    def test_a_bar_gapping_through_the_stop_fills_at_the_open(self) -> None:
        stop = 1.09000
        bars = bars_from(
            [
                flat_bar(1.10000),
                (1.10000, 1.10050, 1.09950, 1.10000),   # entry at the open
                (1.08000, 1.08100, 1.07900, 1.08000),   # gaps straight through
                flat_bar(1.08000),
            ]
        )
        script = {0: OrderIntent(direction=Direction.LONG, lots=1.0, stop_loss=stop)}
        result = Backtester(costs=CostConfig(), instrument=NO_COMMISSION).run(
            Scripted(script), bars
        )

        self.assertEqual(len(result.trades), 1)
        trade = result.trades[0]
        self.assertEqual(trade.exit_reason, ExitReason.STOP_LOSS)
        expected = 1.08000 - (0.5 + 2.0) * Instrument().point
        self.assertAlmostEqual(trade.exit_price, expected, places=9)
        self.assertLess(trade.exit_price, stop, "a gapped stop must fill worse than the stop")

    def test_a_bar_spanning_both_stop_and_target_fills_the_stop(self) -> None:
        bars = bars_from(
            [
                flat_bar(1.10000),
                (1.10000, 1.10050, 1.09950, 1.10000),
                (1.10000, 1.10600, 1.09400, 1.10000),   # brackets both levels
                flat_bar(1.10000),
            ]
        )
        script = {
            0: OrderIntent(
                direction=Direction.LONG, lots=1.0, stop_loss=1.09500, take_profit=1.10500
            )
        }
        result = Backtester(costs=FREE, instrument=NO_COMMISSION).run(Scripted(script), bars)

        self.assertEqual(len(result.trades), 1)
        self.assertEqual(result.trades[0].exit_reason, ExitReason.STOP_LOSS)
        self.assertAlmostEqual(result.trades[0].exit_price, 1.09500, places=9)
        self.assertLess(result.trades[0].net_pnl, 0.0)

    def test_a_gapped_target_fills_at_the_limit_not_the_better_open(self) -> None:
        target = 1.10500
        bars = bars_from(
            [
                flat_bar(1.10000),
                (1.10000, 1.10050, 1.09950, 1.10000),
                (1.11000, 1.11100, 1.10900, 1.11000),   # gaps far past the target
                flat_bar(1.11000),
            ]
        )
        script = {
            0: OrderIntent(
                direction=Direction.LONG, lots=1.0, stop_loss=1.09000, take_profit=target
            )
        }
        result = Backtester(costs=CostConfig(), instrument=NO_COMMISSION).run(
            Scripted(script), bars
        )

        trade = result.trades[0]
        self.assertEqual(trade.exit_reason, ExitReason.TAKE_PROFIT)
        # Half spread, no slippage: a limit fills at its price or not at all.
        self.assertAlmostEqual(trade.exit_price, target - 0.5 * Instrument().point, places=9)
        self.assertLess(trade.exit_price, 1.10900)

    def test_open_position_is_closed_at_the_end_of_data(self) -> None:
        bars = bars_from([flat_bar(1.1), flat_bar(1.1), flat_bar(1.1), flat_bar(1.1)])
        script = {0: OrderIntent(direction=Direction.LONG, lots=1.0)}
        result = Backtester(costs=FREE, instrument=NO_COMMISSION).run(Scripted(script), bars)
        self.assertEqual(len(result.trades), 1)
        self.assertEqual(result.trades[0].exit_reason, ExitReason.END_OF_DATA)
        self.assertEqual(result.trades[0].exit_ts, bars[-1].ts)

    def test_short_entry_sells_the_bid_and_covers_at_the_ask(self) -> None:
        bars = bars_from([flat_bar(1.10000), flat_bar(1.10000), flat_bar(1.10000)])
        script = {
            0: OrderIntent(direction=Direction.SHORT, lots=1.0),
            1: OrderIntent(direction=Direction.FLAT),
        }
        result = Backtester(costs=CostConfig(), instrument=NO_COMMISSION).run(
            Scripted(script), bars
        )
        trade = result.trades[0]
        point = Instrument().point
        self.assertAlmostEqual(trade.entry_price, 1.10000 - 2.5 * point, places=9)
        self.assertAlmostEqual(trade.exit_price, 1.10000 + 2.5 * point, places=9)
        # Price never moved, so the costless result is exactly zero and the
        # whole loss is the round turn: 1.0 point of spread and 4.0 of slippage.
        self.assertAlmostEqual(trade.gross_pnl, 0.0, places=9)
        self.assertAlmostEqual(trade.spread_cost, 1.00, places=9)
        self.assertAlmostEqual(trade.slippage_cost, 4.00, places=9)
        self.assertAlmostEqual(trade.net_pnl, -5.00, places=9)


class SizingTests(unittest.TestCase):
    def _stop_out(self, risk: RiskConfig, stop: float, equity: float = 10_000.0):
        bars = bars_from(
            [
                flat_bar(1.10000),
                (1.10000, 1.10050, 1.09950, 1.10000),
                (1.09990, 1.10000, 1.08000, 1.08100),
                flat_bar(1.08100),
            ]
        )
        script = {0: OrderIntent(direction=Direction.LONG, stop_loss=stop)}
        return Backtester(
            instrument=NO_COMMISSION, costs=FREE, risk=risk, initial_equity=equity
        ).run(Scripted(script), bars)

    def test_a_stopped_out_trade_loses_the_risk_budget(self) -> None:
        risk = RiskConfig(risk_per_trade=0.005, max_daily_loss=None)
        result = self._stop_out(risk, stop=1.09000)
        self.assertEqual(len(result.trades), 1)
        trade = result.trades[0]
        self.assertEqual(trade.exit_reason, ExitReason.STOP_LOSS)
        budget = 10_000.0 * 0.005
        # Lot rounding is the only slack: 0.05 lots over 1000 points is exact.
        self.assertAlmostEqual(-trade.net_pnl, budget, delta=budget * 0.02)
        self.assertAlmostEqual(trade.r_multiple, -1.0, delta=0.02)

    def test_a_stop_inside_the_minimum_is_skipped_with_a_warning(self) -> None:
        risk = RiskConfig(min_stop_points=50.0)
        result = self._stop_out(risk, stop=1.09990)  # 10 points away
        self.assertEqual(result.trades, [])
        self.assertTrue(any("minimum" in w for w in result.warnings), result.warnings)

    def test_an_intent_without_stop_or_lots_is_skipped_with_a_warning(self) -> None:
        bars = bars_from([flat_bar(1.1), flat_bar(1.1), flat_bar(1.1)])
        script = {0: OrderIntent(direction=Direction.LONG)}
        result = Backtester().run(Scripted(script), bars)
        self.assertEqual(result.trades, [])
        self.assertTrue(any("cannot be sized" in w for w in result.warnings), result.warnings)

    def test_a_stop_on_the_wrong_side_of_the_fill_is_refused(self) -> None:
        risk = RiskConfig(min_stop_points=1.0)
        result = self._stop_out(risk, stop=1.10500)  # above a long entry
        self.assertEqual(result.trades, [])
        self.assertTrue(any("wrong side" in w for w in result.warnings), result.warnings)

    def test_size_never_exceeds_max_lots(self) -> None:
        risk = RiskConfig(
            risk_per_trade=0.90, max_lots=0.5, min_stop_points=10.0, max_daily_loss=None
        )
        result = self._stop_out(risk, stop=1.09900)  # 100 points: sizes to 90 lots
        self.assertEqual(len(result.trades), 1)
        self.assertEqual(result.trades[0].lots, 0.5)

    def test_contract_size_is_recorded_from_the_instrument(self) -> None:
        instrument = Instrument(contract_size=1_000.0, commission_per_lot_per_side=0.0)
        bars = bars_from([flat_bar(1.1), flat_bar(1.1), flat_bar(1.1)])
        script = {0: OrderIntent(direction=Direction.LONG, lots=1.0)}
        result = Backtester(instrument=instrument, costs=FREE).run(Scripted(script), bars)
        self.assertEqual(result.trades[0].contract_size, 1_000.0)


class GuardTests(unittest.TestCase):
    def test_no_bars_warns_and_returns_an_empty_result(self) -> None:
        result = Backtester().run(AlwaysFlat(), [])
        self.assertEqual(result.trades, [])
        self.assertEqual(result.equity_curve, [])
        self.assertEqual(result.final_equity, result.initial_equity)
        self.assertTrue(result.warnings)

    def test_warmup_longer_than_the_sample_warns_and_runs_nothing(self) -> None:
        bars = synthetic_random_walk(10, seed=0)
        result = Backtester().run(DonchianBreakout(entry_lookback=200), bars)
        self.assertEqual(result.bars_processed, 0)
        self.assertTrue(any("warmup" in w for w in result.warnings), result.warnings)

    def test_ruin_threshold_halts_the_run(self) -> None:
        # One enormous position, then a collapse far past the ruin floor.
        bars = bars_from(
            [
                flat_bar(1.10000),
                (1.10000, 1.10000, 1.10000, 1.10000),
                (0.50000, 0.50000, 0.50000, 0.50000),
                flat_bar(0.50000),
            ]
        )
        script = {0: OrderIntent(direction=Direction.LONG, lots=1.0)}
        result = Backtester(costs=FREE, instrument=NO_COMMISSION).run(Scripted(script), bars)
        self.assertTrue(result.halted_early)
        self.assertLess(result.bars_processed, len(bars))
        self.assertAlmostEqual(
            result.final_equity,
            result.initial_equity + sum(t.net_pnl for t in result.trades),
            delta=1e-6,
        )

    def test_buy_and_hold_takes_exactly_one_trade(self) -> None:
        bars = synthetic_random_walk(200, seed=5)
        result = Backtester().run(BuyAndHold(lots=0.1), bars)
        self.assertEqual(len(result.trades), 1)
        self.assertEqual(result.trades[0].direction, Direction.LONG)
        self.assertEqual(result.trades[0].exit_reason, ExitReason.END_OF_DATA)


if __name__ == "__main__":
    unittest.main()
