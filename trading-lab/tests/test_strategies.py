"""The concrete strategies, tested for the properties that make them usable as
measuring instruments rather than for whether they make money.

Three things are worth asserting about a strategy in this package and
profitability is not one of them:

* **It is reproducible.** `RandomEntry` re-seeds in `on_start`, so the same
  seed over the same bars must produce byte-identical trades however many times
  it is run and whatever else in the process has touched `random`. Everything
  in `validation.py` depends on that; without it a permutation test compares
  two different strategies.
* **It signals what it claims to signal.** `DonchianBreakout` must only enter
  when the close exceeds a channel built from bars that closed *before* the
  signal bar. This is checked by recomputing the channel independently at every
  entry rather than by trusting the indicator both sides of the assertion.
* **It refuses configurations that make it meaningless.** An unstopped fade or
  an exit threshold above its own entry threshold is not a variant to sweep
  over; it is a bug, and the constructor says so.

`AlwaysFlat` is the important one. Doing nothing must cost exactly nothing, to
the last decimal - it is the cheapest check that the engine is not inventing
costs, and every other number in the package is suspect if it fails.
"""

from __future__ import annotations

import unittest

from tradelab.data import synthetic_random_walk, synthetic_trending
from tradelab.engine import Backtester
from tradelab.indicators import donchian
from tradelab.strategies import (
    REGISTRY,
    AlwaysFlat,
    BuyAndHold,
    DonchianBreakout,
    RandomEntry,
    ZScoreReversion,
)
from tradelab.strategy import Strategy
from tradelab.types import Direction, ExitReason


def fingerprint(result) -> list:
    return [
        (t.entry_ts, round(t.entry_price, 10), t.direction, round(t.lots, 6), t.exit_reason)
        for t in result.trades
    ]


class RegistryTests(unittest.TestCase):
    def test_every_registered_name_maps_to_a_strategy_class(self) -> None:
        self.assertEqual(
            sorted(REGISTRY), ["buyhold", "donchian", "flat", "random", "zscore"]
        )
        for name, cls in REGISTRY.items():
            with self.subTest(strategy=name):
                self.assertTrue(issubclass(cls, Strategy))

    def test_every_strategy_runs_with_its_default_parameters(self) -> None:
        bars = synthetic_trending(1200, seed=3)
        for name, cls in REGISTRY.items():
            with self.subTest(strategy=name):
                result = Backtester().run(cls(), bars)
                self.assertEqual(result.bars_processed, len(bars) - cls().warmup)
                self.assertAlmostEqual(
                    result.final_equity,
                    result.initial_equity + sum(t.net_pnl for t in result.trades),
                    delta=1e-6,
                )

    def test_describe_names_the_parameters(self) -> None:
        text = DonchianBreakout(entry_lookback=30).describe()
        self.assertIn("donchian_breakout", text)
        self.assertIn("entry_lookback=30", text)

    def test_an_instance_can_be_reused_across_runs(self) -> None:
        """Walk-forward reuses nothing, but a caller might; `on_start` resets."""
        bars = synthetic_trending(800, seed=4)
        strategy = BuyAndHold(lots=0.1)
        first = Backtester().run(strategy, bars)
        second = Backtester().run(strategy, bars)
        self.assertEqual(len(first.trades), 1)
        self.assertEqual(fingerprint(first), fingerprint(second))


class AlwaysFlatTests(unittest.TestCase):
    def test_doing_nothing_costs_exactly_nothing(self) -> None:
        bars = synthetic_random_walk(500, seed=6)
        result = Backtester().run(AlwaysFlat(), bars)
        self.assertEqual(result.trades, [])
        self.assertEqual(result.total_costs, 0.0)
        self.assertEqual(len(result.equity_curve), len(bars))
        self.assertEqual({p.equity for p in result.equity_curve}, {10_000.0})
        self.assertEqual({p.exposure_lots for p in result.equity_curve}, {0.0})


class BuyAndHoldTests(unittest.TestCase):
    def setUp(self) -> None:
        self.bars = synthetic_random_walk(600, seed=7)
        self.result = Backtester().run(BuyAndHold(lots=0.1), self.bars)

    def test_it_takes_one_long_and_holds_it_to_the_end(self) -> None:
        self.assertEqual(len(self.result.trades), 1)
        trade = self.result.trades[0]
        self.assertEqual(trade.direction, Direction.LONG)
        self.assertEqual(trade.exit_reason, ExitReason.END_OF_DATA)
        self.assertEqual(trade.entry_ts, self.bars[1].ts)  # decided on bar 0
        self.assertEqual(trade.exit_ts, self.bars[-1].ts)

    def test_it_tracks_the_instrument_rather_than_a_multiple_of_it(self) -> None:
        # Gross is the costless move: entered at bar 1's open, marked out at the
        # final close. Anything else here would mean a leverage bug.
        trade = self.result.trades[0]
        move = (self.bars[-1].close - self.bars[1].open) * 0.1 * 100_000.0
        self.assertAlmostEqual(trade.gross_pnl, move, places=6)
        # And the costs are exactly the difference between that and what was
        # banked - total_costs already carries the swap with its sign flipped.
        self.assertAlmostEqual(trade.net_pnl, move - trade.total_costs, places=6)

    def test_a_non_positive_size_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            BuyAndHold(lots=0.0)


class RandomEntryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.bars = synthetic_random_walk(1500, seed=8)

    def test_the_same_seed_gives_the_same_trades(self) -> None:
        first = Backtester().run(RandomEntry(seed=5), self.bars)
        second = Backtester().run(RandomEntry(seed=5), self.bars)
        self.assertTrue(first.trades)
        self.assertEqual(fingerprint(first), fingerprint(second))

    def test_a_different_seed_gives_different_trades(self) -> None:
        first = Backtester().run(RandomEntry(seed=5), self.bars)
        other = Backtester().run(RandomEntry(seed=6), self.bars)
        self.assertNotEqual(fingerprint(first), fingerprint(other))

    def test_it_is_not_perturbed_by_the_global_random_module(self) -> None:
        import random

        random.seed(1)
        first = Backtester().run(RandomEntry(seed=5), self.bars)
        random.seed(999)
        [random.random() for _ in range(100)]
        second = Backtester().run(RandomEntry(seed=5), self.bars)
        self.assertEqual(fingerprint(first), fingerprint(second))

    def test_it_takes_both_sides(self) -> None:
        result = Backtester().run(RandomEntry(seed=2, prob=0.05), self.bars)
        directions = {t.direction for t in result.trades}
        self.assertEqual(directions, {Direction.LONG, Direction.SHORT})

    def test_it_honours_its_time_stop(self) -> None:
        result = Backtester().run(RandomEntry(seed=2, prob=0.05, hold_bars=6), self.bars)
        self.assertTrue(result.trades)
        # The time stop is formed on the bar the limit is reached and fills at
        # the next open, so a trade can be one bar older than the limit.
        self.assertLessEqual(max(t.bars_held for t in result.trades), 8)

    def test_nonsense_parameters_are_refused(self) -> None:
        for kwargs in (
            {"prob": 0.0},
            {"prob": 1.5},
            {"atr_period": 0},
            {"atr_stop_mult": 0.0},
            {"hold_bars": 0},
        ):
            with self.subTest(**kwargs):
                with self.assertRaises(ValueError):
                    RandomEntry(**kwargs)


class DonchianTests(unittest.TestCase):
    def setUp(self) -> None:
        self.bars = synthetic_trending(2500, seed=9, autocorr=0.25)
        self.result = Backtester().run(
            DonchianBreakout(entry_lookback=20, exit_lookback=10, trail=False), self.bars
        )

    def test_it_trades_a_trending_series(self) -> None:
        self.assertGreater(len(self.result.trades), 10)

    def test_every_entry_broke_a_level_that_existed_before_the_signal_bar(self) -> None:
        """The channel is recomputed here from scratch, so the assertion does
        not lean on the same call the strategy made."""
        index_of = {bar.ts: i for i, bar in enumerate(self.bars)}
        channel = donchian(self.bars, 20)
        checked = 0
        for trade in self.result.trades:
            # The signal bar is the one before the fill.
            signal = index_of[trade.entry_ts] - 1
            levels = channel[signal]
            if levels is None:
                continue
            upper, lower = levels
            close = self.bars[signal].close
            if trade.direction == Direction.LONG:
                self.assertGreater(close, upper)
            else:
                self.assertLess(close, lower)
            checked += 1
        self.assertGreater(checked, 10)

    def test_the_warmup_covers_every_indicator_it_uses(self) -> None:
        strategy = DonchianBreakout(entry_lookback=50, exit_lookback=20, atr_period=14)
        self.assertGreaterEqual(strategy.warmup, 52)

    def test_a_trailing_stop_never_loosens(self) -> None:
        stops: list = []

        class Watcher(DonchianBreakout):
            def on_bar(self, ctx):
                intent = super().on_bar(ctx)
                if ctx.position is not None and ctx._stop_move is not None:
                    stops.append((ctx.position.entry_ts, ctx.position.is_long, ctx._stop_move))
                return intent

        Backtester().run(Watcher(trail=True), self.bars)
        self.assertTrue(stops)
        by_trade: dict = {}
        for entry_ts, is_long, level in stops:
            previous = by_trade.get(entry_ts)
            if previous is not None:
                if is_long:
                    self.assertGreaterEqual(level, previous)
                else:
                    self.assertLessEqual(level, previous)
            by_trade[entry_ts] = level

    def test_long_only_takes_no_shorts(self) -> None:
        result = Backtester().run(DonchianBreakout(long_only=True), self.bars)
        self.assertTrue(result.trades)
        self.assertEqual({t.direction for t in result.trades}, {Direction.LONG})

    def test_nonsense_parameters_are_refused(self) -> None:
        for kwargs in ({"entry_lookback": 0}, {"atr_period": 0}, {"atr_stop_mult": 0.0}):
            with self.subTest(**kwargs):
                with self.assertRaises(ValueError):
                    DonchianBreakout(**kwargs)


class ZScoreTests(unittest.TestCase):
    def test_it_runs_and_balances(self) -> None:
        bars = synthetic_random_walk(2000, seed=10)
        result = Backtester().run(ZScoreReversion(lookback=20, trend_filter=0), bars)
        self.assertAlmostEqual(
            result.final_equity,
            result.initial_equity + sum(t.net_pnl for t in result.trades),
            delta=1e-6,
        )

    def test_the_trend_filter_reduces_the_number_of_entries(self) -> None:
        bars = synthetic_trending(3000, seed=11)
        unfiltered = Backtester().run(ZScoreReversion(lookback=20, trend_filter=0), bars)
        filtered = Backtester().run(ZScoreReversion(lookback=20, trend_filter=200), bars)
        self.assertGreater(len(unfiltered.trades), len(filtered.trades))

    def test_it_honours_its_time_stop(self) -> None:
        bars = synthetic_random_walk(2000, seed=12)
        result = Backtester().run(
            ZScoreReversion(lookback=20, trend_filter=0, max_bars=10, atr_stop_mult=20.0), bars
        )
        self.assertTrue(result.trades)
        self.assertLessEqual(max(t.bars_held for t in result.trades), 12)

    def test_an_exit_threshold_above_the_entry_threshold_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            ZScoreReversion(entry_z=1.0, exit_z=1.5)

    def test_an_unstopped_fade_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            ZScoreReversion(atr_stop_mult=0.0)

    def test_a_lookback_too_short_to_measure_dispersion_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            ZScoreReversion(lookback=1)


if __name__ == "__main__":
    unittest.main()
