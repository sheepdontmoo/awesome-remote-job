"""The validation module is the part of the package that is supposed to say
no, so its tests are written to catch it saying yes.

Three properties carry most of the weight:

* A permutation test run on a **random walk** must usually fail to reject. The
  data is generated with independent returns and therefore contains nothing;
  if the test called that significant most of the time it would be worse than
  useless, because it would be certifying noise. It is checked across several
  independent series rather than one, since any single p-value below 0.05 is
  expected one time in twenty by definition.
* The bootstrap drawdown distribution must bracket the realised drawdown, with
  the p95 at or above it in the typical case. A Monte Carlo that reported a
  tail thinner than the single observed path would be reassuring and wrong.
* Walk-forward segments must be disjoint and in chronological order, and each
  fold's training window must end where its test window begins. That is the
  entire claim the method makes; if the segments overlap, the out-of-sample
  number is in-sample and the whole exercise is decoration.

These tests use short series and small permutation counts on purpose. They are
checking that the machinery is honest, not measuring anybody's edge.
"""

from __future__ import annotations

import math
import unittest
from datetime import datetime, timedelta, timezone

from tradelab.data import synthetic_random_walk, synthetic_trending
from tradelab.engine import Backtester
from tradelab.strategies import DonchianBreakout
from tradelab.types import BacktestResult, CostConfig, Direction, EquityPoint, ExitReason, Trade
from tradelab.validation import (
    MIN_TRADES,
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

START = datetime(2021, 1, 4, tzinfo=timezone.utc)


def factory(params: dict) -> DonchianBreakout:
    return DonchianBreakout(**params)


def backtester_factory() -> Backtester:
    return Backtester()


def make_trade(net: float, index: int) -> Trade:
    entry = START + timedelta(hours=index)
    return Trade(
        direction=Direction.LONG,
        lots=0.1,
        entry_ts=entry,
        entry_price=1.10000,
        exit_ts=entry + timedelta(hours=1),
        exit_price=1.10000 + net / 10_000.0,
        exit_reason=ExitReason.SIGNAL,
        initial_stop=1.09000,
        gross_pnl=net,
        bars_held=1,
    )


class GridTests(unittest.TestCase):
    def test_grid_is_the_cartesian_product(self) -> None:
        self.assertEqual(
            grid(fast=[5, 10], slow=[50]),
            [{"fast": 5, "slow": 50}, {"fast": 10, "slow": 50}],
        )

    def test_grid_length_is_the_number_of_trials(self) -> None:
        self.assertEqual(len(grid(a=[1, 2, 3], b=[1, 2], c=[1])), 6)


class ObjectiveTests(unittest.TestCase):
    def test_a_thin_sample_scores_negative_infinity(self) -> None:
        result = BacktestResult(trades=[make_trade(10.0, i) for i in range(3)])
        self.assertEqual(sharpe_objective(result), float("-inf"))

    def test_an_undefined_sharpe_is_unscoreable_rather_than_a_crash(self) -> None:
        # Enough trades to pass the count filter, but no curve to measure.
        result = BacktestResult(
            trades=[make_trade(10.0, i) for i in range(MIN_TRADES)],
            equity_curve=[EquityPoint(ts=START, equity=10_000.0, balance=10_000.0,
                                      exposure_lots=0.0, drawdown=0.0)],
        )
        self.assertEqual(sharpe_objective(result), float("-inf"))

    def test_a_real_run_scores_a_finite_number(self) -> None:
        bars = synthetic_trending(1500, seed=4)
        result = Backtester().run(DonchianBreakout(), bars)
        self.assertGreaterEqual(len(result.trades), MIN_TRADES)
        self.assertTrue(math.isfinite(sharpe_objective(result)))


class SurrogateTests(unittest.TestCase):
    def setUp(self) -> None:
        import random

        self.bars = synthetic_random_walk(300, seed=5)
        self.fake = surrogate_series(self.bars, random.Random(3))

    def test_the_surrogate_keeps_the_shape_of_the_series(self) -> None:
        self.assertEqual(len(self.fake), len(self.bars))
        self.assertEqual([b.ts for b in self.fake], [b.ts for b in self.bars])
        self.assertEqual(self.fake[0], self.bars[0])

    def test_the_surrogate_keeps_the_multiset_of_returns(self) -> None:
        def log_returns(bars):
            return sorted(
                round(math.log(bars[i].close / bars[i - 1].close), 12)
                for i in range(1, len(bars))
            )

        self.assertEqual(log_returns(self.fake), log_returns(self.bars))

    def test_the_surrogate_destroys_the_ordering(self) -> None:
        self.assertNotEqual(
            [round(b.close, 9) for b in self.fake],
            [round(b.close, 9) for b in self.bars],
        )

    def test_the_surrogate_bars_are_internally_consistent(self) -> None:
        for bar in self.fake:
            self.assertLessEqual(bar.low, min(bar.open, bar.close))
            self.assertGreaterEqual(bar.high, max(bar.open, bar.close))


class PermutationTests(unittest.TestCase):
    def test_a_random_walk_is_usually_not_significant(self) -> None:
        """Five independent driftless series, five permutation tests.

        Each series is generated with independent returns, so there is nothing
        in any of them to find. A p-value below 0.05 is expected occasionally -
        that is what a 5% threshold means - but a test that fired on most of
        these would be certifying noise, which is the failure that matters.
        """
        p_values = []
        for offset in range(5):
            bars = synthetic_random_walk(900, seed=200 + offset)
            outcome = permutation_test(
                factory, {}, bars, backtester_factory, n_perms=39, seed=11
            )
            self.assertGreaterEqual(outcome.p_value, 1.0 / 40.0)
            self.assertLessEqual(outcome.p_value, 1.0)
            p_values.append(outcome.p_value)

        significant = sum(1 for p in p_values if p < 0.05)
        self.assertLessEqual(
            significant, 1, f"noise was called significant {significant} times of 5: {p_values}"
        )

    def test_the_p_value_can_never_be_zero(self) -> None:
        bars = synthetic_trending(600, seed=9)
        outcome = permutation_test(factory, {}, bars, backtester_factory, n_perms=9, seed=2)
        self.assertGreaterEqual(outcome.p_value, 1.0 / 10.0)
        self.assertIn("p =", outcome.verdict)

    def test_a_permutation_count_below_one_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            permutation_test(factory, {}, synthetic_random_walk(100), backtester_factory, n_perms=0)


class MonteCarloTests(unittest.TestCase):
    def setUp(self) -> None:
        bars = synthetic_trending(2000, seed=3)
        self.result = Backtester().run(DonchianBreakout(), bars)
        self.assertGreater(len(self.result.trades), MIN_TRADES)

    def test_the_p95_drawdown_is_at_least_the_observed_one(self) -> None:
        outcome = monte_carlo_drawdown(self.result, n_runs=2000, seed=5)
        self.assertGreaterEqual(outcome.p95_max_drawdown, outcome.observed_max_drawdown)
        self.assertGreaterEqual(outcome.p99_max_drawdown, outcome.p95_max_drawdown)
        self.assertGreaterEqual(outcome.p95_max_drawdown, outcome.median_max_drawdown)

    def test_the_percentiles_are_ordered_and_bounded(self) -> None:
        outcome = monte_carlo_drawdown(self.result, n_runs=1000, seed=5)
        self.assertLessEqual(outcome.p5_max_drawdown, outcome.median_max_drawdown)
        self.assertGreaterEqual(outcome.observed_drawdown_percentile, 0.0)
        self.assertLessEqual(outcome.observed_drawdown_percentile, 1.0)
        self.assertGreaterEqual(outcome.prob_ruin, 0.0)
        self.assertLessEqual(outcome.prob_ruin, 1.0)

    def test_resampling_is_reproducible_from_its_seed(self) -> None:
        first = monte_carlo_drawdown(self.result, n_runs=200, seed=17)
        second = monte_carlo_drawdown(self.result, n_runs=200, seed=17)
        self.assertEqual(first.p95_max_drawdown, second.p95_max_drawdown)
        self.assertEqual(first.median_final_equity, second.median_final_equity)

    def test_too_few_trades_to_resample_says_so_instead_of_pretending(self) -> None:
        thin = BacktestResult(trades=[make_trade(10.0, 0)])
        outcome = monte_carlo_drawdown(thin, n_runs=100)
        self.assertEqual(outcome.n_runs, 0)
        self.assertIn("anecdote", outcome.verdict)

    def test_the_bootstrap_reproduces_a_known_drawdown(self) -> None:
        # Every trade identical, so every reshuffle is the same path: down
        # 100 four times from 10,000 is a 4% fall with no recovery.
        losses = BacktestResult(trades=[make_trade(-100.0, i) for i in range(4)])
        outcome = monte_carlo_drawdown(losses, n_runs=50, seed=1)
        self.assertAlmostEqual(outcome.observed_max_drawdown, 400.0 / 10_000.0, places=12)
        self.assertAlmostEqual(outcome.median_max_drawdown, 400.0 / 10_000.0, places=12)


class WalkForwardSplitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.bars = synthetic_random_walk(1000, seed=6)
        self.splits = split_walk_forward(self.bars, n_folds=5, train_frac=0.6)

    def test_train_and_test_never_overlap(self) -> None:
        for index, (train, test) in enumerate(self.splits):
            with self.subTest(fold=index):
                self.assertTrue(train and test)
                self.assertLess(train[-1].ts, test[0].ts)
                train_stamps = {b.ts for b in train}
                self.assertTrue(train_stamps.isdisjoint({b.ts for b in test}))

    def test_test_segments_are_disjoint_and_chronological(self) -> None:
        previous_end = None
        seen: set = set()
        for _, test in self.splits:
            stamps = {b.ts for b in test}
            self.assertTrue(seen.isdisjoint(stamps))
            seen |= stamps
            if previous_end is not None:
                self.assertGreater(test[0].ts, previous_end)
            previous_end = test[-1].ts

    def test_the_training_window_is_anchored_and_grows(self) -> None:
        lengths = [len(train) for train, _ in self.splits]
        self.assertEqual(lengths, sorted(lengths))
        self.assertTrue(all(self.splits[k][0][0].ts == self.bars[0].ts for k in range(5)))

    def test_the_test_segments_cover_the_tail_exactly_once(self) -> None:
        covered = sum(len(test) for _, test in self.splits)
        self.assertEqual(covered, len(self.bars) - int(len(self.bars) * 0.6))

    def test_impossible_splits_are_refused(self) -> None:
        with self.assertRaises(ValueError):
            split_walk_forward(self.bars, n_folds=0)
        with self.assertRaises(ValueError):
            split_walk_forward(self.bars, train_frac=1.0)
        with self.assertRaises(ValueError):
            split_walk_forward(self.bars[:10], n_folds=50)


class WalkForwardRunTests(unittest.TestCase):
    def setUp(self) -> None:
        bars = synthetic_trending(2500, seed=8)
        self.outcome = walk_forward(
            factory,
            grid(entry_lookback=[10, 20], exit_lookback=[5, 10]),
            bars,
            backtester_factory,
            n_folds=3,
        )

    def test_every_fold_reports_a_disjoint_ordered_test_window(self) -> None:
        self.assertEqual(self.outcome.n_folds, 3)
        previous_end = None
        for fold in self.outcome.folds:
            self.assertLess(fold.test_start, fold.test_end)
            if previous_end is not None:
                self.assertGreater(fold.test_start, previous_end)
            previous_end = fold.test_end

    def test_each_fold_trained_on_strictly_earlier_data(self) -> None:
        for fold in self.outcome.folds:
            with self.subTest(fold=fold.index):
                self.assertGreater(fold.train_bars, 0)
                self.assertGreater(fold.test_bars, 0)
                for trade in fold.train_result.trades:
                    self.assertLess(trade.entry_ts, fold.test_start)

    def test_the_chosen_parameters_come_from_the_grid(self) -> None:
        allowed = grid(entry_lookback=[10, 20], exit_lookback=[5, 10])
        for params in self.outcome.best_params_per_fold:
            self.assertIn(params, allowed)

    def test_the_verdict_is_a_sentence_not_a_score(self) -> None:
        self.assertIsInstance(self.outcome.verdict, str)
        self.assertGreater(len(self.outcome.verdict), 40)

    def test_an_empty_grid_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            walk_forward(factory, [], synthetic_random_walk(500), backtester_factory)


class SensitivityTests(unittest.TestCase):
    def test_the_surface_is_described_not_just_its_peak(self) -> None:
        bars = synthetic_trending(1500, seed=15)
        cells = grid(entry_lookback=[10, 20, 40], exit_lookback=[5, 10])
        outcome = parameter_sensitivity(factory, cells, bars, backtester_factory)
        self.assertEqual(outcome.n_cells, len(cells))
        self.assertGreaterEqual(outcome.best_objective, outcome.median_objective)
        self.assertGreaterEqual(outcome.profitable_fraction, 0.0)
        self.assertLessEqual(outcome.profitable_fraction, 1.0)
        self.assertIn(outcome.best.params, cells)

    def test_an_empty_grid_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            parameter_sensitivity(factory, [], synthetic_random_walk(200), backtester_factory)


class DeflatedSharpeTests(unittest.TestCase):
    def test_more_trials_means_a_deeper_haircut(self) -> None:
        one = deflated_sharpe(1.5, n_trials=1, n_obs=5)
        few = deflated_sharpe(1.5, n_trials=20, n_obs=5)
        many = deflated_sharpe(1.5, n_trials=200, n_obs=5)
        self.assertEqual(one, 1.5)  # a single trial is not a search
        self.assertLess(few, one)
        self.assertLess(many, few)

    def test_a_longer_sample_is_haircut_less(self) -> None:
        short = deflated_sharpe(1.5, n_trials=50, n_obs=3)
        long = deflated_sharpe(1.5, n_trials=50, n_obs=30)
        self.assertLess(short, long)

    def test_nonsense_arguments_are_refused(self) -> None:
        with self.assertRaises(ValueError):
            deflated_sharpe(1.0, n_trials=0, n_obs=10)
        with self.assertRaises(ValueError):
            deflated_sharpe(1.0, n_trials=10, n_obs=1)


class NullBenchmarkTests(unittest.TestCase):
    def test_the_baseline_pays_the_same_costs_and_is_reproducible(self) -> None:
        bars = synthetic_random_walk(1200, seed=19)
        first = null_benchmark(bars, backtester_factory, n_runs=5, seed=3)
        second = null_benchmark(bars, backtester_factory, n_runs=5, seed=3)
        self.assertEqual(first["median_net_pnl"], second["median_net_pnl"])
        self.assertEqual(first["n_runs"], 5)
        self.assertIn("verdict", first)

    def test_a_zero_run_baseline_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            null_benchmark(synthetic_random_walk(200), backtester_factory, n_runs=0)

    def test_the_baseline_costs_money_when_costs_are_doubled(self) -> None:
        bars = synthetic_random_walk(1500, seed=23)
        cheap = null_benchmark(bars, lambda: Backtester(costs=CostConfig()), n_runs=4, seed=1)
        dear = null_benchmark(
            bars, lambda: Backtester(costs=CostConfig().stress(3.0)), n_runs=4, seed=1
        )
        self.assertLess(dear["median_net_pnl"], cheap["median_net_pnl"])


if __name__ == "__main__":
    unittest.main()
