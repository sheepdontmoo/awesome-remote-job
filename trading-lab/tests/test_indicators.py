"""Indicators are checked two ways: against arithmetic done by hand, and
against themselves on a truncated series.

The hand-computed values catch the smoothing constant being wrong - an ATR
using `2/(period+1)` instead of Wilder's `1/period` looks entirely plausible
and makes every ATR-scaled stop half as wide as intended. Small integer series
are used rather than realistic prices precisely so the expected value can be
written down in the test and read by somebody who does not trust the code.

The truncation test catches the more dangerous failure. An indicator that
computes a value at index `i` from data after `i` still produces a smooth,
believable line; the only cheap way to detect it is to recompute the series on
a prefix and check that the shared indices did not move. If the value at index
50 changes when bars 100..200 are deleted, those bars were being read.

`donchian` gets its own class because its contract is the unusual one: the
window ends at `i - 1`, so the current bar is excluded and a breakout is
measured against a level that existed before the bar that broke it.
"""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from tradelab.data import synthetic_random_walk
from tradelab.indicators import (
    atr,
    donchian,
    ema,
    last,
    pct_returns,
    realized_vol,
    rolling_max,
    rolling_min,
    rsi,
    sma,
    stdev,
    true_range,
    zscore,
)
from tradelab.types import Bar

START = datetime(2021, 6, 1, tzinfo=timezone.utc)


def bars_from(rows) -> list:
    return [
        Bar(ts=START + timedelta(hours=i), open=o, high=h, low=lo, close=c)
        for i, (o, h, lo, c) in enumerate(rows)
    ]


# Four bars whose true ranges are hand-checkable: 4, 3 and 3.
HAND_BARS = bars_from(
    [
        (10.0, 10.0, 10.0, 10.0),
        (10.0, 12.0, 8.0, 11.0),   # tr = max(4, |12-10|, |8-10|)  = 4
        (11.0, 13.0, 10.0, 12.0),  # tr = max(3, |13-11|, |10-11|) = 3
        (12.0, 14.0, 11.0, 13.0),  # tr = max(3, |14-12|, |11-12|) = 3
    ]
)


class MovingAverageTests(unittest.TestCase):
    def test_sma_is_none_until_the_window_is_full(self) -> None:
        self.assertEqual(sma([1.0, 2.0, 3.0, 4.0, 5.0], 3), [None, None, 2.0, 3.0, 4.0])

    def test_sma_of_a_short_series_is_all_none(self) -> None:
        self.assertEqual(sma([1.0, 2.0], 5), [None, None])

    def test_ema_is_seeded_with_the_first_sma(self) -> None:
        # period 3 -> alpha 0.5; seed = mean(1,2,3) = 2
        # index 3: 0.5*4 + 0.5*2 = 3 ; index 4: 0.5*5 + 0.5*3 = 4
        self.assertEqual(ema([1.0, 2.0, 3.0, 4.0, 5.0], 3), [None, None, 2.0, 3.0, 4.0])

    def test_ema_of_a_constant_series_is_that_constant(self) -> None:
        values = [7.0] * 20
        self.assertEqual([v for v in ema(values, 5) if v is not None], [7.0] * 16)

    def test_a_nonsense_period_is_rejected(self) -> None:
        for period in (0, -3):
            with self.subTest(period=period):
                with self.assertRaises(ValueError):
                    sma([1.0, 2.0, 3.0], period)
        with self.assertRaises(ValueError):
            stdev([1.0, 2.0, 3.0], 1)


class DispersionTests(unittest.TestCase):
    def test_stdev_matches_the_sample_formula(self) -> None:
        # values 2,4,4,4,5,5,7,9 over the last 4: 5,5,7,9 -> mean 6.5,
        # deviations -1.5,-1.5,0.5,2.5 -> m2 = 2.25+2.25+0.25+6.25 = 11
        # sample stdev = sqrt(11/3)
        values = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
        self.assertAlmostEqual(stdev(values, 4)[-1], (11.0 / 3.0) ** 0.5, places=12)

    def test_stdev_of_a_flat_window_is_zero_not_a_crash(self) -> None:
        self.assertEqual(stdev([3.0] * 10, 4)[-1], 0.0)

    def test_zscore_of_a_flat_window_is_zero(self) -> None:
        self.assertEqual(zscore([3.0] * 10, 4)[-1], 0.0)

    def test_zscore_is_the_deviation_over_the_sample_stdev(self) -> None:
        values = [1.0, 2.0, 3.0, 10.0]
        mean = 4.0
        deviation = (((1 - 4) ** 2 + (2 - 4) ** 2 + (3 - 4) ** 2 + (10 - 4) ** 2) / 3.0) ** 0.5
        self.assertAlmostEqual(zscore(values, 4)[-1], (10.0 - mean) / deviation, places=12)


class RangeTests(unittest.TestCase):
    def test_true_range_is_undefined_on_the_first_bar(self) -> None:
        ranges = true_range(HAND_BARS)
        self.assertIsNone(ranges[0])
        self.assertEqual(ranges[1:], [4.0, 3.0, 3.0])

    def test_true_range_covers_a_gap(self) -> None:
        gapped = bars_from([(10.0, 10.0, 10.0, 10.0), (20.0, 21.0, 20.0, 20.5)])
        # bar range is 1.0 but the gap from the previous close is 11.0
        self.assertEqual(true_range(gapped)[1], 11.0)

    def test_atr_uses_wilder_smoothing(self) -> None:
        # period 2: seed = mean(4, 3) = 3.5 at index 2
        # index 3: (3.5 * (2 - 1) + 3) / 2 = 3.25
        self.assertEqual(atr(HAND_BARS, 2), [None, None, 3.5, 3.25])

    def test_atr_needs_period_plus_one_bars(self) -> None:
        self.assertEqual(atr(HAND_BARS, 4), [None] * 4)

    def test_realized_vol_lines_up_with_prices_not_returns(self) -> None:
        values = [1.0, 1.1, 1.2, 1.1, 1.3, 1.25]
        vol = realized_vol(values, 2)
        self.assertIsNone(vol[1])
        self.assertIsNotNone(vol[2])
        self.assertEqual(len(vol), len(values))


class OscillatorTests(unittest.TestCase):
    def test_rsi_is_hand_checkable(self) -> None:
        # period 2 over [10, 11, 10, 12]
        #   seed: gains {1}, losses {1} -> avg 0.5 / 0.5 -> RSI 50 at index 2
        #   index 3: change +2 -> avg_gain 0.5*0.5 + 2/2 = 1.25
        #                        avg_loss 0.5*0.5 + 0    = 0.25
        #            RS = 5 -> RSI = 100 - 100/6 = 83.333...
        values = [10.0, 11.0, 10.0, 12.0]
        out = rsi(values, 2)
        self.assertEqual(out[:2], [None, None])
        self.assertAlmostEqual(out[2], 50.0, places=12)
        self.assertAlmostEqual(out[3], 100.0 - 100.0 / 6.0, places=12)

    def test_rsi_edges_are_stated_rather_than_divided(self) -> None:
        self.assertEqual(rsi([1.0, 2.0, 3.0, 4.0], 2)[-1], 100.0)
        self.assertEqual(rsi([4.0, 3.0, 2.0, 1.0], 2)[-1], 0.0)
        self.assertEqual(rsi([5.0] * 6, 2)[-1], 50.0)

    def test_rsi_stays_inside_its_bounds_on_real_looking_data(self) -> None:
        closes = [b.close for b in synthetic_random_walk(400, seed=3)]
        for value in rsi(closes, 14):
            if value is not None:
                self.assertGreaterEqual(value, 0.0)
                self.assertLessEqual(value, 100.0)


class DonchianTests(unittest.TestCase):
    def test_the_channel_excludes_the_current_bar(self) -> None:
        # highs 10, 12, 13, 14 ; lows 10, 8, 10, 11
        # index 2 reads bars 0..1 -> (12, 8) ; index 3 reads bars 1..2 -> (13, 8)
        self.assertEqual(donchian(HAND_BARS, 2), [None, None, (12.0, 8.0), (13.0, 8.0)])

    def test_the_current_bars_own_extreme_never_defines_its_level(self) -> None:
        bars = synthetic_random_walk(300, seed=7)
        channel = donchian(bars, 20)
        for index, value in enumerate(channel):
            if value is None:
                continue
            upper, lower = value
            window = bars[index - 20 : index]
            self.assertAlmostEqual(upper, max(b.high for b in window), places=12)
            self.assertAlmostEqual(lower, min(b.low for b in window), places=12)

    def test_the_first_value_is_one_bar_later_than_an_inclusive_channel(self) -> None:
        bars = synthetic_random_walk(60, seed=1)
        channel = donchian(bars, 20)
        inclusive = rolling_max([b.high for b in bars], 20)
        self.assertIsNone(channel[19])
        self.assertIsNotNone(channel[20])
        self.assertIsNotNone(inclusive[19])

    def test_rolling_extremes_include_the_current_value(self) -> None:
        values = [3.0, 1.0, 4.0, 1.0, 5.0]
        self.assertEqual(rolling_max(values, 2), [None, 3.0, 4.0, 4.0, 5.0])
        self.assertEqual(rolling_min(values, 2), [None, 1.0, 1.0, 1.0, 1.0])


class CausalityTests(unittest.TestCase):
    """Recomputing on a prefix must not move any shared index."""

    def setUp(self) -> None:
        self.bars = synthetic_random_walk(400, seed=17)
        self.closes = [b.close for b in self.bars]
        self.cut = 200

    def _assert_prefix_stable(self, full, partial) -> None:
        self.assertEqual(len(partial), self.cut)
        for index in range(self.cut):
            self.assertEqual(
                full[index], partial[index], f"value at index {index} changed when the future was removed"
            )

    def test_price_indicators_are_causal(self) -> None:
        cases = {
            "sma": lambda values: sma(values, 20),
            "ema": lambda values: ema(values, 20),
            "stdev": lambda values: stdev(values, 20),
            "zscore": lambda values: zscore(values, 20),
            "rsi": lambda values: rsi(values, 14),
            "rolling_max": lambda values: rolling_max(values, 20),
            "rolling_min": lambda values: rolling_min(values, 20),
            "realized_vol": lambda values: realized_vol(values, 20),
        }
        for name, function in cases.items():
            with self.subTest(indicator=name):
                self._assert_prefix_stable(
                    function(self.closes), function(self.closes[: self.cut])
                )

    def test_bar_indicators_are_causal(self) -> None:
        cases = {
            "true_range": true_range,
            "atr": lambda bars: atr(bars, 14),
            "donchian": lambda bars: donchian(bars, 20),
        }
        for name, function in cases.items():
            with self.subTest(indicator=name):
                self._assert_prefix_stable(function(self.bars), function(self.bars[: self.cut]))

    def test_every_indicator_returns_the_length_it_was_given(self) -> None:
        self.assertEqual(len(sma(self.closes, 20)), len(self.closes))
        self.assertEqual(len(atr(self.bars, 14)), len(self.bars))
        self.assertEqual(len(donchian(self.bars, 20)), len(self.bars))
        # The one deliberate exception, documented in the module.
        self.assertEqual(len(pct_returns(self.closes)), len(self.closes) - 1)


class AccessTests(unittest.TestCase):
    def test_pct_returns_align_one_step_behind_prices(self) -> None:
        self.assertEqual(pct_returns([1.0, 2.0, 4.0]), [1.0, 1.0])
        self.assertEqual(pct_returns([1.0]), [])
        self.assertEqual(pct_returns([0.0, 5.0]), [0.0])

    def test_last_scans_backwards_past_the_warmup_padding(self) -> None:
        self.assertEqual(last([None, 1.0, 2.0, None]), 2.0)
        self.assertIsNone(last([None, None]))
        self.assertEqual(last([], default=9.0), 9.0)


if __name__ == "__main__":
    unittest.main()
