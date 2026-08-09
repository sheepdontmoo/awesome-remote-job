"""Loading is where a backtest goes wrong before a single rule has run, so the
loader tests are mostly about what it *refuses* to do.

A loader that silently repairs a file is worse than one that crashes, because
the repair is invisible and the result still looks like a result. The
assertions below therefore check that a high below a low, a duplicated
timestamp, a timestamp that goes backwards and an ambiguous date format all
raise `DataError` naming the row - and that nothing is dropped, filled or
guessed at along the way.

The synthetic generators get the opposite treatment: they are tested for the
properties the rest of the package relies on. Bars are internally consistent so
the engine's stop logic is meaningful. Timestamps skip weekends so the
annualisation constants mean what they say. The same seed produces the same
series, or every reproducibility claim in `validation.py` is false. And the
trending generator really is more autocorrelated than the random walk, because
otherwise the positive control is not a control.
"""

from __future__ import annotations

import math
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone

from tradelab.data import (
    DataError,
    load_csv,
    load_mt5_csv,
    resample,
    save_csv,
    summary,
    synthetic_random_walk,
    synthetic_trending,
)
from tradelab.types import Bar


class TempFileTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)

    def write(self, text: str, name: str = "prices.csv") -> str:
        path = os.path.join(self.directory.name, name)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(text)
        return path


class LoadCsvTests(TempFileTestCase):
    HEADED = (
        "time,open,high,low,close,volume\n"
        "2021-03-01 00:00:00,1.10000,1.10500,1.09800,1.10200,120\n"
        "2021-03-01 01:00:00,1.10200,1.10600,1.10100,1.10300,90\n"
    )

    def test_a_headed_file_loads(self) -> None:
        bars = load_csv(self.write(self.HEADED))
        self.assertEqual(len(bars), 2)
        self.assertEqual(bars[0].ts, datetime(2021, 3, 1, tzinfo=timezone.utc))
        self.assertEqual(bars[0].open, 1.10000)
        self.assertEqual(bars[0].volume, 120.0)

    def test_columns_may_appear_in_any_order(self) -> None:
        text = (
            "close,open,timestamp,low,high\n"
            "1.10200,1.10000,2021-03-01 00:00:00,1.09800,1.10500\n"
        )
        bar = load_csv(self.write(text))[0]
        self.assertEqual((bar.open, bar.high, bar.low, bar.close),
                         (1.10000, 1.10500, 1.09800, 1.10200))

    def test_separate_date_and_time_columns(self) -> None:
        text = (
            "<DATE>\t<TIME>\t<OPEN>\t<HIGH>\t<LOW>\t<CLOSE>\t<TICKVOL>\n"
            "2021.03.01\t00:00:00\t1.10000\t1.10500\t1.09800\t1.10200\t120\n"
        )
        bar = load_csv(self.write(text, "mt5.csv"))[0]
        self.assertEqual(bar.ts, datetime(2021, 3, 1, tzinfo=timezone.utc))
        self.assertEqual(bar.volume, 120.0)

    def test_a_headerless_file_is_read_positionally(self) -> None:
        text = (
            "2021-03-01 00:00:00,1.10000,1.10500,1.09800,1.10200\n"
            "2021-03-01 01:00:00,1.10200,1.10600,1.10100,1.10300\n"
        )
        self.assertEqual(len(load_csv(self.write(text))), 2)

    def test_blank_lines_and_comments_are_skipped(self) -> None:
        text = "# exported by hand\n\n" + self.HEADED
        self.assertEqual(len(load_csv(self.write(text))), 2)

    def test_iso_timestamps_with_an_offset_are_converted_to_utc(self) -> None:
        text = "time,open,high,low,close\n2021-03-01T02:00:00+02:00,1.1,1.1,1.1,1.1\n"
        self.assertEqual(load_csv(self.write(text))[0].ts,
                         datetime(2021, 3, 1, 0, 0, tzinfo=timezone.utc))

    def test_naive_timestamps_take_the_assumed_timezone(self) -> None:
        text = "time,open,high,low,close\n2021-03-01 02:00:00,1.1,1.1,1.1,1.1\n"
        broker = timezone(timedelta(hours=2))
        bars = load_csv(self.write(text), assume_tz=broker)
        self.assertEqual(bars[0].ts, datetime(2021, 3, 1, 0, 0, tzinfo=timezone.utc))

    def test_a_high_below_a_low_names_the_row(self) -> None:
        text = "time,open,high,low,close\n2021-03-01 00:00:00,1.1,1.0,1.2,1.1\n"
        with self.assertRaises(DataError) as caught:
            load_csv(self.write(text))
        self.assertIn("row 2", str(caught.exception))

    def test_a_close_outside_the_bar_range_is_refused(self) -> None:
        text = "time,open,high,low,close\n2021-03-01 00:00:00,1.1,1.2,1.0,1.5\n"
        with self.assertRaises(DataError):
            load_csv(self.write(text))

    def test_a_duplicated_timestamp_is_refused_rather_than_deduplicated(self) -> None:
        text = (
            "time,open,high,low,close\n"
            "2021-03-01 00:00:00,1.1,1.1,1.1,1.1\n"
            "2021-03-01 00:00:00,1.1,1.1,1.1,1.1\n"
        )
        with self.assertRaises(DataError) as caught:
            load_csv(self.write(text))
        self.assertIn("duplicates", str(caught.exception))

    def test_timestamps_going_backwards_are_refused(self) -> None:
        text = (
            "time,open,high,low,close\n"
            "2021-03-01 01:00:00,1.1,1.1,1.1,1.1\n"
            "2021-03-01 00:00:00,1.1,1.1,1.1,1.1\n"
        )
        with self.assertRaises(DataError) as caught:
            load_csv(self.write(text))
        self.assertIn("backwards", str(caught.exception))

    def test_an_ambiguous_date_is_refused_rather_than_guessed(self) -> None:
        text = "time,open,high,low,close\n03/04/2021 00:00,1.1,1.1,1.1,1.1\n"
        with self.assertRaises(DataError) as caught:
            load_csv(self.write(text))
        self.assertIn("timestamp_formats", str(caught.exception))

    def test_an_explicit_format_resolves_the_ambiguity(self) -> None:
        text = "time,open,high,low,close\n03/04/2021 00:00,1.1,1.1,1.1,1.1\n"
        bars = load_csv(self.write(text), timestamp_formats=["%d/%m/%Y %H:%M"])
        self.assertEqual(bars[0].ts, datetime(2021, 4, 3, tzinfo=timezone.utc))

    def test_a_non_finite_price_is_refused(self) -> None:
        text = "time,open,high,low,close\n2021-03-01 00:00:00,1.1,1.1,1.1,nan\n"
        with self.assertRaises(DataError):
            load_csv(self.write(text))

    def test_an_empty_file_is_refused(self) -> None:
        with self.assertRaises(DataError):
            load_csv(self.write("   \n"))

    def test_a_header_missing_columns_says_which_are_needed(self) -> None:
        text = "time,bid,ask\n2021-03-01 00:00:00,1.1,1.2\n"
        with self.assertRaises(DataError) as caught:
            load_csv(self.write(text))
        message = str(caught.exception)
        self.assertIn("looks like a header", message)
        self.assertIn("open, high, low, close", message)

    def test_a_missing_file_raises_rather_than_returning_nothing(self) -> None:
        with self.assertRaises(OSError):
            load_csv(os.path.join(self.directory.name, "absent.csv"))


class RoundTripTests(TempFileTestCase):
    def test_saving_and_reloading_changes_no_price(self) -> None:
        bars = synthetic_random_walk(200, seed=31)
        path = os.path.join(self.directory.name, "round_trip.csv")
        save_csv(bars, path)
        reloaded = load_csv(path)
        self.assertEqual(len(reloaded), len(bars))
        for original, copy in zip(bars, reloaded):
            self.assertEqual(original.ts, copy.ts)
            self.assertEqual(original.open, copy.open)
            self.assertEqual(original.high, copy.high)
            self.assertEqual(original.low, copy.low)
            self.assertEqual(original.close, copy.close)

    def test_the_mt5_loader_reads_an_mt5_export(self) -> None:
        text = (
            "<DATE>\t<TIME>\t<OPEN>\t<HIGH>\t<LOW>\t<CLOSE>\t<TICKVOL>\t<VOL>\t<SPREAD>\n"
            "2021.03.01\t00:00:00\t1.10000\t1.10500\t1.09800\t1.10200\t120\t0\t10\n"
            "2021.03.01\t01:00:00\t1.10200\t1.10600\t1.10100\t1.10300\t90\t0\t12\n"
        )
        bars = load_mt5_csv(self.write(text, "EURUSD_H1.csv"))
        self.assertEqual(len(bars), 2)
        # tickvol, not the always-zero real volume column
        self.assertEqual(bars[0].volume, 120.0)
        self.assertEqual(bars[1].ts, datetime(2021, 3, 1, 1, 0, tzinfo=timezone.utc))


class ResampleTests(unittest.TestCase):
    def quarter_hours(self, count: int) -> list:
        start = datetime(2021, 3, 3, 0, 0, tzinfo=timezone.utc)  # a Wednesday
        return [
            Bar(
                ts=start + timedelta(minutes=15 * i),
                open=1.1000 + i * 0.0001,
                high=1.1000 + i * 0.0001 + 0.0002,
                low=1.1000 + i * 0.0001 - 0.0002,
                close=1.1000 + i * 0.0001 + 0.0001,
                volume=10.0,
            )
            for i in range(count)
        ]

    def test_aggregation_takes_first_open_last_close_and_the_extremes(self) -> None:
        source = self.quarter_hours(8)
        hourly = resample(source, 60)
        self.assertEqual(len(hourly), 2)
        first = hourly[0]
        self.assertEqual(first.ts, source[0].ts)
        self.assertEqual(first.open, source[0].open)
        self.assertEqual(first.close, source[3].close)
        self.assertEqual(first.high, max(b.high for b in source[:4]))
        self.assertEqual(first.low, min(b.low for b in source[:4]))
        self.assertEqual(first.volume, 40.0)

    def test_an_incomplete_trailing_bucket_is_dropped(self) -> None:
        # Seven bars: the second hour is only three quarters full.
        self.assertEqual(len(resample(self.quarter_hours(7), 60)), 1)

    def test_upsampling_is_refused_because_it_would_invent_prices(self) -> None:
        with self.assertRaises(ValueError):
            resample(self.quarter_hours(8), 5)

    def test_a_nonsense_period_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            resample(self.quarter_hours(8), 0)

    def test_resampling_nothing_returns_nothing(self) -> None:
        self.assertEqual(resample([], 60), [])

    def test_buckets_are_aligned_to_the_epoch(self) -> None:
        start = datetime(2021, 3, 3, 0, 30, tzinfo=timezone.utc)
        source = [
            Bar(ts=start + timedelta(minutes=15 * i), open=1.1, high=1.1, low=1.1, close=1.1)
            for i in range(6)
        ]
        hourly = resample(source, 60)
        self.assertTrue(all(b.ts.minute == 0 for b in hourly))


class SyntheticTests(unittest.TestCase):
    def test_bars_are_internally_consistent(self) -> None:
        for bars in (synthetic_random_walk(500, seed=1), synthetic_trending(500, seed=1)):
            for bar in bars:
                self.assertLessEqual(bar.low, min(bar.open, bar.close))
                self.assertGreaterEqual(bar.high, max(bar.open, bar.close))
                self.assertGreater(bar.low, 0.0)

    def test_timestamps_are_utc_strictly_increasing_and_skip_weekends(self) -> None:
        bars = synthetic_random_walk(400, seed=2)
        for index, bar in enumerate(bars):
            self.assertEqual(bar.ts.tzinfo, timezone.utc)
            self.assertLess(bar.ts.weekday(), 5, f"bar {index} landed on a weekend")
            if index:
                self.assertGreater(bar.ts, bars[index - 1].ts)

    def test_the_same_seed_produces_the_same_series(self) -> None:
        first = synthetic_random_walk(200, seed=42)
        second = synthetic_random_walk(200, seed=42)
        third = synthetic_random_walk(200, seed=43)
        self.assertEqual([b.close for b in first], [b.close for b in second])
        self.assertNotEqual([b.close for b in first], [b.close for b in third])

    def test_the_random_walk_has_no_material_autocorrelation(self) -> None:
        self.assertLess(abs(_autocorrelation(synthetic_random_walk(6000, seed=5))), 0.05)

    def test_the_trending_series_has_the_autocorrelation_it_advertises(self) -> None:
        measured = _autocorrelation(synthetic_trending(6000, seed=5, autocorr=0.30))
        self.assertGreater(measured, 0.20)

    def test_the_first_bar_opens_at_the_requested_price(self) -> None:
        self.assertEqual(synthetic_random_walk(10, seed=0, start=1.2345)[0].open, 1.2345)

    def test_the_bar_interval_is_honoured(self) -> None:
        bars = synthetic_random_walk(5, minutes=15)
        self.assertEqual((bars[1].ts - bars[0].ts).total_seconds(), 900.0)

    def test_zero_bars_is_an_empty_series_not_an_error(self) -> None:
        self.assertEqual(synthetic_random_walk(0), [])

    def test_nonsense_parameters_are_refused(self) -> None:
        for kwargs in (
            {"n": -1},
            {"n": 10, "start": 0.0},
            {"n": 10, "annual_vol": -0.1},
            {"n": 10, "bars_per_year": 0.0},
            {"n": 10, "minutes": 0},
        ):
            with self.subTest(**kwargs):
                with self.assertRaises(ValueError):
                    synthetic_random_walk(**kwargs)

    def test_unknown_keywords_are_refused_rather_than_ignored(self) -> None:
        with self.assertRaises(TypeError):
            synthetic_trending(10, drift=0.5)


class SummaryTests(unittest.TestCase):
    def test_summary_reports_the_span_and_the_interval(self) -> None:
        text = summary(synthetic_random_walk(300, seed=3))
        self.assertIn("Bars", text)
        self.assertIn("1h", text)
        self.assertIn("Gaps", text)

    def test_summary_of_nothing_says_so(self) -> None:
        self.assertEqual(summary([]), "No bars.")


def _autocorrelation(bars) -> float:
    """Lag-1 autocorrelation of log returns. Written out rather than imported
    so the test does not measure the module against its own helper."""
    closes = [b.close for b in bars]
    returns = [math.log(closes[i] / closes[i - 1]) for i in range(1, len(closes))]
    mean = sum(returns) / len(returns)
    centred = [r - mean for r in returns]
    variance = sum(c * c for c in centred)
    if variance <= 0.0:
        return 0.0
    covariance = sum(centred[i] * centred[i - 1] for i in range(1, len(centred)))
    return covariance / variance


if __name__ == "__main__":
    unittest.main()
