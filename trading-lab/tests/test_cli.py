"""Smoke tests for the command line, including the one command whose whole
purpose is to work on a freshly cloned tree.

`python3 -m tradelab demo` is the package's first impression. It has to run
with no data files, no arguments, no install step and no network, and it has to
produce a report rather than a traceback. That is checked here twice: once
in-process through `cli.main`, and once as a real subprocess, because a working
`main` and a broken `__main__.py` look identical from inside the interpreter.

The rest of the file checks the plumbing that is easy to get quietly wrong: a
`--param` value arriving as the string "40" instead of the integer 40, a
`--stress` factor that never reaches the cost model, an unknown strategy
exiting 0 with an empty report. Every command is run against synthetic data and
tiny permutation counts - these tests are about wiring, not about statistics.
"""

from __future__ import annotations

import contextlib
import io
import os
import subprocess
import sys
import tempfile
import unittest

from tradelab import __version__
from tradelab.cli import build_parser, main
from tradelab.data import save_csv, synthetic_random_walk

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run(*argv) -> tuple:
    """Run one command in-process and return (exit code, captured output).

    Standard error is captured too, and discarded: several of these tests
    deliberately provoke usage errors, and their messages would otherwise be
    printed in the middle of the suite's own output and read as failures.
    """
    stream = io.StringIO()
    with contextlib.redirect_stderr(io.StringIO()):
        code = main(list(argv), stream=stream)
    return code, stream.getvalue()


@contextlib.contextmanager
def quiet_stderr():
    """Swallow argparse's usage message when a bad argument is expected."""
    with contextlib.redirect_stderr(io.StringIO()):
        yield


class DemoTests(unittest.TestCase):
    def test_the_demo_runs_clean_with_no_arguments(self) -> None:
        code, text = run("demo")
        self.assertEqual(code, 0)
        for expected in (
            "tradelab backtest report",
            "tradelab comparison",
            "donchian",
            "random",
            "buyhold",
            "CAVEATS",
        ):
            self.assertIn(expected, text)

    def test_the_demo_says_the_data_is_synthetic(self) -> None:
        _, text = run("demo", "--bars", "400")
        self.assertIn("synthetic", text)
        self.assertIn("no exploitable edge", text)

    def test_the_demo_runs_as_a_module(self) -> None:
        """`python3 -m tradelab demo` - the command in the README."""
        completed = subprocess.run(
            [sys.executable, "-m", "tradelab", "demo", "--bars", "400"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=300,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("tradelab backtest report", completed.stdout)
        self.assertEqual(completed.stderr, "")

    def test_the_demo_honours_its_seed(self) -> None:
        _, first = run("demo", "--bars", "300", "--seed", "1")
        _, again = run("demo", "--bars", "300", "--seed", "1")
        _, other = run("demo", "--bars", "300", "--seed", "2")
        self.assertEqual(first, again)
        self.assertNotEqual(first, other)


class RunTests(unittest.TestCase):
    def test_run_defaults_to_synthetic_data(self) -> None:
        code, text = run("run", "--bars", "600")
        self.assertEqual(code, 0)
        self.assertIn("donchian_breakout", text)
        self.assertIn("synthetic walk", text)

    def test_run_accepts_typed_parameters(self) -> None:
        code, text = run("run", "--bars", "600", "--param", "entry_lookback=40",
                         "--param", "trail=false")
        self.assertEqual(code, 0)
        self.assertIn("entry_lookback=40", text)
        self.assertIn("trail=False", text)

    def test_an_unknown_parameter_is_a_usage_error(self) -> None:
        code, _ = run("run", "--bars", "200", "--param", "nonsense=1")
        self.assertEqual(code, 2)

    def test_an_invalid_parameter_value_is_a_usage_error(self) -> None:
        code, _ = run("run", "--bars", "200", "--param", "atr_stop_mult=0")
        self.assertEqual(code, 2)

    def test_a_malformed_parameter_is_rejected_by_the_parser(self) -> None:
        with quiet_stderr(), self.assertRaises(SystemExit):
            build_parser().parse_args(["run", "--param", "no_equals_sign"])

    def test_an_unknown_strategy_is_rejected_by_the_parser(self) -> None:
        with quiet_stderr(), self.assertRaises(SystemExit):
            build_parser().parse_args(["run", "--strategy", "moon_phase"])

    def test_stress_reaches_the_cost_model(self) -> None:
        _, cheap = run("run", "--bars", "600", "--strategy", "flat")
        _, dear = run("run", "--bars", "600", "--strategy", "flat", "--stress", "2.0")
        self.assertIn("1.0 / 2.0", cheap)
        self.assertIn("2.0 / 4.0", dear)

    def test_baselines_add_a_comparison_table(self) -> None:
        code, text = run("run", "--bars", "800", "--baselines")
        self.assertEqual(code, 0)
        self.assertIn("CONTROLS", text)
        self.assertIn("tradelab comparison", text)

    def test_every_registered_strategy_is_runnable_from_the_command_line(self) -> None:
        from tradelab.strategies import REGISTRY

        for name in sorted(REGISTRY):
            with self.subTest(strategy=name):
                code, text = run("run", "--bars", "500", "--strategy", name)
                self.assertEqual(code, 0)
                self.assertIn("tradelab backtest report", text)


class CsvTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = os.path.join(self.directory.name, "EURUSD_H1.csv")
        save_csv(synthetic_random_walk(800, seed=4), self.path)

    def test_a_csv_is_loaded_and_reported_as_the_source(self) -> None:
        code, text = run("run", "--csv", self.path)
        self.assertEqual(code, 0)
        self.assertIn(self.path, text)
        self.assertNotIn("synthetic walk", text)

    def test_a_missing_file_is_a_usage_error_not_a_traceback(self) -> None:
        code, _ = run("run", "--csv", os.path.join(self.directory.name, "absent.csv"))
        self.assertEqual(code, 2)

    def test_a_corrupt_file_is_a_usage_error(self) -> None:
        bad = os.path.join(self.directory.name, "bad.csv")
        with open(bad, "w", encoding="utf-8") as handle:
            handle.write("time,open,high,low,close\n2021-01-01 00:00:00,1.1,1.0,1.2,1.1\n")
        code, _ = run("run", "--csv", bad)
        self.assertEqual(code, 2)


class ValidateTests(unittest.TestCase):
    def test_validate_runs_every_test_it_advertises(self) -> None:
        code, text = run(
            "validate", "--bars", "900", "--perms", "5", "--folds", "2",
            "--mc-runs", "100", "--null-runs", "2",
        )
        self.assertEqual(code, 0)
        self.assertIn("tradelab validation", text)
        for expected in (
            "SWEEP",
            "WALK FORWARD RESULT",
            "PERMUTATION RESULT",
            "MONTE CARLO RESULT",
            "DEFLATION",
            "NULL BENCHMARK",
            "WHAT THIS DOES NOT SHOW",
        ):
            self.assertIn(expected, text)

    def test_pinning_a_parameter_removes_it_from_the_sweep(self) -> None:
        code, text = run(
            "validate", "--bars", "800", "--perms", "3", "--folds", "2",
            "--mc-runs", "50", "--null-runs", "1",
            "--param", "entry_lookback=20", "--param", "exit_lookback=10",
        )
        self.assertEqual(code, 0)
        self.assertIn("Sweeping 1 configuration(s)", text)

    def test_a_sample_too_short_to_fold_still_reports_the_other_tests(self) -> None:
        code, text = run(
            "validate", "--bars", "120", "--perms", "2", "--folds", "50",
            "--mc-runs", "20", "--null-runs", "1",
        )
        self.assertEqual(code, 0)
        self.assertIn("tradelab validation", text)

    def test_nonsense_counts_are_a_usage_error(self) -> None:
        self.assertEqual(run("validate", "--bars", "300", "--perms", "0")[0], 2)
        self.assertEqual(run("validate", "--bars", "300", "--folds", "0")[0], 2)


class ParserTests(unittest.TestCase):
    def test_no_command_prints_help_rather_than_failing(self) -> None:
        code, text = run()
        self.assertEqual(code, 0)
        self.assertIn("demo", text)
        self.assertIn("validate", text)

    def test_the_version_is_reported(self) -> None:
        self.assertEqual(__version__, "0.1.0")
        completed = subprocess.run(
            [sys.executable, "-m", "tradelab", "--version"],
            cwd=ROOT, capture_output=True, text=True, timeout=120,
        )
        self.assertEqual(completed.returncode, 0)
        self.assertIn(__version__, completed.stdout + completed.stderr)


class PublicSurfaceTests(unittest.TestCase):
    def test_everything_advertised_by_all_actually_exists(self) -> None:
        import tradelab

        for name in tradelab.__all__:
            with self.subTest(name=name):
                self.assertTrue(hasattr(tradelab, name), f"tradelab.{name} is missing")

    def test_the_documented_five_line_workflow_runs(self) -> None:
        from tradelab import Backtester, render_report, summarize
        from tradelab.data import synthetic_random_walk as walk
        from tradelab.strategies import DonchianBreakout

        result = Backtester().run(DonchianBreakout(), walk(600, seed=1))
        text = render_report(result, summarize(result))
        self.assertIn("tradelab backtest report", text)


if __name__ == "__main__":
    unittest.main()
