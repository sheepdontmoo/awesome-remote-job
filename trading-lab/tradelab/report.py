"""Turning a result into prose that argues with itself.

A backtest report is a persuasive document whether its author intended one or
not. Pick the flattering numbers, put them at the top in a large font, and the
reader concludes there is an edge. Every reporting choice made here pushes the
other way:

* **Gross and net sit next to each other, with the arithmetic shown.** The
  costs section is a ladder from what a naive backtest would report down to
  what would have reached the account, one line per cost. That gap is the
  whole story of most retail strategies and it should be impossible to skim
  past.
* **The drawdown is drawn, not just tabulated.** A 30% maximum drawdown is a
  number; fourteen rows of underwater plot showing it lasted eight months is
  an experience, and the second one is what actually decides whether a system
  is tradable by the person reading the report.
* **The caveats come last, so the report ends on them.** Everything above is
  what the rules did on one sample at one set of assumed costs. The closing
  section states what that is not entitled to claim, and when nothing
  triggers it says so without turning the silence into praise.
* **No forward-looking figure is ever presented as an expectation.** CAGR and
  the annualised ratios appear, because they make samples of different lengths
  comparable, and they are labelled as restatements of the past every time.
  There is no "projected annual return" line here and there never will be.

Output is plain ASCII at a fixed width: no colour, no box-drawing characters,
no emoji. It has to survive a terminal, an email, a code comment and a
screenshot without turning into mojibake, and a report that renders as
question marks is a report nobody reads twice.
"""

from __future__ import annotations

import dataclasses
import math
import re
import textwrap
from datetime import datetime
from typing import Any, Iterable, Mapping, Optional, Sequence

from .metrics import Metrics, summarize
from .types import BacktestResult, EquityPoint

__all__ = ["render_report", "render_comparison", "render_validation"]

_LABEL_WIDTH = 28
_VALUE_WIDTH = 13
_PLOT_WIDTH = 60
_PLOT_HEIGHT = 14
_UNDERWATER_HEIGHT = 6
_GUTTER = 9

_NO_CAVEATS = "No automatic caveats triggered. This is not an endorsement."

# Field names in a validation result that read as sentences rather than
# numbers. They are pulled out of the table and wrapped as prose at the end of
# their block, because a conclusion squeezed into a 14-character value column
# is a conclusion nobody reads.
_PROSE_FIELDS = (
    "verdict", "conclusion", "summary", "interpretation", "caveat", "caveats",
    "note", "notes", "warning", "warnings", "recommendation", "message",
)


# ---------------------------------------------------------------------------
# small formatting helpers
# ---------------------------------------------------------------------------


def _rule(width: int, char: str = "-") -> str:
    return char * width


def _section(title: str, width: int) -> str:
    prefix = f"-- {title} "
    return (prefix + "-" * max(0, width - len(prefix)))[:width]


def _kv(label: str, value: str, width: int, note: str = "") -> str:
    """One label/value row, with an optional note in the right-hand margin.

    A note that will not fit moves to its own wrapped line rather than being
    truncated. Half a sentence about what a number means is worse than none:
    it reads as complete and says something different from what was written.
    """
    value = str(value)
    line = "  " + label.ljust(_LABEL_WIDTH) + value.rjust(_VALUE_WIDTH)
    if len(line) > width:
        compact = "  " + label + "  " + value
        line = compact if len(compact) <= width else "  " + label + "\n" + value.rjust(width)
    if not note:
        return line
    if len(line) + 2 + len(note) <= width:
        return line + "  " + note
    return line + "\n" + "\n".join(_wrap(note, width, indent=" " * 6))


def _wrap(text: str, width: int, indent: str = "  ", first: Optional[str] = None) -> list[str]:
    return textwrap.wrap(
        text,
        width=max(20, width),
        initial_indent=first if first is not None else indent,
        subsequent_indent=indent,
    ) or [indent.rstrip()]


def _money(value: Optional[float], signed: bool = False) -> str:
    if value is None:
        return "n/a"
    return f"{value:+,.2f}" if signed else f"{value:,.2f}"


def _pct(value: Optional[float], digits: int = 2) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.{digits}f}%"


def _ratio(value: Optional[float], digits: int = 2) -> str:
    if value is None:
        return "n/a"
    if not math.isfinite(value):
        return "n/a"
    return f"{value:.{digits}f}"


def _plain(lines: Sequence[str]) -> str:
    """Join rendered lines and strip trailing whitespace from every one.

    Rows are built by padding to fixed columns, which leaves trailing spaces on
    any line whose right-hand side is empty. They are invisible until someone
    diffs two reports or pastes one into an editor that highlights them.
    """
    text = "\n".join(lines)
    return "\n".join(line.rstrip() for line in text.split("\n"))


def _timestamp(ts: Optional[datetime]) -> str:
    return ts.strftime("%Y-%m-%d %H:%M") if ts else "n/a"


def _describe_interval(seconds: float) -> str:
    if seconds <= 0:
        return "unknown"
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes = seconds / 60.0
    if minutes < 60:
        return f"{minutes:.0f}m"
    hours = minutes / 60.0
    if hours < 24:
        return f"{hours:.0f}h"
    return f"{hours / 24.0:.1f}d"


def _median_gap_seconds(curve: Sequence[EquityPoint]) -> Optional[float]:
    if len(curve) < 2:
        return None
    gaps = sorted(
        (curve[i].ts - curve[i - 1].ts).total_seconds()
        for i in range(1, len(curve))
    )
    gaps = [g for g in gaps if g > 0]
    if not gaps:
        return None
    return gaps[len(gaps) // 2]


# ---------------------------------------------------------------------------
# ASCII plots
# ---------------------------------------------------------------------------


def _columnise(curve: Sequence[EquityPoint], columns: int) -> list[tuple[float, float]]:
    """Reduce an arbitrarily long curve to `columns` (equity, drawdown) pairs.

    Equity takes the *last* value in each bucket, so the right-hand end of the
    plot is the true final equity. Drawdown takes the *worst* value in the
    bucket rather than the last: compressing a curve should never make a
    drawdown look shallower than it was.
    """
    n = len(curve)
    out: list[tuple[float, float]] = []
    previous = (curve[0].equity, curve[0].drawdown)
    for col in range(columns):
        start = col * n // columns
        stop = (col + 1) * n // columns
        if stop <= start:
            out.append(previous)
            continue
        chunk = curve[start:stop]
        previous = (chunk[-1].equity, max(point.drawdown for point in chunk))
        out.append(previous)
    return out


def _equity_plot(
    curve: Sequence[EquityPoint],
    initial_equity: float,
    width: int,
    plot_width: int = _PLOT_WIDTH,
    height: int = _PLOT_HEIGHT,
) -> list[str]:
    """Equity as a line plot, with the underwater curve directly beneath it.

    The two share an x-axis on purpose. Looking at an equity curve alone, the
    eye follows the upward slope; stacking the drawdown under it puts the
    duration of each decline in the same glance as the recovery.
    """
    if len(curve) < 2:
        return ["  (equity curve too short to plot)"]

    plot_width = max(20, min(plot_width, width - _GUTTER - 3))
    columns = _columnise(curve, plot_width)
    values = [equity for equity, _ in columns]
    drawdowns = [dd for _, dd in columns]

    high = max(values + [initial_equity])
    low = min(values + [initial_equity])
    if high - low < 1e-9:
        high, low = high + 1.0, low - 1.0

    def row_of(value: float) -> int:
        fraction = (high - value) / (high - low)
        return max(0, min(height - 1, int(round(fraction * (height - 1)))))

    grid = [[" "] * plot_width for _ in range(height)]
    base_row = row_of(initial_equity)
    for col in range(plot_width):
        grid[base_row][col] = "."

    previous_row: Optional[int] = None
    for col, value in enumerate(values):
        row = row_of(value)
        if previous_row is not None:
            step = 1 if row > previous_row else -1
            for between in range(previous_row + step, row, step):
                if grid[between][col] in (" ", "."):
                    grid[between][col] = "|"
        grid[row][col] = "*"
        previous_row = row

    lines: list[str] = []
    for index, row_cells in enumerate(grid):
        if index == 0:
            label = f"{high:>{_GUTTER},.0f}"
        elif index == height - 1:
            label = f"{low:>{_GUTTER},.0f}"
        elif index == base_row:
            label = f"{initial_equity:>{_GUTTER},.0f}"
        else:
            label = " " * _GUTTER
        lines.append(f"{label} |" + "".join(row_cells))

    lines.append(" " * _GUTTER + " +" + "-" * plot_width)

    start_label = _timestamp(curve[0].ts)
    end_label = _timestamp(curve[-1].ts)
    axis = start_label.ljust(plot_width - len(end_label)) + end_label
    lines.append(" " * (_GUTTER + 2) + axis[:plot_width])

    lines.append("")
    lines.extend(_underwater_plot(drawdowns, plot_width))
    return lines


def _underwater_plot(drawdowns: Sequence[float], plot_width: int) -> list[str]:
    """Drawdown depth, filled downward from a surface line at zero."""
    worst = max(drawdowns) if drawdowns else 0.0
    height = _UNDERWATER_HEIGHT
    grid = [[" "] * plot_width for _ in range(height)]
    grid[0] = ["-"] * plot_width

    if worst > 0:
        for col, depth in enumerate(drawdowns[:plot_width]):
            if depth <= 0:
                continue
            rows = int(round(depth / worst * (height - 1)))
            if rows < 1:
                # Too shallow for a full block, but not zero. A dot rather than
                # a blank: a plot that rounds small drawdowns away is prettier
                # than the account it describes.
                if grid[1][col] == " ":
                    grid[1][col] = "."
                continue
            for row in range(1, min(rows, height - 1) + 1):
                grid[row][col] = "#"

    lines = []
    for index, row_cells in enumerate(grid):
        if index == 0:
            label = "0%".rjust(_GUTTER)
        elif index == height - 1:
            label = f"-{worst * 100:.1f}%".rjust(_GUTTER)
        else:
            label = " " * _GUTTER
        lines.append(f"{label} |" + "".join(row_cells))
    lines.append(" " * (_GUTTER + 2) + "underwater (depth below running peak)")
    return lines


# ---------------------------------------------------------------------------
# report sections
# ---------------------------------------------------------------------------


def _header_block(result: BacktestResult, metrics: Metrics, width: int) -> list[str]:
    curve = result.equity_curve
    instrument = result.instrument
    costs = result.costs
    risk = result.risk

    first_ts = curve[0].ts if curve else None
    last_ts = curve[-1].ts if curve else None
    span_days = (last_ts - first_ts).total_seconds() / 86400.0 if curve else 0.0
    gap = _median_gap_seconds(curve)

    lines = [
        _kv("Strategy", result.strategy_name or "(unnamed)", width),
        _kv(
            "Instrument",
            instrument.symbol,
            width,
            f"point {instrument.point:g}, size {instrument.contract_size:,.0f}",
        ),
        _kv(
            "Period (UTC)",
            f"{_timestamp(first_ts)} -> {_timestamp(last_ts)}",
            width,
        ),
        _kv("Span", f"{span_days:,.1f} days", width, f"bars of {_describe_interval(gap or 0.0)}"),
        _kv("Bars processed", f"{result.bars_processed:,}", width, f"{len(curve):,} equity points"),
        _kv(
            "Spread / slippage assumed",
            f"{costs.spread_points:.1f} / {costs.slippage_points:.1f}",
            width,
            "points, " + ("mid prices" if costs.price_is_mid else "bid prices"),
        ),
        _kv(
            "Swap long / short",
            f"{costs.swap_long_points:+.2f} / {costs.swap_short_points:+.2f}",
            width,
            "points per night",
        ),
        _kv(
            "Commission",
            f"{instrument.commission_per_lot_per_side:.2f}",
            width,
            "per lot per side",
        ),
        _kv(
            "Risk per trade",
            _pct(risk.risk_per_trade),
            width,
            f"max {risk.max_lots:g} lots, min stop {risk.min_stop_points:g} pt",
        ),
        _kv(
            "Equity",
            f"{result.initial_equity:,.2f} -> {metrics.final_equity:,.2f}",
            width,
        ),
    ]
    if result.params:
        joined = ", ".join(f"{key}={value}" for key, value in sorted(result.params.items()))
        lines.extend(_wrap(f"Parameters: {joined}", width))
    if not instrument.quote_is_account_currency:
        lines.extend(
            _wrap(
                "Account currency differs from the quote currency, so every cash "
                "figure below is an approximation that ignores the conversion rate "
                "at the time of each fill.",
                width,
            )
        )
    return lines


def _performance_block(result: BacktestResult, metrics: Metrics, width: int) -> list[str]:
    curve = result.equity_curve
    sample_years: Optional[float] = None
    if len(curve) >= 2 and metrics.periods_per_year:
        sample_years = len(curve) / metrics.periods_per_year

    lines = [
        _kv("Net P&L (after every cost)", _money(metrics.net_pnl, signed=True), width),
        _kv("Total return", _pct(metrics.total_return), width, "over the whole sample"),
        _kv("CAGR", _pct(metrics.cagr), width, "sample restated per year"),
        _kv("Sharpe (annualised, rf = 0)", _ratio(metrics.sharpe), width),
        _kv("Sortino (annualised, rf = 0)", _ratio(metrics.sortino), width),
        _kv("Calmar", _ratio(metrics.calmar), width),
        _kv("Profit factor", _ratio(metrics.profit_factor), width),
        _kv("Expectancy per trade", f"{_ratio(metrics.expectancy_r)} R", width),
        _kv("Tail ratio (p95 / |p05|)", _ratio(metrics.tail_ratio), width),
    ]
    if metrics.periods_per_year:
        lines.append(
            _kv("Annualisation basis", f"{metrics.periods_per_year:,.0f} /yr", width)
        )
    lines.append("")
    lines.extend(
        _wrap(
            "Annualised figures restate what this sample did at a yearly rate so "
            "that runs of different lengths can be compared. They are descriptions "
            "of the past. Nothing here is an expected or projected future return.",
            width,
        )
    )
    if sample_years is not None and sample_years < 1.0:
        lines.append("")
        lines.extend(
            _wrap(
                f"This sample covers {sample_years:.2f} of a trading year. "
                "Annualising it multiplies the sampling error along with the "
                "return: the CAGR and the ratios above are arithmetic performed "
                "on a short run, not a rate the account sustained for a year.",
                width,
            )
        )
    return lines


def _risk_block(result: BacktestResult, metrics: Metrics, width: int) -> list[str]:
    recovery = (
        _pct(1.0 / (1.0 - metrics.max_drawdown) - 1.0)
        if 0.0 < metrics.max_drawdown < 1.0
        else "n/a"
    )
    return [
        _kv(
            "Max drawdown",
            _pct(metrics.max_drawdown),
            width,
            f"{_money(metrics.max_drawdown_cash)} cash, {recovery} to recover",
        ),
        _kv(
            "Longest drawdown",
            f"{metrics.longest_drawdown_bars:,} bars",
            width,
            f"{metrics.longest_drawdown_days:,.1f} days underwater",
        ),
        _kv("Ulcer index", f"{metrics.ulcer_index:.2f}", width, "RMS drawdown, percent"),
        _kv("Max consecutive losses", f"{metrics.max_consecutive_losses:,}", width),
        _kv("Time in market", _pct(metrics.time_in_market, digits=1), width, "fraction of bars holding"),
        _kv("Halted on ruin threshold", "yes" if result.halted_early else "no", width),
    ]


def _trades_block(result: BacktestResult, metrics: Metrics, width: int) -> list[str]:
    lines = [
        _kv("Closed trades", f"{metrics.n_trades:,}", width),
        _kv("Win rate", _pct(metrics.win_rate, digits=1), width),
        _kv("Average win", _money(metrics.avg_win, signed=True), width),
        _kv("Average loss", _money(metrics.avg_loss, signed=True), width),
        _kv("Payoff ratio", _ratio(metrics.payoff_ratio), width, "avg win / |avg loss|"),
        _kv("Average R", _ratio(metrics.avg_r), width),
        _kv(
            "Best / worst R",
            f"{_ratio(metrics.best_r)} / {_ratio(metrics.worst_r)}",
            width,
        ),
        _kv("Average bars held", _ratio(metrics.avg_bars_held, digits=1), width),
        _kv("Trades per year", _ratio(metrics.trades_per_year, digits=1), width),
    ]
    if not result.trades:
        return lines

    lines.append("")
    lines.append("  Exits, and the net P&L attributable to each:")
    tally: dict[str, list[float]] = {}
    for trade in result.trades:
        reason = getattr(trade.exit_reason, "value", str(trade.exit_reason))
        tally.setdefault(reason, []).append(trade.net_pnl)
    total = len(result.trades)
    for reason, pnls in sorted(tally.items(), key=lambda item: -len(item[1])):
        share = len(pnls) / total
        lines.append(
            _kv(
                f"    {reason}",
                f"{len(pnls):,}",
                width,
                f"{share * 100:5.1f}% of exits, {_money(sum(pnls), signed=True)}",
            )
        )
    return lines


def _costs_block(result: BacktestResult, metrics: Metrics, width: int) -> list[str]:
    ladder_width = min(width, _LABEL_WIDTH + _VALUE_WIDTH + 4)
    lines = [
        _kv("Gross P&L (naive backtest)", _money(metrics.gross_pnl, signed=True), width),
        _kv("  Commission", _money(-metrics.commission_total, signed=True), width),
        _kv("  Spread", _money(-metrics.spread_total, signed=True), width),
        _kv("  Slippage", _money(-metrics.slippage_total, signed=True), width),
        _kv("  Swap / financing", _money(metrics.swap_total, signed=True), width),
        "  " + _rule(ladder_width - 2),
        _kv("Net P&L (reaches the account)", _money(metrics.net_pnl, signed=True), width),
        "",
        _kv("Total costs", _money(metrics.total_costs), width, "positive = paid away"),
        _kv(
            "Costs vs starting equity",
            _pct(metrics.total_costs / result.initial_equity)
            if result.initial_equity
            else "n/a",
            width,
            "of the account, to the broker",
        ),
        _kv(
            "Cost drag",
            _pct(metrics.cost_drag, digits=1),
            width,
            "of |gross P&L|",
        ),
        _kv(
            "Break-even edge",
            _ratio(metrics.break_even_edge_points, digits=1) + " pt",
            width,
            "needed per round turn",
        ),
    ]
    if metrics.n_trades:
        lines.append(
            _kv(
                "Cost per round turn",
                _money(metrics.total_costs / metrics.n_trades),
                width,
            )
        )
    lines.append("")
    lines.extend(
        _wrap(
            "The gross line is what a backtest without a cost model would have "
            "printed. Every line under it was charged on assumptions, not on "
            "fills you were given: real spread widens around news and the rollover, "
            "and real slippage is not a constant. Re-run at CostConfig.stress(2.0) "
            "and treat the result of that as the honest one.",
            width,
        )
    )
    return lines


def _warning_shape(message: str) -> str:
    """Grouping key: the warning with every number and timestamp flattened.

    A run that skips two thousand trades emits two thousand warnings that
    differ only in the date and the stop distance. Printing ten of them and
    hiding the rest reads as ten separate problems; printing the shape once
    with a count reads as the one problem it is.
    """
    return re.sub(r"[0-9][0-9.:,\-]*", "#", message)


def _warnings_block(result: BacktestResult, width: int) -> list[str]:
    if not result.warnings:
        return ["  None. The engine completed without complaint."]

    tally: dict[str, list[str]] = {}
    for message in result.warnings:
        tally.setdefault(_warning_shape(message), []).append(message)

    kinds = sorted(tally.values(), key=len, reverse=True)
    lines = _wrap(
        f"{len(result.warnings):,} warning(s) of {len(kinds):,} kind(s), grouped "
        "with numbers and timestamps ignored:",
        width,
    )
    for group in kinds[:10]:
        prefix = f"  [x{len(group):,}] " if len(group) > 1 else "  - "
        lines.extend(_wrap(group[0], width, indent=" " * 6, first=prefix))
    if len(kinds) > 10:
        lines.append(f"  ... and {len(kinds) - 10:,} further kind(s).")
    return lines


def _caveats_block(result: BacktestResult, metrics: Metrics, width: int) -> list[str]:
    """The closing section: everything the numbers above are not entitled to claim.

    Sources are `Metrics.caveats()` and the engine's own warnings. The warnings
    are already listed in full above, so only their existence is repeated here
    - and only when the metrics have not already said it, which happens when a
    `Metrics` computed elsewhere is passed in alongside this result.
    """
    caveats = [c for c in metrics.caveats() if c.strip()]
    substantive = [c for c in caveats if c.strip() != _NO_CAVEATS]

    if result.warnings and not any("warning" in c.lower() for c in substantive):
        substantive.append(
            f"The engine emitted {len(result.warnings):,} warning(s) during this "
            "run. They are listed above and each one is a bar on which the "
            "strategy did not do what its rules say it does."
        )

    if not substantive:
        return _wrap(_NO_CAVEATS, width)
    caveats = substantive

    lines: list[str] = []
    for index, caveat in enumerate(caveats, start=1):
        if index > 1:
            lines.append("")
        lines.extend(_wrap(caveat, width, indent=" " * 6, first=f"  {index:>2}. "))
    return lines


# ---------------------------------------------------------------------------
# public renderers
# ---------------------------------------------------------------------------


def render_report(result: BacktestResult, metrics: Metrics, width: int = 78) -> str:
    """Render a full backtest report as fixed-width ASCII.

    Sections run header, equity, performance, risk, trades, costs, warnings,
    caveats - and the caveats are last because that is the sentence the reader
    should leave with. Nothing in the output is a forecast.
    """
    width = max(48, int(width))
    out: list[str] = [
        _rule(width, "="),
        "tradelab backtest report",
        _rule(width, "="),
    ]
    out.extend(_header_block(result, metrics, width))

    out.append("")
    out.append(_section("EQUITY", width))
    out.extend(_equity_plot(result.equity_curve, result.initial_equity, width))

    for title, block in (
        ("PERFORMANCE", _performance_block(result, metrics, width)),
        ("RISK", _risk_block(result, metrics, width)),
        ("TRADES", _trades_block(result, metrics, width)),
        ("COSTS", _costs_block(result, metrics, width)),
        ("WARNINGS", _warnings_block(result, width)),
        ("CAVEATS", _caveats_block(result, metrics, width)),
    ):
        out.append("")
        out.append(_section(title, width))
        out.extend(block)

    return _plain(out)


def _normalise_named(
    named_results: Any,
) -> list[tuple[str, BacktestResult, Metrics]]:
    """Accept a mapping or a sequence of pairs/triples, and fill in metrics."""
    entries: Iterable[Any]
    if isinstance(named_results, Mapping):
        entries = list(named_results.items())
    else:
        entries = list(named_results)

    out: list[tuple[str, BacktestResult, Metrics]] = []
    for entry in entries:
        items = list(entry)
        name = str(items[0])
        if len(items) >= 3:
            result, metrics = items[1], items[2]
        else:
            payload = items[1]
            if isinstance(payload, (tuple, list)):
                result, metrics = payload[0], (payload[1] if len(payload) > 1 else None)
            else:
                result, metrics = payload, None
        if metrics is None:
            metrics = summarize(result)
        out.append((name, result, metrics))
    return out


def render_comparison(named_results: Any, width: int = 78) -> str:
    """Render several runs side by side in one table.

    Accepts a mapping of name -> `BacktestResult`, or a sequence of
    `(name, result)` / `(name, result, metrics)`. Metrics are computed where
    they are not supplied.

    The table exists to make one comparison easy and one comparison
    uncomfortable. The easy one is a strategy against its own baselines - a
    random-entry monkey, the same rules on `synthetic_random_walk` data, the
    same rules at doubled costs. The uncomfortable one is a parameter sweep,
    where the best row is best partly because it is the maximum of a sample of
    noisy estimates, and the footer says so.
    """
    width = max(48, int(width))
    rows = _normalise_named(named_results)
    if not rows:
        return "No runs to compare."

    # 59 columns are spoken for by the numeric fields; the name gets the rest.
    name_width = max(4, min(24, width - 59))
    header = (
        "Run".ljust(name_width)
        + "Trades".rjust(8)
        + "Net P&L".rjust(12)
        + "Return".rjust(9)
        + "MaxDD".rjust(8)
        + "Sharpe".rjust(8)
        + "PF".rjust(7)
        + "Exp R".rjust(7)
    )
    out = [
        _rule(width, "="),
        "tradelab comparison",
        _rule(width, "="),
        header[:width],
        _rule(min(width, len(header))),
    ]
    for name, _result, metrics in rows:
        label = name if len(name) <= name_width - 1 else name[: name_width - 2] + "~"
        out.append(
            (
                label.ljust(name_width)
                + f"{metrics.n_trades:,}".rjust(8)
                + _money(metrics.net_pnl, signed=True).rjust(12)
                + _pct(metrics.total_return, digits=1).rjust(9)
                + _pct(metrics.max_drawdown, digits=1).rjust(8)
                + _ratio(metrics.sharpe).rjust(8)
                + _ratio(metrics.profit_factor).rjust(7)
                + _ratio(metrics.expectancy_r).rjust(7)
            )[:width]
        )

    out.append("")
    out.append(_section("CAVEAT COUNT", width))
    for name, result, metrics in rows:
        caveats = [c for c in metrics.caveats() if c.strip() != _NO_CAVEATS]
        out.append(
            _kv(
                name[:_LABEL_WIDTH],
                f"{len(caveats):,} caveat" + ("" if len(caveats) == 1 else "s"),
                width,
                f"{len(result.warnings):,} engine warnings",
            )
        )

    out.append("")
    out.append(_section("READING THIS TABLE", width))
    out.extend(
        _wrap(
            f"These are {len(rows)} chances to be lucky on the same data. The best "
            "row is the maximum of a set of noisy estimates, and the maximum of a "
            "noisy set is biased upward even when every underlying edge is zero - "
            "which is exactly what a parameter sweep on random data demonstrates. "
            "Before believing the winner, deflate its Sharpe by the number of "
            "configurations tried, counting the ones abandoned along the way, and "
            "check that its neighbours in parameter space are also profitable. A "
            "lone spike surrounded by losers is a fit, not an edge.",
            width,
        )
    )
    return _plain(out)


# ---------------------------------------------------------------------------
# validation rendering - deliberately generic
# ---------------------------------------------------------------------------


def _auto_number(value: float) -> str:
    if not math.isfinite(value):
        return "n/a"
    magnitude = abs(value)
    if magnitude >= 1000:
        return f"{value:,.0f}"
    if magnitude >= 1:
        return f"{value:,.3f}"
    return f"{value:.4f}"


def _fields_of(obj: Any) -> list[tuple[str, Any]]:
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return [(f.name, getattr(obj, f.name)) for f in dataclasses.fields(obj)]
    if isinstance(obj, Mapping):
        return [(str(key), value) for key, value in obj.items()]
    if hasattr(obj, "__dict__") and vars(obj):
        return [(key, value) for key, value in vars(obj).items() if not key.startswith("_")]
    return []


def _inline_value(value: Any) -> Optional[str]:
    """One-line rendering of a scalar, or None when it needs its own block."""
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return _auto_number(value)
    if isinstance(value, datetime):
        return _timestamp(value)
    if isinstance(value, str):
        return value if len(value) <= _VALUE_WIDTH else None
    return None


def _sequence_summary(values: Sequence[Any]) -> Optional[str]:
    if not values:
        return "empty"
    if all(isinstance(v, bool) for v in values):
        return f"n={len(values)}, {sum(1 for v in values if v)} true"
    if all(isinstance(v, (int, float)) for v in values):
        ordered = sorted(float(v) for v in values)
        median = ordered[len(ordered) // 2]
        return (
            f"n={len(ordered)}  min {_auto_number(ordered[0])}  "
            f"med {_auto_number(median)}  max {_auto_number(ordered[-1])}"
        )
    return None


def _compact(obj: Any, limit: int) -> str:
    """A nested record squeezed onto one line, for list elements."""
    parts = []
    for name, value in _fields_of(obj)[:6]:
        inline = _inline_value(value)
        if inline is None:
            inline = str(value)
        parts.append(f"{name}={inline}")
    text = ", ".join(parts) if parts else str(obj)
    return text if len(text) <= limit else text[: limit - 1] + "~"


def _render_record(obj: Any, width: int, depth: int = 0) -> list[str]:
    fields = _fields_of(obj)
    pad = "  " + "  " * depth
    if not fields:
        return _wrap(str(obj), width, indent=pad)

    def row(label: str, value: str) -> str:
        label_room = max(8, _LABEL_WIDTH - 2 * depth)
        return (pad + label.ljust(label_room) + value.rjust(_VALUE_WIDTH))[:width]

    lines: list[str] = []
    prose: list[tuple[str, Any]] = []
    for name, value in fields:
        if name.lower() in _PROSE_FIELDS or (isinstance(value, str) and len(value) > _VALUE_WIDTH):
            prose.append((name, value))
            continue

        label = name.replace("_", " ")
        inline = _inline_value(value)
        if inline is not None:
            lines.append(row(label, inline))
            continue

        if isinstance(value, (list, tuple)):
            summary = _sequence_summary(value)
            if summary is not None:
                lines.append(row(label, summary))
                continue
            lines.append(f"{pad}{label} ({len(value)}):")
            for element in list(value)[:10]:
                lines.append(pad + "    " + _compact(element, width - len(pad) - 4))
            if len(value) > 10:
                lines.append(f"{pad}    ... and {len(value) - 10:,} more")
            continue

        if _fields_of(value) and depth < 2:
            lines.append(f"{pad}{label}:")
            lines.extend(_render_record(value, width, depth=depth + 1))
            continue

        lines.append(row(label, str(value)))

    for name, value in prose:
        lines.append("")
        if isinstance(value, (list, tuple)):
            for element in value:
                lines.extend(_wrap(str(element), width, indent="     ", first="  -  "))
        else:
            lines.extend(_wrap(f"{name.replace('_', ' ')}: {value}", width))
    return lines


def _humanise(name: str) -> str:
    out: list[str] = []
    for index, char in enumerate(name):
        if char.isupper() and index and not name[index - 1].isupper():
            out.append(" ")
        out.append(char)
    return "".join(out).replace("_", " ").strip().upper()


def render_validation(*validation_results: Any, width: int = 78) -> str:
    """Render the output of `validation.py` - whatever shape it arrived in.

    Deliberately structural rather than typed: it reads dataclass fields (or a
    mapping, or an object's attributes) and lays them out, so a new validation
    routine gets a readable block without this module needing to know it
    exists. Fields that read as sentences - verdict, conclusion, caveats - are
    pulled out and wrapped as prose underneath their table.

    Nothing here converts a p-value into a verdict about a strategy. A
    permutation test that fails to reject the null has not proved the strategy
    worthless, and one that rejects it has not proved the edge is real - it has
    ruled out one specific way of being fooled, on one sample, at one set of
    assumed costs.
    """
    width = max(48, int(width))
    out = [
        _rule(width, "="),
        "tradelab validation",
        _rule(width, "="),
    ]
    if not validation_results:
        out.extend(_wrap("No validation results supplied.", width))
        return _plain(out)

    for item in validation_results:
        title = _humanise(type(item).__name__) if not isinstance(item, Mapping) else "RESULT"
        out.append("")
        out.append(_section(title, width))
        out.extend(_render_record(item, width))

    out.append("")
    out.append(_section("WHAT THIS DOES NOT SHOW", width))
    out.extend(
        _wrap(
            "Every test above was run against the same history the rules were "
            "chosen on, with the same cost assumptions and the same survivorship "
            "of an instrument that still trades. Passing them removes specific "
            "ways of being wrong; it does not supply evidence of an edge, and no "
            "arrangement of historical data can. The next test is a demo account "
            "and several months of waiting.",
            width,
        )
    )
    return _plain(out)
