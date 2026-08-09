"""Getting bars in, and manufacturing bars whose answer is already known.

Two jobs live here, and they exist for the same reason.

The first is loading. Price files are where a backtest goes wrong before a
single rule has run: a duplicated timestamp, a row where high < low, a
midnight bar the exporter wrote as `24:00`, a broker whose "server time" is
UTC+2 in winter and UTC+3 in summer. None of these crash a naive loader. They
quietly change the answer. So the loaders here validate rather than repair,
and every failure names the exact row in the file that caused it. A loader
that silently drops bad rows is a loader that silently drops evidence.

The second is fabrication, and it is the more useful half.
`synthetic_random_walk` produces a price series with **no exploitable edge, by
construction**: independent, zero-mean log returns with no drift and no
autocorrelation. Any strategy scored against it is being scored on luck alone.
That number - the Sharpe, the profit factor, the equity curve that goes up and
to the right - is the score to beat, and most real strategies do not beat it
by enough to matter. Running against noise before running against history is
the cheapest way to find out that an "edge" is a pattern-matching artefact.

`synthetic_trending` is the control for the control: it embeds a genuine,
known autocorrelation, so a trend-following harness that finds nothing there
is broken rather than merely honest.

Conventions
-----------
* Every timestamp returned is timezone-aware UTC. Naive input is *assumed*
  UTC and that assumption is documented at each call site, never hidden.
* Bars must be strictly increasing in time. Equal timestamps are an error,
  not a duplicate to be de-duplicated: which of the two rows is the real bar
  is not something this module can know.
* Nothing here interpolates, forward-fills, or otherwise invents a price.
  Gaps stay gaps; `summary()` reports them so they can be looked at.
"""

from __future__ import annotations

import csv
import math
import random
import statistics
from datetime import datetime, timedelta, timezone
from typing import Optional, Sequence

from .types import Bar

__all__ = [
    "DataError",
    "load_csv",
    "load_mt5_csv",
    "save_csv",
    "resample",
    "summary",
    "synthetic_random_walk",
    "synthetic_trending",
]


class DataError(ValueError):
    """A price file said something impossible, and here is the row that said it."""


# ---------------------------------------------------------------------------
# header and delimiter detection
# ---------------------------------------------------------------------------

# Candidate delimiters in priority order. Tab first: MetaTrader exports are
# tab-separated and a tab-separated file never legitimately contains commas in
# a numeric field, so a tab anywhere in the header settles the question.
_DELIMITERS = ("\t", ",", ";", "|")

_TIMESTAMP_KEYS = ("timestamp", "datetime", "date_time", "bar_time", "open_time", "ts")
_DATE_KEYS = ("date",)
_TIME_KEYS = ("time",)
_OPEN_KEYS = ("open", "o")
_HIGH_KEYS = ("high", "h")
_LOW_KEYS = ("low", "l")
_CLOSE_KEYS = ("close", "c", "last")
# `tickvol` outranks `vol` on purpose: an MT5 export carries both, and its
# `<VOL>` column is real traded size, which on a retail FX feed is always zero.
_VOLUME_KEYS = ("volume", "tickvol", "tick_volume", "tickvolume", "vol", "v")

# Formats tried when the caller does not supply their own. Deliberately absent:
# anything of the form D/M/Y or M/D/Y. `03/04/2020` is either the third of
# April or the fourth of March and no amount of cleverness makes that knowable
# from one row. Guessing would corrupt a backtest in a way nothing downstream
# could detect, so the loader refuses and asks for `timestamp_formats`.
_DEFAULT_TIMESTAMP_FORMATS = (
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%dT%H:%M",
    "%Y-%m-%d",
    "%Y.%m.%d %H:%M:%S",
    "%Y.%m.%d %H:%M",
    "%Y.%m.%d",
    "%Y/%m/%d %H:%M:%S",
    "%Y/%m/%d %H:%M",
    "%Y/%m/%d",
    "%Y%m%d %H:%M:%S",
    "%Y%m%d %H%M%S",
    "%Y%m%d",
)


def _normalise_key(cell: str) -> str:
    """Header cell -> comparable key. Handles MT5's `<OPEN>` and stray BOMs."""
    text = cell.strip().lstrip("\ufeff").strip()
    text = text.strip("<>").strip()
    text = text.replace(" ", "_").replace("-", "_")
    return text.lower()


def _sniff_delimiter(line: str) -> str:
    best = ","
    best_count = 0
    for candidate in _DELIMITERS:
        count = line.count(candidate)
        if count > best_count:
            best, best_count = candidate, count
    return best


def _find_column(keys: Sequence[str], wanted: Sequence[str]) -> Optional[int]:
    for want in wanted:
        if want in keys:
            return keys.index(want)
    return None


class _Layout:
    """Which column holds what, resolved once from the header."""

    __slots__ = ("timestamp", "date", "time", "open", "high", "low", "close", "volume")

    def __init__(self) -> None:
        self.timestamp: Optional[int] = None
        self.date: Optional[int] = None
        self.time: Optional[int] = None
        self.open: int = 1
        self.high: int = 2
        self.low: int = 3
        self.close: int = 4
        self.volume: Optional[int] = None

    @property
    def max_index(self) -> int:
        indices = [self.open, self.high, self.low, self.close]
        for optional in (self.timestamp, self.date, self.time, self.volume):
            if optional is not None:
                indices.append(optional)
        return max(indices)


def _looks_like_header(cells: Sequence[str]) -> bool:
    """True when a row is labels rather than prices.

    The test is the last four fields: in any data row those are OHLC and parse
    as floats. A header that fails `_layout_from_header` is worth a different
    error message from a data row, because the fix is different - rename a
    column, rather than add one.
    """
    tail = [cell.strip().strip('"') for cell in cells[-4:]]
    if len(tail) < 4:
        return True
    for cell in tail:
        try:
            float(cell.replace(",", ""))
        except ValueError:
            return True
    return False


def _positional_layout() -> _Layout:
    """ts, open, high, low, close[, volume] - the headerless fallback."""
    layout = _Layout()
    layout.timestamp = 0
    layout.open, layout.high, layout.low, layout.close = 1, 2, 3, 4
    return layout


def _layout_from_header(cells: Sequence[str]) -> Optional[_Layout]:
    """Resolve a header row, or None if this row is not a header at all."""
    keys = [_normalise_key(cell) for cell in cells]
    layout = _Layout()

    open_idx = _find_column(keys, _OPEN_KEYS)
    high_idx = _find_column(keys, _HIGH_KEYS)
    low_idx = _find_column(keys, _LOW_KEYS)
    close_idx = _find_column(keys, _CLOSE_KEYS)
    if None in (open_idx, high_idx, low_idx, close_idx):
        return None

    date_idx = _find_column(keys, _DATE_KEYS)
    time_idx = _find_column(keys, _TIME_KEYS)
    stamp_idx = _find_column(keys, _TIMESTAMP_KEYS)

    if date_idx is not None and time_idx is not None:
        layout.date, layout.time = date_idx, time_idx
    elif stamp_idx is not None:
        layout.timestamp = stamp_idx
    elif date_idx is not None:
        layout.timestamp = date_idx
    elif time_idx is not None:
        layout.timestamp = time_idx
    else:
        return None

    layout.open = int(open_idx)
    layout.high = int(high_idx)
    layout.low = int(low_idx)
    layout.close = int(close_idx)
    layout.volume = _find_column(keys, _VOLUME_KEYS)
    return layout


# ---------------------------------------------------------------------------
# parsing helpers
# ---------------------------------------------------------------------------


def _parse_timestamp(
    text: str,
    formats: Sequence[str],
    assume_tz: timezone,
    row: int,
    path: str,
) -> datetime:
    raw = text.strip().strip('"').strip()
    if not raw:
        raise DataError(f"{path}: row {row}: empty timestamp")

    # Epoch seconds / milliseconds. Unambiguous, so worth supporting.
    if raw.isdigit() and len(raw) in (10, 13):
        value = int(raw)
        seconds = value / 1000.0 if len(raw) == 13 else float(value)
        try:
            return datetime.fromtimestamp(seconds, tz=timezone.utc)
        except (OverflowError, OSError, ValueError) as exc:
            raise DataError(f"{path}: row {row}: bad epoch timestamp {raw!r}: {exc}") from exc

    iso_candidate = raw[:-1] + "+00:00" if raw.endswith("Z") else raw
    try:
        parsed = datetime.fromisoformat(iso_candidate)
    except ValueError:
        parsed = None
    if parsed is None:
        for fmt in formats:
            try:
                parsed = datetime.strptime(raw, fmt)
                break
            except ValueError:
                continue
    if parsed is None:
        hint = ""
        if "/" in raw or ("." in raw and not raw[:4].isdigit()):
            hint = (
                " - day-first and month-first dates are indistinguishable from one "
                "row, so pass timestamp_formats=[...] explicitly rather than let "
                "the loader guess"
            )
        raise DataError(f"{path}: row {row}: cannot parse timestamp {raw!r}{hint}")

    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=assume_tz).astimezone(timezone.utc)
    return parsed.astimezone(timezone.utc)


def _parse_price(text: str, field: str, row: int, path: str) -> float:
    raw = text.strip().strip('"').replace(",", "")
    if not raw:
        raise DataError(f"{path}: row {row}: empty {field}")
    try:
        value = float(raw)
    except ValueError as exc:
        raise DataError(f"{path}: row {row}: {field} {text!r} is not a number") from exc
    if not math.isfinite(value):
        raise DataError(
            f"{path}: row {row}: {field} is {raw!r}. A NaN or infinity in a price "
            "file is a broken export, not a missing value to be filled in."
        )
    return value


def _parse_volume(text: str, row: int, path: str) -> float:
    raw = text.strip().strip('"').replace(",", "")
    if not raw:
        return 0.0
    try:
        value = float(raw)
    except ValueError:
        return 0.0
    return value if math.isfinite(value) and value >= 0.0 else 0.0


# ---------------------------------------------------------------------------
# loaders
# ---------------------------------------------------------------------------


def load_csv(
    path: str,
    *,
    delimiter: Optional[str] = None,
    timestamp_formats: Optional[Sequence[str]] = None,
    assume_tz: timezone = timezone.utc,
    encoding: str = "utf-8-sig",
) -> list[Bar]:
    """Load OHLCV bars from a delimited text file.

    The header is detected case-insensitively and may name its columns in any
    order: a single `time` / `timestamp` / `datetime` column, or separate
    `date` and `time` columns, plus `open` / `high` / `low` / `close` and an
    optional `volume` (also `vol`, `tickvol`). MetaTrader's angle brackets
    (`<OPEN>`) are stripped. If no header is recognisable the file is read
    positionally as `timestamp, open, high, low, close[, volume]`.

    Parameters
    ----------
    delimiter:
        Forced field separator. When None it is sniffed from the first
        non-empty line among tab, comma, semicolon and pipe.
    timestamp_formats:
        `strptime` formats tried in order, before falling back to ISO-8601 and
        epoch seconds. Supply this for any file using D/M/Y or M/D/Y dates -
        the loader deliberately refuses to guess between them, because
        guessing wrong shifts every bar by up to eleven months and nothing
        downstream would notice.
    assume_tz:
        Timezone attributed to timestamps that carry no offset. Defaults to
        UTC. MetaTrader exports are in *broker server time*, which is usually
        UTC+2/+3 - if that matters to your session filters, pass it here
        rather than pretending the file was UTC.

    Raises
    ------
    DataError
        On an unparseable value, a non-finite price, `high < low`, an open or
        close outside the bar's range, or timestamps that do not strictly
        increase. The message always names the offending row number as
        counted in the file, so it can be opened and looked at.
    """
    formats = tuple(timestamp_formats) if timestamp_formats else _DEFAULT_TIMESTAMP_FORMATS
    bars: list[Bar] = []
    layout: Optional[_Layout] = None
    previous_ts: Optional[datetime] = None
    previous_row = 0

    with open(path, "r", encoding=encoding, newline="") as handle:
        raw_text = handle.read()

    if not raw_text.strip():
        raise DataError(f"{path}: file is empty")

    first_line = next(line for line in raw_text.splitlines() if line.strip())
    sep = delimiter if delimiter else _sniff_delimiter(first_line)

    reader = csv.reader(raw_text.splitlines(), delimiter=sep)
    for cells in reader:
        row = reader.line_num
        if not cells or not any(cell.strip() for cell in cells):
            continue
        if cells[0].lstrip().startswith("#"):
            continue

        if layout is None:
            layout = _layout_from_header(cells)
            if layout is not None:
                continue  # the header row itself carries no bar
            if _looks_like_header(cells):
                found = ", ".join(_normalise_key(cell) or "?" for cell in cells)
                raise DataError(
                    f"{path}: row {row}: this looks like a header but does not name "
                    f"the columns needed. Found: {found}. Expected a time column "
                    "(time / timestamp / datetime, or date + time) plus open, high, "
                    "low, close."
                )
            if len(cells) < 5:
                raise DataError(
                    f"{path}: row {row}: no recognisable header and only "
                    f"{len(cells)} columns; expected at least "
                    "timestamp, open, high, low, close"
                )
            layout = _positional_layout()

        if len(cells) <= layout.max_index:
            raise DataError(
                f"{path}: row {row}: {len(cells)} columns but column "
                f"{layout.max_index + 1} is needed"
            )

        if layout.timestamp is not None:
            ts = _parse_timestamp(cells[layout.timestamp], formats, assume_tz, row, path)
        else:
            joined = f"{cells[layout.date].strip()} {cells[layout.time].strip()}"
            ts = _parse_timestamp(joined, formats, assume_tz, row, path)

        open_ = _parse_price(cells[layout.open], "open", row, path)
        high = _parse_price(cells[layout.high], "high", row, path)
        low = _parse_price(cells[layout.low], "low", row, path)
        close = _parse_price(cells[layout.close], "close", row, path)
        volume = _parse_volume(cells[layout.volume], row, path) if layout.volume is not None else 0.0

        if high < low:
            raise DataError(
                f"{path}: row {row}: high {high} is below low {low}. The file is "
                "wrong; the backtest built on it would be wrong more quietly."
            )
        if previous_ts is not None and ts <= previous_ts:
            order = "duplicates" if ts == previous_ts else "goes backwards from"
            raise DataError(
                f"{path}: row {row}: timestamp {ts.isoformat()} {order} row "
                f"{previous_row} ({previous_ts.isoformat()}). Bars must strictly "
                "increase; which of the two rows is the real bar is not something "
                "this loader can decide for you."
            )

        try:
            bars.append(Bar(ts=ts, open=open_, high=high, low=low, close=close, volume=volume))
        except ValueError as exc:
            raise DataError(f"{path}: row {row}: {exc}") from exc

        previous_ts, previous_row = ts, row

    if not bars:
        raise DataError(f"{path}: no data rows found")
    return bars


def load_mt5_csv(path: str) -> list[Bar]:
    """Load a MetaTrader 5 "Export Bars" file.

    The format is tab-separated with the header

        <DATE>	<TIME>	<OPEN>	<HIGH>	<LOW>	<CLOSE>	<TICKVOL>	<VOL>	<SPREAD>

    and dates written `YYYY.MM.DD`. `<TICKVOL>` is used as volume: it counts
    price changes, not traded size, and on a retail FX feed real volume is
    almost always zero. Do not build a strategy on it without knowing that.

    **These are BID prices.** MT5 charts and exports are bid-side, so the ask
    is bid + spread and the mid is half a spread above every printed close.
    Pass `CostConfig(price_is_mid=False)` when backtesting this data, or the
    engine will add half a spread that the file has already taken off and you
    will pay the spread one-and-a-half times - or, worse, tune a strategy
    against a cost that is not the one your broker charges.

    Two further honesty notes. The `<SPREAD>` column records the spread in
    points at each bar and is *not* read here: the cost model uses the spread
    you configured, deliberately, so that a backtest cannot flatter itself
    with the tightest quotes in the file. And MT5 timestamps are in **broker
    server time**, typically UTC+2 in winter and UTC+3 in summer; they are
    tagged UTC here without conversion. That is fine for measuring returns and
    wrong for anything keyed to a session or a news hour - for those, reload
    with `load_csv(path, delimiter="\\t", assume_tz=...)`.
    """
    return load_csv(
        path,
        delimiter="\t",
        timestamp_formats=("%Y.%m.%d %H:%M:%S", "%Y.%m.%d %H:%M", "%Y.%m.%d"),
    )


def save_csv(bars: Sequence[Bar], path: str) -> None:
    """Write bars to a comma-separated file that `load_csv` reads back exactly.

    Timestamps are ISO-8601 with an explicit `+00:00`, and prices are written
    with `repr`, which is the shortest string that round-trips to the same
    float. Saving a synthetic series and reloading it must not change a single
    price, or the reproducibility of everything downstream is theatre.
    """
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["time", "open", "high", "low", "close", "volume"])
        for bar in bars:
            writer.writerow(
                [
                    bar.ts.astimezone(timezone.utc).isoformat(),
                    repr(bar.open),
                    repr(bar.high),
                    repr(bar.low),
                    repr(bar.close),
                    repr(bar.volume),
                ]
            )


# ---------------------------------------------------------------------------
# synthetic series
# ---------------------------------------------------------------------------

# Six sub-steps per bar: enough that the high and low look like a real intrabar
# excursion rather than max(open, close), cheap enough that a million bars is
# still a couple of seconds. It is still a model of the intrabar path, and the
# engine's pessimistic tie-breaking exists because no such model is the truth.
_SUB_STEPS = 6

_DEFAULT_START_TS = datetime(2020, 1, 1, 0, 0, tzinfo=timezone.utc)


def _next_ts(ts: datetime, minutes: int) -> datetime:
    """Advance one bar, skipping Saturday and Sunday.

    A forex week is five days long. Skipping the weekend keeps the generated
    series consistent with the 260 * 24 = 6240 hourly bars a year that the
    annualisation elsewhere in the package assumes.
    """
    nxt = ts + timedelta(minutes=minutes)
    while nxt.weekday() >= 5:
        nxt += timedelta(days=1)
    return nxt


def _bar_from_return(
    ts: datetime,
    open_price: float,
    total_log_return: float,
    sigma: float,
    rng: random.Random,
) -> Bar:
    """Build one bar whose close is fixed and whose path is a Brownian bridge.

    The sub-path is generated, then de-trended and re-trended onto the bar's
    actual log return, so the endpoint is exactly `open * exp(total)` while the
    excursion in between is random. High and low are taken from that path, so
    `low <= open, close <= high` holds by construction rather than by a
    post-hoc clamp - clamping is how synthetic data quietly acquires an
    intrabar bias that a stop-loss strategy then "discovers".
    """
    steps = _SUB_STEPS
    step_sd = sigma / math.sqrt(steps)
    noise = [rng.gauss(0.0, step_sd) for _ in range(steps)]
    noise_total = sum(noise)

    log_open = math.log(open_price)
    path: list[float] = []
    cumulative = 0.0
    for index, increment in enumerate(noise, start=1):
        cumulative += increment
        fraction = index / steps
        bridged = cumulative - fraction * noise_total + fraction * total_log_return
        path.append(math.exp(log_open + bridged))

    close = path[-1]
    high = max(path + [open_price])
    low = min(path + [open_price])
    # Fabricated tick volume, loosely scaled by how far the bar travelled. It
    # carries no information beyond |return| and must not be modelled on.
    volume = float(round(400 + 900 * abs(total_log_return) / sigma)) if sigma > 0 else 0.0
    return Bar(ts=ts, open=open_price, high=high, low=low, close=close, volume=volume)


def _generate(
    n: int,
    *,
    seed: int,
    start: float,
    annual_vol: float,
    bars_per_year: float,
    start_ts: Optional[datetime],
    minutes: int,
    autocorr: float,
) -> list[Bar]:
    if n < 0:
        raise ValueError("n must be non-negative")
    if start <= 0.0:
        raise ValueError("start price must be positive")
    if annual_vol < 0.0:
        raise ValueError("annual_vol must be non-negative")
    if bars_per_year <= 0.0:
        raise ValueError("bars_per_year must be positive")
    if minutes <= 0:
        raise ValueError("minutes must be positive")
    if not -0.99 <= autocorr <= 0.99:
        raise ValueError("autocorr must lie in [-0.99, 0.99]")

    rng = random.Random(seed)
    sigma = annual_vol / math.sqrt(bars_per_year)
    # An AR(1) with this innovation size has stationary standard deviation
    # exactly `sigma`, so changing the autocorrelation changes the structure of
    # the series without changing its volatility. Otherwise a trending series
    # would also be a noisier one and the comparison would prove nothing.
    innovation_sd = sigma * math.sqrt(1.0 - autocorr * autocorr)

    ts = start_ts or _DEFAULT_START_TS
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    ts = ts.astimezone(timezone.utc)
    while ts.weekday() >= 5:
        ts += timedelta(days=1)

    bars: list[Bar] = []
    price = float(start)
    previous_return = 0.0
    for _ in range(n):
        log_return = autocorr * previous_return + rng.gauss(0.0, innovation_sd)
        previous_return = log_return
        bar = _bar_from_return(ts, price, log_return, sigma, rng)
        bars.append(bar)
        price = bar.close
        ts = _next_ts(ts, minutes)
    return bars


def synthetic_random_walk(
    n: int,
    *,
    seed: int = 0,
    start: float = 1.1000,
    annual_vol: float = 0.08,
    bars_per_year: float = 6240,
    start_ts: Optional[datetime] = None,
    minutes: int = 60,
) -> list[Bar]:
    """Geometric Brownian motion with no drift and no autocorrelation.

    **This series has zero exploitable edge, by construction.** Log returns are
    independent draws from a zero-mean normal distribution: the next bar is
    unrelated to every bar before it, there is no trend to follow, no level to
    revert to, and no expected return to capture in either direction. Nothing
    in it can be predicted from its own history, because nothing in it was
    generated from its own history.

    That makes it the single most useful diagnostic in the package. Whatever a
    strategy scores here is the score of luck - and it will score something,
    because a few hundred random bars will always contain runs, breakouts and
    clean reversals to be found after the fact. Run the strategy over several
    seeds, look at the distribution of Sharpe ratios and equity curves it
    produces on data that certainly has no edge, and treat that distribution as
    the bar a result on real data must clear. Most do not clear it. A backtest
    that looks impressive but sits inside the range noise produces has
    demonstrated nothing except that the fitting worked.

    The default `bars_per_year=6240` is 260 five-day weeks of 24 hourly bars,
    matching the weekend-skipping calendar the timestamps use, so `annual_vol`
    means what it says.

    Being noise, this data is also *kind*: no gaps, no news spikes, no spread
    widening, constant volatility. Real data is worse in every one of those
    respects. Beating this is necessary, never sufficient.
    """
    return _generate(
        n,
        seed=seed,
        start=start,
        annual_vol=annual_vol,
        bars_per_year=bars_per_year,
        start_ts=start_ts,
        minutes=minutes,
        autocorr=0.0,
    )


def synthetic_trending(n: int, *, seed: int = 0, **kw) -> list[Bar]:
    """A random walk with a mild, deliberate momentum component.

    Log returns follow an AR(1): each bar's return is `autocorr` times the
    previous bar's, plus fresh noise. That is a real, persistent edge - small,
    but present in every bar and not an artefact of the sample - so a trend
    strategy that cannot find it has a bug, not a shortage of alpha.

    This is the positive control. `synthetic_random_walk` proves a harness does
    not manufacture edges that are absent; this proves the same harness can
    detect one that is present. A test suite with only the first can pass by
    never trading.

    Accepts every keyword of `synthetic_random_walk`, plus:

    autocorr:
        Lag-1 autocorrelation of log returns, default 0.15. Real markets show
        something between roughly -0.05 and +0.05 at this horizon, and only
        intermittently. 0.15 is chosen to be findable, not to be realistic;
        an edge this stable does not exist and a strategy tuned on it will be
        disappointed by history.
    """
    autocorr = float(kw.pop("autocorr", 0.15))
    defaults = {
        "start": 1.1000,
        "annual_vol": 0.08,
        "bars_per_year": 6240,
        "start_ts": None,
        "minutes": 60,
    }
    unknown = set(kw) - set(defaults)
    if unknown:
        raise TypeError(f"unexpected keyword arguments: {', '.join(sorted(unknown))}")
    defaults.update(kw)
    return _generate(n, seed=seed, autocorr=autocorr, **defaults)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# reshaping and inspection
# ---------------------------------------------------------------------------


def _median_gap_seconds(bars: Sequence[Bar]) -> Optional[float]:
    """Median spacing between bars. Median, not mean, because the weekend hole
    drags the mean upward and would report hourly data as three-hourly."""
    if len(bars) < 2:
        return None
    gaps = [
        (bars[i].ts - bars[i - 1].ts).total_seconds()
        for i in range(1, len(bars))
        if (bars[i].ts - bars[i - 1].ts).total_seconds() > 0
    ]
    return statistics.median(gaps) if gaps else None


def resample(bars: list[Bar], minutes: int) -> list[Bar]:
    """Aggregate bars up to a coarser timeframe.

    Buckets are aligned to the UTC epoch, so a 60-minute resample starts on the
    hour regardless of where the data starts. Open is the first bar's open,
    close the last bar's close, high and low the extremes, volume the sum.

    An incomplete trailing bucket is dropped. A half-formed final bar is the
    classic way a backtest acquires a look-ahead-shaped artefact at the very
    end of the sample, right where an author is most likely to be looking. The
    bucket is judged complete only if the source data reaches its end.

    Note what resampling cannot recover: the order of events inside the new
    bar. Aggregating twelve five-minute bars into an hour throws away the path
    that decided whether a stop or a target was hit first, which is exactly the
    information the engine's pessimistic tie-breaking is compensating for.
    Backtest on the finest data you have; resample to look at it, not to trade.
    """
    if minutes <= 0:
        raise ValueError("minutes must be positive")
    if not bars:
        return []

    base = _median_gap_seconds(bars)
    target_seconds = float(minutes) * 60.0
    if base is not None and target_seconds < base:
        raise ValueError(
            f"cannot resample {base / 60.0:.0f}-minute bars to {minutes} minutes: "
            "resampling only coarsens. Upsampling would invent prices."
        )

    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    step = timedelta(minutes=minutes)
    out: list[Bar] = []
    bucket_start: Optional[datetime] = None
    group: list[Bar] = []

    def flush(start: datetime, members: list[Bar]) -> None:
        out.append(
            Bar(
                ts=start,
                open=members[0].open,
                high=max(b.high for b in members),
                low=min(b.low for b in members),
                close=members[-1].close,
                volume=sum(b.volume for b in members),
            )
        )

    for bar in bars:
        start = epoch + ((bar.ts - epoch) // step) * step
        if bucket_start is None:
            bucket_start = start
        elif start != bucket_start:
            flush(bucket_start, group)
            bucket_start, group = start, []
        group.append(bar)

    if bucket_start is not None and group:
        # The final bucket only counts if the data ran to its far edge: the
        # last source bar plus one of its own periods must reach the boundary.
        reach = group[-1].ts + timedelta(seconds=base or target_seconds)
        if reach >= bucket_start + step:
            flush(bucket_start, group)
    return out


def summary(bars: Sequence[Bar]) -> str:
    """Human-readable description of a bar series: span, count, gaps.

    Gaps are reported rather than filled. In FX most of them are the weekend
    and are harmless; the rest are missing data, and missing data in the middle
    of a backtest is a silent hole in the sample that every statistic then
    reports as if it were continuous history.
    """
    if not bars:
        return "No bars."

    first, last = bars[0], bars[-1]
    span_days = (last.ts - first.ts).total_seconds() / 86400.0
    base = _median_gap_seconds(bars)
    lines = [
        f"Bars      : {len(bars):,}",
        f"Span      : {first.ts.strftime('%Y-%m-%d %H:%M')} UTC -> "
        f"{last.ts.strftime('%Y-%m-%d %H:%M')} UTC ({span_days:,.1f} days)",
    ]
    if base is not None:
        lines.append(f"Interval  : {_describe_interval(base)} (median gap)")

    if base:
        threshold = base * 1.5
        gaps = [
            (bars[i].ts - bars[i - 1].ts, bars[i - 1].ts)
            for i in range(1, len(bars))
            if (bars[i].ts - bars[i - 1].ts).total_seconds() > threshold
        ]
        weekend = sum(1 for delta, start in gaps if start.weekday() >= 4 and delta.days <= 3)
        if gaps:
            widest, widest_at = max(gaps, key=lambda item: item[0])
            lines.append(
                f"Gaps      : {len(gaps):,} longer than "
                f"{_describe_interval(threshold)} ({weekend:,} look like weekends); "
                f"largest {_describe_interval(widest.total_seconds())} after "
                f"{widest_at.strftime('%Y-%m-%d %H:%M')} UTC"
            )
        else:
            lines.append("Gaps      : none longer than 1.5x the median interval")

    closes = [bar.close for bar in bars]
    lines.append(
        f"Prices    : low {min(bar.low for bar in bars):.5f}  "
        f"high {max(bar.high for bar in bars):.5f}  "
        f"first close {closes[0]:.5f}  last close {closes[-1]:.5f}"
    )
    volumes = [bar.volume for bar in bars]
    if any(volumes):
        lines.append(f"Volume    : total {sum(volumes):,.0f}  median {statistics.median(volumes):,.0f}")
    else:
        lines.append("Volume    : absent (all zero)")
    return "\n".join(lines)


def _describe_interval(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes = seconds / 60.0
    if minutes < 60:
        return f"{minutes:.0f}m"
    hours = minutes / 60.0
    if hours < 24:
        return f"{hours:.0f}h"
    return f"{hours / 24.0:.1f}d"
