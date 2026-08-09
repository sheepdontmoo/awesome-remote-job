"""Strategy interface and the read-only view of the world a strategy gets.

The design goal of this module is a single guarantee:

    **A strategy physically cannot see the future.**

Look-ahead bias is not a discipline problem you solve by being careful. It is
a design problem you solve by making the future unreachable. `BarWindow`
below hands a strategy a window that ends at the current bar and raises if
anything reaches past it, so a look-ahead bug becomes a loud crash instead of
a beautiful equity curve.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Iterator, Optional, Sequence

from .types import Bar, Direction, Instrument, OrderIntent


class LookAheadError(IndexError):
    """Raised when strategy code reaches for a bar that has not happened yet."""


class BarWindow(Sequence[Bar]):
    """A read-only view of bars[0 .. end], where `end` is the current bar.

    Indexing follows Python conventions with one addition: `window[-1]` is the
    bar that just closed, `window[-2]` the one before it. Any positive index
    beyond `end`, or any attempt to slice past it, raises `LookAheadError`
    rather than silently returning tomorrow's price.

    The guard covers every public access path - indexing, slicing, iteration,
    `len`, and the `Sequence` mixins built on them. It is not a sandbox: the
    full series is held in `_bars`, and code that reaches for that private
    attribute gets the future with no complaint. Copying the visible prefix on
    every bar would close that hole and make the class quadratic, which for a
    package whose validation suite runs thousands of backtests is the worse
    trade. So: accidental look-ahead crashes, deliberate look-ahead does not.
    """

    __slots__ = ("_bars", "_end")

    def __init__(self, bars: list[Bar], end: int) -> None:
        self._bars = bars
        self._end = end

    def __len__(self) -> int:
        return self._end + 1

    def __iter__(self) -> Iterator[Bar]:
        for i in range(self._end + 1):
            yield self._bars[i]

    def __getitem__(self, key):  # type: ignore[override]
        n = self._end + 1
        if isinstance(key, slice):
            # The raw stop has to be tested *before* `indices`, which silently
            # clamps it to n. Clamping is why `window[0:n + 5]` would otherwise
            # return a short window instead of announcing the look-ahead.
            if key.stop is not None and key.stop > n:
                raise LookAheadError(f"slice reaches bar {key.stop} but only {n} bars have closed")
            start, stop, step = key.indices(n)
            return [self._bars[i] for i in range(start, stop, step)]
        idx = int(key)
        if idx < 0:
            idx += n
        if idx < 0:
            raise IndexError("index out of range (before start of data)")
        if idx >= n:
            raise LookAheadError(
                f"strategy asked for bar {idx} but only {n} bars have closed - "
                "this is look-ahead bias, not a missing bar"
            )
        return self._bars[idx]

    # -- convenience accessors, all strictly causal -----------------------

    @property
    def current(self) -> Bar:
        """The bar that just closed. Decisions are made on this bar and
        executed at the open of the next one."""
        return self._bars[self._end]

    def closes(self, n: Optional[int] = None) -> list[float]:
        """Last `n` closes, oldest first. All of them if `n` is None."""
        return [b.close for b in self._tail(n)]

    def opens(self, n: Optional[int] = None) -> list[float]:
        return [b.open for b in self._tail(n)]

    def highs(self, n: Optional[int] = None) -> list[float]:
        return [b.high for b in self._tail(n)]

    def lows(self, n: Optional[int] = None) -> list[float]:
        return [b.low for b in self._tail(n)]

    def _tail(self, n: Optional[int]) -> list[Bar]:
        if n is None:
            return [self._bars[i] for i in range(self._end + 1)]
        if n <= 0:
            return []
        start = max(0, self._end + 1 - n)
        return [self._bars[i] for i in range(start, self._end + 1)]


class PositionView:
    """Read-only snapshot of the open position, or None when flat."""

    __slots__ = (
        "direction", "lots", "entry_price", "entry_ts", "stop_loss",
        "take_profit", "bars_held", "unrealized_pnl", "mae", "mfe",
    )

    def __init__(
        self,
        direction: Direction,
        lots: float,
        entry_price: float,
        entry_ts: datetime,
        stop_loss: Optional[float],
        take_profit: Optional[float],
        bars_held: int,
        unrealized_pnl: float,
        mae: float,
        mfe: float,
    ) -> None:
        self.direction = direction
        self.lots = lots
        self.entry_price = entry_price
        self.entry_ts = entry_ts
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.bars_held = bars_held
        self.unrealized_pnl = unrealized_pnl
        self.mae = mae
        self.mfe = mfe

    @property
    def is_long(self) -> bool:
        return self.direction == Direction.LONG

    @property
    def is_short(self) -> bool:
        return self.direction == Direction.SHORT


class Context:
    """What a strategy is allowed to know at the close of the current bar."""

    __slots__ = ("bars", "position", "equity", "balance", "instrument", "_stop_move", "_notes")

    def __init__(
        self,
        bars: BarWindow,
        position: Optional[PositionView],
        equity: float,
        balance: float,
        instrument: Instrument,
    ) -> None:
        self.bars = bars
        self.position = position
        self.equity = equity
        self.balance = balance
        self.instrument = instrument
        self._stop_move: Optional[float] = None
        self._notes: list[str] = []

    @property
    def now(self) -> datetime:
        return self.bars.current.ts

    @property
    def price(self) -> float:
        """Mid close of the bar that just closed."""
        return self.bars.current.close

    @property
    def is_flat(self) -> bool:
        return self.position is None

    def move_stop(self, new_stop: float) -> None:
        """Request a protective-stop move, applied before the next bar.

        This is not look-ahead: the decision uses only closed-bar information
        and takes effect on subsequent bars, exactly as a real trailing stop
        sitting on the broker's server would.
        """
        self._stop_move = float(new_stop)

    def note(self, message: str) -> None:
        """Attach a diagnostic note to this bar. Shown in verbose reports."""
        self._notes.append(message)


class Strategy(ABC):
    """Base class for every strategy.

    Subclasses implement `on_bar` and declare `warmup` - the number of closed
    bars they need before their indicators mean anything. The engine will not
    call `on_bar` until that many bars exist, which stops the classic bug of
    trading on a moving average computed from three data points.
    """

    name: str = "unnamed"

    def __init__(self, **params) -> None:
        self.params = dict(params)
        for key, value in params.items():
            setattr(self, key, value)

    @property
    def warmup(self) -> int:
        """Bars of history required before the first decision. Override."""
        return 0

    def on_start(self, instrument: Instrument) -> None:
        """Called once before the first bar. Optional."""

    def on_finish(self) -> None:
        """Called once after the last bar. Optional."""

    @abstractmethod
    def on_bar(self, ctx: Context) -> Optional[OrderIntent]:
        """Decide what to do at the close of `ctx.bars.current`.

        Return an `OrderIntent` to act, or None to do nothing. Whatever is
        returned executes at the **open of the next bar**, with spread and
        slippage applied against you.
        """

    def describe(self) -> str:
        if not self.params:
            return self.name
        joined = ", ".join(f"{k}={v}" for k, v in sorted(self.params.items()))
        return f"{self.name}({joined})"
