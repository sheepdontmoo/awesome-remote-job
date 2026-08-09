"""Donchian channel breakout - the Turtle-style trend follower.

This strategy exists in the package for two reasons, and neither of them is
that it makes money.

The first is that it is the honest reference implementation of the oldest
published mechanical edge in futures and FX. Breakout systems have a long
out-of-sample record, and they have it precisely because they are unpleasant
to trade: the win rate sits somewhere around 30-40%, the losing streaks run
into the dozens, and the profit comes from a handful of trades a year that
are allowed to run. Any backtest of this idea that reports a high win rate
has a bug in it.

The second is that it is a stress test for the engine. A breakout system
gaps through its stops, holds positions across rollovers for weeks, and
trades rarely enough that costs per trade are visible rather than buried.
If the execution model is wrong anywhere, this is where it shows.

The entry channel deliberately excludes the current bar (see the long note in
`indicators.donchian`). A breakout measured against a level that includes the
bar doing the breaking is not a signal, it is a definition.
"""

from __future__ import annotations

from typing import Optional, Tuple

from ..indicators import atr, donchian, last
from ..strategy import BarWindow, Context, Strategy
from ..types import Direction, OrderIntent

# Wilder smoothing has infinite memory, so an exact ATR would need the whole
# series on every bar and turn a backtest into an O(n^2) crawl. Recomputing
# over a bounded tail instead leaves the seed a residual weight of
# (1 - 1/period)^(10*period) ~ e^-10 ~ 4.5e-5. Measured against the
# full-history ATR over 6000 synthetic EURUSD-like bars, the worst-case
# relative error is 2.0e-5 at period=14 and 1.1e-5 at period=50 - about
# 0.003 of a point on a 5-digit feed, against stops hundreds of points wide.
# It is an approximation, and this is the measured size of it.
_ATR_TAIL_MULTIPLE = 10


def _atr_now(bars: BarWindow, period: int) -> Optional[float]:
    """Current ATR, computed over a bounded tail of closed bars."""
    tail = bars[-(period * _ATR_TAIL_MULTIPLE + 1):]
    if len(tail) < period + 1:
        return None
    return last(atr(tail, period))


def _channel_now(bars: BarWindow, period: int) -> Optional[Tuple[float, float]]:
    """Donchian `(upper, lower)` over the `period` bars ending at the previous
    bar. The tail is `period + 1` long because the channel skips bar `i`."""
    tail = bars[-(period + 1):]
    if len(tail) < period + 1:
        return None
    return last(donchian(tail, period))


class DonchianBreakout(Strategy):
    """Buy new highs, sell new lows, exit on the shorter channel in reverse.

    Entry: the close of the bar that just finished is above the highest high
    of the previous `entry_lookback` bars (mirrored for shorts). The order
    fills at the next open, so the actual entry is worse than the signal price
    by a gap plus the spread plus slippage - which is exactly the point.

    Stop: `atr_stop_mult` ATRs from the signal close. The engine sizes the
    position from that distance, so a wider ATR buys fewer lots and the cash
    at risk stays roughly constant. Note that the stop is placed against the
    close we can see, not the open we will get; if the market gaps past it
    overnight the engine refuses the trade rather than pretending the fill was
    good. That refusal is a warning in the result, not a silent skip.

    Exit: the opposite `exit_lookback` channel, either as a signal on the
    close or, with `trail=True`, as a stop that ratchets up behind a long and
    down behind a short and never loosens. The trailing version exits sooner
    and gives back less of an open profit; it also gets shaken out of trends
    that the signal version would have held. Which is better is a property of
    the data, not of the idea, so both are available and neither is default
    in the sense of being correct.

    `long_only=True` disables shorts, which is the honest configuration when
    the instrument has a structural upward drift and the swap on a short
    position would have eaten the difference anyway.
    """

    name = "donchian_breakout"

    def __init__(
        self,
        entry_lookback: int = 20,
        exit_lookback: int = 10,
        atr_period: int = 14,
        atr_stop_mult: float = 2.0,
        trail: bool = True,
        long_only: bool = False,
    ) -> None:
        entry_lookback = int(entry_lookback)
        exit_lookback = int(exit_lookback)
        atr_period = int(atr_period)
        if entry_lookback < 1 or exit_lookback < 1 or atr_period < 1:
            raise ValueError("lookbacks and atr_period must be >= 1")
        if atr_stop_mult <= 0.0:
            raise ValueError("atr_stop_mult must be positive; a zero-width stop cannot be sized")
        super().__init__(
            entry_lookback=entry_lookback,
            exit_lookback=exit_lookback,
            atr_period=atr_period,
            atr_stop_mult=float(atr_stop_mult),
            trail=bool(trail),
            long_only=bool(long_only),
        )

    @property
    def warmup(self) -> int:
        # +2 covers the extra bar the exclusive Donchian window needs and the
        # bar true range loses at the start of the series.
        return max(self.entry_lookback, self.exit_lookback, self.atr_period) + 2

    def on_bar(self, ctx: Context) -> Optional[OrderIntent]:
        exit_channel = _channel_now(ctx.bars, self.exit_lookback)
        if ctx.position is not None:
            return self._manage(ctx, exit_channel)

        entry_channel = _channel_now(ctx.bars, self.entry_lookback)
        stop_distance = self._stop_distance(ctx)
        if entry_channel is None or stop_distance is None:
            return None
        return self._enter(ctx, entry_channel, stop_distance)

    # -- entries ----------------------------------------------------------

    def _stop_distance(self, ctx: Context) -> Optional[float]:
        value = _atr_now(ctx.bars, self.atr_period)
        if value is None or value <= 0.0:
            return None
        return value * self.atr_stop_mult

    def _enter(
        self, ctx: Context, channel: Tuple[float, float], stop_distance: float
    ) -> Optional[OrderIntent]:
        upper, lower = channel
        close = ctx.price
        if close > upper:
            return OrderIntent(
                direction=Direction.LONG,
                stop_loss=close - stop_distance,
                tag="donchian_long",
            )
        if close < lower and not self.long_only:
            return OrderIntent(
                direction=Direction.SHORT,
                stop_loss=close + stop_distance,
                tag="donchian_short",
            )
        return None

    # -- management -------------------------------------------------------

    def _manage(
        self, ctx: Context, channel: Optional[Tuple[float, float]]
    ) -> Optional[OrderIntent]:
        """Exit on the opposite channel break, otherwise ratchet the stop."""
        if channel is None:
            return None
        upper, lower = channel
        position = ctx.position
        close = ctx.price

        if position.is_long and close < lower:
            return OrderIntent(direction=Direction.FLAT, tag="donchian_exit_long")
        if position.is_short and close > upper:
            return OrderIntent(direction=Direction.FLAT, tag="donchian_exit_short")

        if self.trail:
            self._ratchet(ctx, upper, lower)
        return None

    def _ratchet(self, ctx: Context, upper: float, lower: float) -> None:
        """Move the stop towards the market, never away from it.

        A trailing stop that is allowed to widen is not a trailing stop, it is
        a strategy quietly increasing its risk after the trade has gone
        against it. The comparison below is the only thing preventing that.
        """
        position = ctx.position
        close = ctx.price
        current = position.stop_loss
        if position.is_long:
            candidate = lower
            if candidate >= close:
                return  # already through the level; the exit test owns this bar
            if current is None or candidate > current:
                ctx.move_stop(candidate)
        else:
            candidate = upper
            if candidate <= close:
                return
            if current is None or candidate < current:
                ctx.move_stop(candidate)
