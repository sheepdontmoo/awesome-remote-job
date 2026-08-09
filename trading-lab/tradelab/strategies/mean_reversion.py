"""Z-score mean reversion - the strategy that flatters itself the hardest.

This module exists so the package contains a worked example of the failure
mode it warns about everywhere else, rather than only describing it.

Mean reversion is structurally asymmetric. Fading a stretched price wins
often and loses big: most excursions revert, so the hit rate is high, and the
ones that do not revert keep going in the direction you are leaning against
until they take back many winners at once. The equity curve that results is
close to a straight line - smooth, high Sharpe, tiny drawdowns - right up
until it is not, and the bar on which it stops being a straight line is not
predictable from the part of the curve you have already seen.

The consequence for anyone reading a report: **the win rate of a mean
reversion strategy is the most misleading statistic in retail trading.** An
80% win rate means nothing on its own, because it says nothing about the size
of the 20%. Read profit factor, the worst single trade, the worst R-multiple,
and the drawdown distribution from `validation.monte_carlo_drawdown` instead.
If the strategy has no stop at all its win rate approaches 100% and its
expectancy approaches ruin; that is why `atr_stop_mult` here is not optional
and why the stop is deliberately wide rather than absent.

The trend filter is not decoration either. Fading a move that runs with the
prevailing trend is fading the trend, and the trend is the thing most likely
to keep going. Only counter-trend stretches are faded here.
"""

from __future__ import annotations

from typing import Optional

from ..indicators import atr, last, sma, zscore
from ..strategy import BarWindow, Context, Strategy
from ..types import Direction, OrderIntent

# See the measured note in `donchian.py`: Wilder smoothing never fully forgets
# its seed, so the ATR is recomputed over a bounded tail. Worst-case error
# against a full-history ATR is ~2e-5 relative, in exchange for a backtest
# that runs in linear time rather than quadratic.
_ATR_TAIL_MULTIPLE = 10


def _atr_now(bars: BarWindow, period: int) -> Optional[float]:
    """Current ATR, computed over a bounded tail of closed bars."""
    tail = bars[-(period * _ATR_TAIL_MULTIPLE + 1):]
    if len(tail) < period + 1:
        return None
    return last(atr(tail, period))


class ZScoreReversion(Strategy):
    """Short stretched closes, buy crushed ones, but only against the trend.

    Signal: the close's z-score over the last `lookback` closes. The window
    includes the current close (see `indicators.zscore`), which bounds the
    score at roughly `sqrt(lookback - 1)` - so `entry_z=2.0` on a 20-bar
    window is a genuinely uncommon reading, not the 2-sigma of a normal
    distribution. Do not read these numbers as probabilities.

    Filter: `trend_filter` is the period of a slow SMA. A short is only taken
    when the close is *below* that average, and a long only when it is above.
    The stretch being faded is then a move against the prevailing direction,
    which is the only kind worth fading. Set `trend_filter=0` to disable it
    and watch what happens to the tail of the trade distribution.

    Exits, in the order the engine applies them:

    * the ATR stop, checked against the bar's range before anything else, and
      wide (`atr_stop_mult=3.0` by default) because a mean reversion entry is
      by construction placed into an ongoing adverse move;
    * `abs(z) < exit_z`, the reversion actually happening, which is where the
      profit comes from;
    * `max_bars`, a time stop. A reversion trade that has not reverted within
      a couple of days was not a reversion trade; holding it is hoping, and
      hoping across rollovers also pays swap every night.

    Entries are skipped while a position is open. This strategy never adds to
    a loser, which is the single most common way the idea is destroyed in
    practice.
    """

    name = "zscore_reversion"

    def __init__(
        self,
        lookback: int = 20,
        entry_z: float = 2.0,
        exit_z: float = 0.5,
        atr_period: int = 14,
        atr_stop_mult: float = 3.0,
        max_bars: int = 48,
        trend_filter: int = 200,
    ) -> None:
        lookback = int(lookback)
        atr_period = int(atr_period)
        max_bars = int(max_bars)
        trend_filter = int(trend_filter)
        if lookback < 2:
            raise ValueError("lookback must be >= 2; a z-score needs dispersion to measure")
        if atr_period < 1:
            raise ValueError("atr_period must be >= 1")
        if entry_z <= 0.0 or exit_z < 0.0:
            raise ValueError("entry_z must be positive and exit_z non-negative")
        if exit_z >= entry_z:
            raise ValueError("exit_z must be below entry_z, or every entry exits immediately")
        if atr_stop_mult <= 0.0:
            raise ValueError("atr_stop_mult must be positive; an unstopped fade is not a strategy")
        if max_bars < 1:
            raise ValueError("max_bars must be >= 1")
        if trend_filter < 0:
            raise ValueError("trend_filter must be >= 0 (0 disables the filter)")
        super().__init__(
            lookback=lookback,
            entry_z=float(entry_z),
            exit_z=float(exit_z),
            atr_period=atr_period,
            atr_stop_mult=float(atr_stop_mult),
            max_bars=max_bars,
            trend_filter=trend_filter,
        )

    @property
    def warmup(self) -> int:
        # +2 covers the bar true range loses at the start of the series and
        # leaves one spare, so no indicator is asked for its very first value.
        return max(self.lookback, self.trend_filter, self.atr_period + 1) + 2

    def on_bar(self, ctx: Context) -> Optional[OrderIntent]:
        score = self._zscore_now(ctx.bars)
        if score is None:
            return None
        if ctx.position is not None:
            return self._manage(ctx, score)
        return self._enter(ctx, score)

    # -- signal -----------------------------------------------------------

    def _zscore_now(self, bars: BarWindow) -> Optional[float]:
        closes = bars.closes(self.lookback)
        if len(closes) < self.lookback:
            return None
        return last(zscore(closes, self.lookback))

    def _trend_allows(self, ctx: Context, direction: Direction) -> bool:
        """True when the fade leans against the slow average, not with it."""
        if self.trend_filter <= 0:
            return True
        closes = ctx.bars.closes(self.trend_filter)
        if len(closes) < self.trend_filter:
            return False
        average = last(sma(closes, self.trend_filter))
        if average is None:
            return False
        if direction == Direction.LONG:
            return ctx.price > average
        return ctx.price < average

    # -- entries ----------------------------------------------------------

    def _enter(self, ctx: Context, score: float) -> Optional[OrderIntent]:
        if -self.entry_z <= score <= self.entry_z:
            return None
        direction = Direction.SHORT if score > 0.0 else Direction.LONG
        if not self._trend_allows(ctx, direction):
            return None

        value = _atr_now(ctx.bars, self.atr_period)
        if value is None or value <= 0.0:
            return None
        distance = value * self.atr_stop_mult
        close = ctx.price
        if direction == Direction.LONG:
            return OrderIntent(
                direction=Direction.LONG,
                stop_loss=close - distance,
                tag="zscore_long",
            )
        return OrderIntent(
            direction=Direction.SHORT,
            stop_loss=close + distance,
            tag="zscore_short",
        )

    # -- management -------------------------------------------------------

    def _manage(self, ctx: Context, score: float) -> Optional[OrderIntent]:
        position = ctx.position
        if abs(score) < self.exit_z:
            return OrderIntent(direction=Direction.FLAT, tag="zscore_reverted")
        if position.bars_held >= self.max_bars:
            ctx.note(f"time stop after {position.bars_held} bars without reversion")
            return OrderIntent(direction=Direction.FLAT, tag="zscore_time_stop")
        return None
