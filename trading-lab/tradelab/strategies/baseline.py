"""Controls. Without these, no other number in the package means anything.

A backtest result read on its own is uninterpretable. "42% return, Sharpe
1.3" answers no question, because the question is always *compared to what*.
Three comparisons are worth having, and all three are strategies, so the
engine can produce them on identical data with identical costs:

* `BuyAndHold` - what the instrument did while you were being clever. If a
  strategy trades a hundred times to underperform holding, the trading was
  the problem.
* `RandomEntry` - the null hypothesis, made runnable. It takes coin-flip
  directional trades with the same ATR stop, the same time stop and the same
  position sizing as a real strategy, so the only difference between it and a
  real strategy is the signal. Whatever a random signal earns on this data is
  the score to beat, and beating it by a little on one sample is not beating
  it. Run it across many seeds (`validation` has the machinery) and compare
  against the distribution, not against one run.
* `AlwaysFlat` - the degenerate case that proves the engine is honest. Doing
  nothing must cost exactly nothing: no trades, no commission, no swap, final
  equity identical to initial equity to the last decimal. If a flat strategy
  loses money, the accounting is broken and every other result is suspect.

None of these are trading ideas. They are measuring instruments.
"""

from __future__ import annotations

import random
from typing import Optional

from ..indicators import atr, last
from ..strategy import BarWindow, Context, Strategy
from ..types import Direction, Instrument, OrderIntent

# See the measured note in `donchian.py`: Wilder smoothing never fully forgets
# its seed, so ATR is recomputed over a bounded tail rather than the whole
# series, at a worst-case relative error of ~2e-5. The control has to carry
# the same approximation as the strategies it is controlling for, or the
# comparison is between two different measurements.
_ATR_TAIL_MULTIPLE = 10


def _atr_now(bars: BarWindow, period: int) -> Optional[float]:
    """Current ATR, computed over a bounded tail of closed bars."""
    tail = bars[-(period * _ATR_TAIL_MULTIPLE + 1):]
    if len(tail) < period + 1:
        return None
    return last(atr(tail, period))


class BuyAndHold(Strategy):
    """Enter long once, at the first opportunity, and never exit voluntarily.

    Size is fixed in lots rather than risked against a stop, because there is
    no stop: a benchmark that sized itself from a stop distance would be a
    different strategy. The default of 0.1 lots is 10,000 units of the base
    currency, which against the engine's default 10,000 starting equity is
    roughly unlevered - the benchmark should be the instrument's return, not
    the instrument's return times an arbitrary multiplier.

    "Never exit voluntarily" is the whole behaviour, but the engine may still
    close the position: a daily risk halt or the ruin threshold will flatten
    it like any other. This strategy does not re-enter afterwards. A benchmark
    that quietly re-opens after being stopped out by a circuit breaker is
    measuring the circuit breaker, not the instrument.
    """

    name = "buy_and_hold"

    def __init__(self, lots: float = 0.1) -> None:
        if lots <= 0.0:
            raise ValueError("lots must be positive")
        super().__init__(lots=float(lots))
        self._ordered = False

    @property
    def warmup(self) -> int:
        return 0

    def on_start(self, instrument: Instrument) -> None:
        # Reset per run, so the same instance can be reused across folds of a
        # walk-forward without the second fold silently refusing to trade.
        self._ordered = False

    def on_bar(self, ctx: Context) -> Optional[OrderIntent]:
        if self._ordered:
            return None
        self._ordered = True
        return OrderIntent(
            direction=Direction.LONG,
            lots=self.lots,
            tag="buy_and_hold",
        )


class RandomEntry(Strategy):
    """Coin-flip entries with a real risk profile. The null hypothesis.

    Every bar it is flat, it draws once: with probability `prob` it opens a
    position, and its direction is a second coin flip. The stop is
    `atr_stop_mult` ATRs away and the position is sized from it exactly as a
    real strategy's would be, so the trade distribution has the same shape -
    same average risk, same cost per round turn, same exposure to swap - and
    differs only in that the signal carries no information.

    That is what makes it the right control. A strategy is not validated by
    making money; it is validated by making more money than this does on the
    same bars, by a margin that survives being run again with a different
    seed. Most published retail strategies do not clear this bar, and many
    have never been asked to.

    Reproducibility is deliberate: the instance owns a `random.Random(seed)`
    re-seeded in `on_start`, so a run is a pure function of (seed, data). It
    never touches the global `random` module, which means a caller's own use
    of `random` cannot perturb a backtest and validation runs are repeatable.
    """

    name = "random_entry"

    def __init__(
        self,
        seed: int = 0,
        prob: float = 0.02,
        atr_period: int = 14,
        atr_stop_mult: float = 2.0,
        hold_bars: int = 24,
    ) -> None:
        atr_period = int(atr_period)
        hold_bars = int(hold_bars)
        if not 0.0 < prob <= 1.0:
            raise ValueError("prob must be in (0, 1]")
        if atr_period < 1:
            raise ValueError("atr_period must be >= 1")
        if atr_stop_mult <= 0.0:
            raise ValueError("atr_stop_mult must be positive")
        if hold_bars < 1:
            raise ValueError("hold_bars must be >= 1")
        super().__init__(
            seed=int(seed),
            prob=float(prob),
            atr_period=atr_period,
            atr_stop_mult=float(atr_stop_mult),
            hold_bars=hold_bars,
        )
        self._rng = random.Random(self.seed)

    @property
    def warmup(self) -> int:
        return self.atr_period + 2

    def on_start(self, instrument: Instrument) -> None:
        self._rng = random.Random(self.seed)

    def on_bar(self, ctx: Context) -> Optional[OrderIntent]:
        position = ctx.position
        if position is not None:
            if position.bars_held >= self.hold_bars:
                return OrderIntent(direction=Direction.FLAT, tag="random_time_stop")
            return None

        # Drawn on every flat bar, before any other test, so the random stream
        # depends only on how long the strategy was flat - not on whether an
        # ATR happened to be available. Same seed, same data, same trades.
        if self._rng.random() >= self.prob:
            return None
        direction = Direction.LONG if self._rng.random() < 0.5 else Direction.SHORT

        value = _atr_now(ctx.bars, self.atr_period)
        if value is None or value <= 0.0:
            return None
        distance = value * self.atr_stop_mult
        close = ctx.price
        if direction == Direction.LONG:
            return OrderIntent(
                direction=Direction.LONG,
                stop_loss=close - distance,
                tag="random_long",
            )
        return OrderIntent(
            direction=Direction.SHORT,
            stop_loss=close + distance,
            tag="random_short",
        )


class AlwaysFlat(Strategy):
    """Never trades. Exists to prove that nothing costs nothing.

    Run this over any data and the result must contain zero trades, an equity
    curve that is flat at `initial_equity`, and no warnings beyond whatever
    the data itself provokes. It is the cheapest available check that the
    engine is not inventing costs, and it is the first thing to run after
    changing anything in `costs.py` or `engine.py`.
    """

    name = "always_flat"

    @property
    def warmup(self) -> int:
        return 0

    def on_bar(self, ctx: Context) -> Optional[OrderIntent]:
        return None
