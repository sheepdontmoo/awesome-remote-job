"""Broker costs, modelled explicitly and charged on every fill and every night.

A backtest that ignores costs is a picture of an idea, not of a business. The
four numbers in this module - half-spread, slippage, commission per side and
overnight financing - are most of the distance between the two, and they are
the numbers a strategy author is most tempted to be optimistic about. So they
live here, in one small object, where they can be read, stressed and argued
with.

Conventions
-----------
* Everything is quoted in *points* for as long as possible, because that is how
  brokers quote spreads and swaps, and converted to account currency in exactly
  one place: `CostModel.cash`. One conversion is one place to be wrong.
* Costs always run against the trader. Buying lifts the ask, selling hits the
  bid, and a stop pays slippage on top because a stop becomes a market order the
  instant it triggers. A take-profit is a resting limit: it pays the
  half-spread but no slippage, since a limit order fills at its price or does
  not fill at all.
* Swap is signed, in the direction the broker quotes it: negative means the
  account pays. Nothing here assumes financing is a rounding error; on a
  slow-moving carry position it is the whole result.

Nothing in this module models spread widening around news, session rolls or the
Sunday open. Those are real and they are worse than the average. `CostConfig`
has `stress()` for exactly that reason, and it should be used.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from .types import CostConfig, Direction, Instrument


@dataclass(frozen=True)
class Fill:
    """A price to trade at, with its cost components kept separately.

    `price` is what the account actually deals at. `spread_points` and
    `slippage_points` are the two adverse adjustments already baked into it,
    broken out so the engine can attribute them to a `Trade` instead of losing
    them inside a single number.
    """

    price: float
    spread_points: float
    slippage_points: float


class CostModel:
    """Turns a `CostConfig` plus an `Instrument` into money.

    Stateless: every method is a pure function of its arguments and the
    configuration handed to the constructor. That makes it safe to share one
    instance across a walk-forward run and easy to test in isolation.
    """

    def __init__(self, config: CostConfig, instrument: Instrument) -> None:
        self.config = config
        self.instrument = instrument

    # -- pricing ----------------------------------------------------------

    def fill(self, mid: float, buying: bool, *, is_limit: bool = False) -> Fill:
        """Price a fill at `mid`, adjusted against the trader.

        `buying` covers both entering long and exiting short - the account is
        lifting the offer either way. `is_limit=True` marks a resting order
        (the take-profit), which pays the half-spread but no slippage.
        """
        half = self.config.half_spread()
        slippage = 0.0 if is_limit else self.config.slippage_points
        sign = 1.0 if buying else -1.0
        price = mid + sign * (half + slippage) * self.instrument.point
        return Fill(price=price, spread_points=half, slippage_points=slippage)

    def commission(self, lots: float) -> float:
        """Commission for ONE side of a round turn, in account currency."""
        return abs(lots) * self.instrument.commission_per_lot_per_side

    def cash(self, points: float, lots: float) -> float:
        """Convert a distance in points into account currency for `lots` lots."""
        return points * self.instrument.point * lots * self.instrument.contract_size

    # -- financing --------------------------------------------------------

    def swap(self, direction: Direction, lots: float, ts: datetime) -> float:
        """Financing for holding `lots` in `direction` over the night at `ts`.

        Signed the way the broker quotes it: negative is money leaving the
        account. Tripled on `CostConfig.triple_swap_weekday`, which is how the
        weekend value date is charged on FX.
        """
        if direction == Direction.LONG:
            points = self.config.swap_long_points
        elif direction == Direction.SHORT:
            points = self.config.swap_short_points
        else:
            return 0.0
        triple_day = self.config.triple_swap_weekday
        if triple_day is not None and ts.weekday() == triple_day:
            points *= 3.0
        return self.cash(points, lots)

    def rollovers_between(self, prev_ts: datetime, ts: datetime) -> list[datetime]:
        """Rollover instants in `(prev_ts, ts]`, one per calendar day.

        Returned in chronological order. Bar data has gaps - weekends, holidays,
        a broker's feed going quiet - and a position held across a gap is still
        financed for every night inside it. Walking day by day rather than
        asking whether a single boundary was crossed is what makes a
        Friday-to-Monday hold pay three nights instead of one.
        """
        if ts <= prev_ts:
            return []
        hour = self.config.rollover_hour_utc
        day = prev_ts.date()
        instant = datetime(day.year, day.month, day.day, hour, tzinfo=prev_ts.tzinfo)
        crossings: list[datetime] = []
        while instant <= ts:
            if instant > prev_ts:
                crossings.append(instant)
            instant += timedelta(days=1)
        return crossings
