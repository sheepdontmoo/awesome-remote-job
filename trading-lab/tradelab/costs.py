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

    def spread_points(self, buying: bool) -> float:
        """Points of spread paid on ONE leg, in the direction against the trader.

        The split depends on what the bars actually are, and getting this wrong
        is a silent subsidy rather than a crash:

        * **Mid-priced bars** (`price_is_mid=True`): the quoted price sits
          between bid and ask, so each leg pays half. Buy at mid + half, sell
          at mid - half.
        * **Bid-priced bars** (`price_is_mid=False`, which is what MetaTrader
          exports): the printed price *is* the bid. Selling deals there and
          pays nothing; buying must lift the ask, which is a **full** spread
          above it.

        Either way a round turn pays exactly one spread, which is the only
        invariant that matters. An earlier version returned zero for both legs
        on bid-priced data, so anyone backtesting their own broker's export
        traded free.
        """
        if self.config.price_is_mid:
            return self.config.spread_points / 2.0
        return self.config.spread_points if buying else 0.0

    def fill(self, mid: float, buying: bool, *, is_limit: bool = False) -> Fill:
        """Price a fill at `mid`, adjusted against the trader.

        `buying` covers both entering long and exiting short - the account is
        lifting the offer either way. `is_limit=True` marks a resting order
        (the take-profit), which pays the spread but no slippage.

        `mid` is the reference price the bar printed. The returned `price` is
        what the account deals at; the two point components are the adverse
        adjustments baked into it, kept separate so the engine can attribute
        them without recovering them by subtraction.
        """
        spread = self.spread_points(buying)
        slippage = 0.0 if is_limit else self.config.slippage_points
        sign = 1.0 if buying else -1.0
        price = mid + sign * (spread + slippage) * self.instrument.point
        return Fill(price=price, spread_points=spread, slippage_points=slippage)

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
        """Rollover instants in `(prev_ts, ts]`, in chronological order.

        Bar data has gaps - weekends, holidays, a broker's feed going quiet -
        and a position held across a gap is still financed for every night
        inside it. Walking day by day rather than asking whether a single
        boundary was crossed is what makes a Friday-to-Monday hold pay for the
        whole weekend.

        **Weekends are not charged directly.** On FX the Saturday and Sunday
        rollovers are skipped precisely *because* the Wednesday rollover is
        tripled to settle the weekend value date. Charging both, as an earlier
        version did, bills nine nights per calendar week instead of seven - a
        29% error on the financing line, which for a slow carry position is the
        entire result.

        The weekend skip is tied to `triple_swap_weekday`: when it is None the
        instrument is assumed to finance every calendar day (the usual
        arrangement on crypto and some index CFDs) and no day is skipped.
        """
        if ts <= prev_ts:
            return []
        hour = self.config.rollover_hour_utc
        triple_day = self.config.triple_swap_weekday
        skip_weekends = triple_day is not None
        day = prev_ts.date()
        instant = datetime(day.year, day.month, day.day, hour, tzinfo=prev_ts.tzinfo)
        crossings: list[datetime] = []
        while instant <= ts:
            if instant > prev_ts and not (skip_weekends and instant.weekday() >= 5):
                crossings.append(instant)
            instant += timedelta(days=1)
        return crossings
