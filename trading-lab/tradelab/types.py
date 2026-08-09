"""Core value types for the backtester.

Everything here is deliberately plain: dataclasses, enums, and stdlib only.
No pandas, no numpy. A backtest you cannot read line-by-line is a backtest
you cannot trust.

Conventions used throughout the package
---------------------------------------
* All prices are floats in the instrument's quote currency.
* `point` is the smallest price increment (0.00001 on a 5-digit EURUSD feed).
* `pip` is the conventional pip (0.0001 on EURUSD). Costs are quoted in
  *points*, because that is how brokers - including IC Markets - quote
  spreads and swaps.
* Bar data is assumed to be **mid** prices unless `CostConfig.price_is_mid`
  is False, in which case bars are treated as bid and the ask is derived.
* Timestamps are timezone-aware UTC `datetime` objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, IntEnum
from typing import Optional


class Direction(IntEnum):
    """Trade direction. The integer value doubles as the P&L sign multiplier."""

    SHORT = -1
    FLAT = 0
    LONG = 1

    @property
    def opposite(self) -> "Direction":
        return Direction(-int(self))


class ExitReason(str, Enum):
    """Why a position was closed. Useful for diagnosing where P&L comes from."""

    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    TRAILING_STOP = "trailing_stop"
    SIGNAL = "signal"
    TIME_STOP = "time_stop"
    END_OF_DATA = "end_of_data"
    RISK_HALT = "risk_halt"


@dataclass(frozen=True)
class Bar:
    """A single OHLCV bar. Immutable on purpose - the engine hands these to
    strategy code and must be able to guarantee they were not mutated."""

    ts: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0

    def __post_init__(self) -> None:
        if not (self.low <= self.open <= self.high):
            raise ValueError(f"bar at {self.ts}: open {self.open} outside [{self.low}, {self.high}]")
        if not (self.low <= self.close <= self.high):
            raise ValueError(f"bar at {self.ts}: close {self.close} outside [{self.low}, {self.high}]")
        if self.low > self.high:
            raise ValueError(f"bar at {self.ts}: low {self.low} > high {self.high}")

    @property
    def range(self) -> float:
        return self.high - self.low

    @property
    def typical(self) -> float:
        return (self.high + self.low + self.close) / 3.0


@dataclass(frozen=True)
class Instrument:
    """Contract specification.

    Defaults describe a 5-digit EUR/USD on an IC Markets Raw Spread account.
    Always replace these with the real numbers from your own account - the
    contract specs page and, better, an actual statement. Guessing costs is
    the single most common way a backtest flatters itself.
    """

    symbol: str = "EURUSD"
    point: float = 0.00001
    pip: float = 0.0001
    digits: int = 5
    contract_size: float = 100_000.0
    min_lot: float = 0.01
    max_lot: float = 100.0
    lot_step: float = 0.01
    # Round-turn commission is charged per side; IC Markets Raw Spread is
    # ~USD 3.50 per lot per side on FX majors.
    commission_per_lot_per_side: float = 3.50
    # True when the account currency equals the quote currency (EURUSD on a
    # USD account). When False the engine's P&L is an approximation and says so.
    quote_is_account_currency: bool = True

    def points(self, price_delta: float) -> float:
        """Convert a price distance into points."""
        return price_delta / self.point

    def price(self, points: float) -> float:
        """Convert a distance in points into a price distance."""
        return points * self.point

    def round_lots(self, lots: float) -> float:
        """Snap a lot size down to a tradable size, or 0.0 if below the minimum.

        The cap is applied *before* the step flooring, not after. Capping last
        would return `max_lot` itself even when it is not a whole number of
        lot steps - a size the broker rejects.
        """
        capped = min(lots, self.max_lot)
        if capped < self.min_lot:
            return 0.0
        stepped = int(round(capped / self.lot_step + 1e-9)) * self.lot_step
        if stepped > capped + 1e-12:
            stepped -= self.lot_step
        if stepped < self.min_lot:
            return 0.0
        return round(stepped, 8)

    def value_per_price_unit(self, lots: float) -> float:
        """Account-currency value of a 1.0 move in price, for `lots` lots."""
        return lots * self.contract_size


@dataclass(frozen=True)
class CostConfig:
    """Transaction costs. These are the numbers that decide whether an edge
    survives contact with a real broker.

    Spread and swap are in *points*. Defaults are deliberately conservative
    rather than best-case: a backtest run at the tightest spread the broker
    has ever advertised is a marketing document, not a test.
    """

    # Average spread you actually pay, in points. IC Markets Raw Spread EURUSD
    # averages ~0.1 pip = 1.0 point in liquid hours, but widens hard around
    # news, at the session roll, and on Sunday open.
    spread_points: float = 1.0
    # Extra adverse fill on market/stop orders, in points. Stops slip; limits
    # do not (they just fail to fill, which the engine models separately).
    slippage_points: float = 2.0
    # Overnight financing per night, in points, signed: negative = you pay.
    swap_long_points: float = -0.8
    swap_short_points: float = 0.2
    # Wednesday carries a triple swap on FX to cover the weekend value date.
    triple_swap_weekday: Optional[int] = 2  # Monday=0 ... Sunday=6
    # Hour (UTC) at which financing is applied to open positions.
    rollover_hour_utc: int = 21
    price_is_mid: bool = True

    # How the spread splits across the two legs of a round turn depends on
    # whether these bars are mid or bid, so it lives in `CostModel.spread_points`
    # where both cases are handled together. There is deliberately no
    # `half_spread()` helper here: the symmetric-half assumption it encoded is
    # simply wrong for the bid-priced bars MetaTrader exports, and it charged
    # nothing at all on them.

    def stress(self, factor: float) -> "CostConfig":
        """Return a copy with spread and slippage multiplied.

        Run every promising strategy at `stress(2.0)` before believing it.
        An edge that dies when costs double was never an edge; it was a
        rounding error with good marketing.
        """
        return CostConfig(
            spread_points=self.spread_points * factor,
            slippage_points=self.slippage_points * factor,
            swap_long_points=self.swap_long_points * factor,
            swap_short_points=self.swap_short_points * factor,
            triple_swap_weekday=self.triple_swap_weekday,
            rollover_hour_utc=self.rollover_hour_utc,
            price_is_mid=self.price_is_mid,
        )


@dataclass(frozen=True)
class RiskConfig:
    """Position sizing and account-level circuit breakers."""

    # Fraction of current equity risked between entry and initial stop.
    risk_per_trade: float = 0.005  # 0.5%
    # Hard cap on position size regardless of what the risk maths says.
    max_lots: float = 5.0
    # Stop trading for the rest of the day after losing this fraction of the
    # day's starting equity. Set to None to disable.
    max_daily_loss: Optional[float] = 0.03
    # Abandon the backtest entirely below this fraction of starting equity.
    ruin_threshold: float = 0.30
    # Refuse to size a trade whose stop is closer than this, in points.
    # Protects against divide-by-nearly-zero sizing blowups.
    min_stop_points: float = 50.0


@dataclass
class Trade:
    """A completed round turn, with costs broken out.

    `gross_pnl` is what a naive backtest would report. `net_pnl` is what
    would have landed in the account. The gap between them is the entire
    reason most published backtests do not survive live trading.

    The two must be measured from different prices or the arithmetic
    double-counts. `entry_price` and `exit_price` are the prices actually
    dealt at, already moved against the trader by spread and slippage, while
    `gross_pnl` is measured from the *reference* prices the bars printed. So
    `gross_pnl` is the costless ideal, each cost field is charged exactly
    once, and `net_pnl` lands back on the realised round turn.
    """

    direction: Direction
    lots: float
    entry_ts: datetime
    entry_price: float
    exit_ts: datetime
    exit_price: float
    exit_reason: ExitReason
    initial_stop: Optional[float] = None
    gross_pnl: float = 0.0
    commission: float = 0.0
    swap: float = 0.0
    slippage_cost: float = 0.0
    spread_cost: float = 0.0
    bars_held: int = 0
    mae: float = 0.0  # maximum adverse excursion, account currency
    mfe: float = 0.0  # maximum favourable excursion, account currency
    tag: str = ""
    # Account-currency value of a 1.0 price move per lot. The engine sets this
    # from the instrument; without it, risk cannot be expressed in cash and
    # every R-multiple would be wrong by a factor of the contract size.
    contract_size: float = 100_000.0

    @property
    def total_costs(self) -> float:
        """All costs including financing, as a positive number when they hurt."""
        return self.commission + self.spread_cost + self.slippage_cost - self.swap

    @property
    def net_pnl(self) -> float:
        return self.gross_pnl - self.commission - self.spread_cost - self.slippage_cost + self.swap

    @property
    def risk_amount(self) -> Optional[float]:
        """Account-currency loss the initial stop was sized to accept."""
        if self.initial_stop is None:
            return None
        return abs(self.entry_price - self.initial_stop) * self.lots * self.contract_size

    @property
    def r_multiple(self) -> Optional[float]:
        """Net P&L expressed in units of initial risk. The only trade metric
        worth comparing across instruments and account sizes."""
        risk = self.risk_amount
        if not risk:
            return None
        return self.net_pnl / risk

    @property
    def is_win(self) -> bool:
        return self.net_pnl > 0


@dataclass(frozen=True)
class EquityPoint:
    """One mark-to-market observation of the account."""

    ts: datetime
    equity: float          # balance plus open-position P&L
    balance: float         # realised only
    exposure_lots: float   # signed: positive long, negative short
    drawdown: float        # fraction below the running peak, >= 0


@dataclass
class OrderIntent:
    """What a strategy asks for at the close of a bar.

    The engine executes it at the **open of the next bar**. A strategy can
    never act on the bar it is looking at; that is the single most important
    invariant in this package.
    """

    direction: Direction
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    lots: Optional[float] = None  # None -> size from RiskConfig
    tag: str = ""
    close_existing: bool = True

    def __post_init__(self) -> None:
        if self.lots is not None and self.lots < 0:
            raise ValueError("lots must be non-negative")


@dataclass
class BacktestResult:
    """Everything a run produced. Deliberately raw - `metrics` and `report`
    turn this into numbers and prose, but the trades and the equity curve are
    the ground truth and stay inspectable."""

    trades: list[Trade] = field(default_factory=list)
    equity_curve: list[EquityPoint] = field(default_factory=list)
    initial_equity: float = 10_000.0
    instrument: Instrument = field(default_factory=Instrument)
    costs: CostConfig = field(default_factory=CostConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    strategy_name: str = ""
    params: dict = field(default_factory=dict)
    bars_processed: int = 0
    halted_early: bool = False
    warnings: list[str] = field(default_factory=list)

    @property
    def final_equity(self) -> float:
        return self.equity_curve[-1].equity if self.equity_curve else self.initial_equity

    @property
    def gross_pnl(self) -> float:
        return sum(t.gross_pnl for t in self.trades)

    @property
    def net_pnl(self) -> float:
        return sum(t.net_pnl for t in self.trades)

    @property
    def total_costs(self) -> float:
        return sum(t.total_costs for t in self.trades)
