"""The bar loop: the only place in this package where money changes hands.

Everything else - indicators, strategies, metrics, reports - is arithmetic over
numbers this module produced. So this module is written to be boring and to be
audited line by line, because a subtle error here does not announce itself; it
shows up as an equity curve that is quietly too good.

Three ideas hold it together.

**Causality.** A strategy sees bar `i` and its intent executes at the open of
bar `i + 1`. The strategy is handed a `BarWindow` that physically cannot reach
past `i`, and the engine never consults a bar it has not reached. There is no
"if the close was better we would have used a limit" anywhere in here.

**Pessimism at every fork.** When a bar's high and low bracket both the stop and
the target, the engine has no idea which came first, so it fills the stop. When
price gaps through a stop, the fill is the gapped open, not the stop price -
the gap is a loss, not a courtesy. When price gaps through a target, the fill is
the target, not the better open - the favourable half of a gap is not something
the engine hands out for free. These choices are not conservatism for its own
sake; they are the only assumptions that survive not knowing the intrabar path.

**Accounting that is true by construction.** Realised P&L, net of every cost,
lands in `balance` when a trade closes and nowhere else. Costs accrued by an
open position are carried on the position and only reach `balance` through the
`Trade` that closes it. Equity is `balance` plus the open position's mark. Since
the run always ends flat, `final_equity == initial_equity + sum(net_pnl)` is a
consequence of the structure rather than a coincidence that has to be checked.

Known limitations
-----------------
* One position at a time. No pyramiding, no hedging, no basket of instruments.
* No partial fills. An order fills in full or not at all.
* No requotes and no rejected orders. Real brokers do both, usually at the
  moment a strategy most wants to trade.
* Bar data hides the intrabar path. The engine knows the open, high, low and
  close and nothing about the order they arrived in, which is why the
  tie-breaking above is deliberately unkind.
* The account currency is assumed equal to the quote currency. When
  `Instrument.quote_is_account_currency` is False the P&L is an approximation,
  and the run says so in `BacktestResult.warnings` rather than pretending.
* Spread and slippage are constants. Real spreads widen exactly when a strategy
  is most likely to be trading.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Optional

from .costs import CostModel
from .strategy import BarWindow, Context, PositionView, Strategy
from .types import (
    BacktestResult,
    Bar,
    CostConfig,
    Direction,
    EquityPoint,
    ExitReason,
    Instrument,
    OrderIntent,
    RiskConfig,
    Trade,
)

# Warnings are diagnostics, not a log file. A pathological strategy can skip
# thousands of trades; keeping the first few hundred is enough to diagnose it.
_MAX_WARNINGS = 200


@dataclass
class _OpenPosition:
    """The engine's mutable position. Strategies never see this object - they
    get a `PositionView` snapshot - so the engine can be sure nothing outside
    this module edited a stop or a lot size behind its back.

    The cost fields hold the *entry leg only*, accrued but not yet realised.
    They reach the account through the `Trade` that closes the position.
    """

    direction: Direction
    lots: float
    entry_ts: datetime
    entry_price: float
    # The bar's own price at entry, before spread and slippage moved the fill.
    # Gross P&L is measured from here so that the cost fields below can be
    # charged once rather than being buried in the fill and subtracted again.
    entry_mid: float
    stop_loss: Optional[float]
    take_profit: Optional[float]
    initial_stop: Optional[float]
    tag: str = ""
    bars_held: int = 0
    commission: float = 0.0
    spread_cost: float = 0.0
    slippage_cost: float = 0.0
    swap: float = 0.0  # signed: negative means the account paid
    mae: float = 0.0  # magnitudes, account currency, gross of costs
    mfe: float = 0.0
    stop_moved: bool = False


class Backtester:
    """Runs one strategy over one list of bars and reports what happened.

    The instance holds only configuration between runs; all run state is
    rebuilt inside `run`, so the same `Backtester` can be reused across a
    walk-forward or a permutation test without leaking equity from one fold
    into the next.
    """

    def __init__(
        self,
        instrument: Instrument = Instrument(),
        costs: CostConfig = CostConfig(),
        risk: RiskConfig = RiskConfig(),
        initial_equity: float = 10_000.0,
    ) -> None:
        self.instrument = instrument
        self.costs = costs
        self.risk = risk
        self.initial_equity = float(initial_equity)
        self.cost_model = CostModel(costs, instrument)

    # -- public API -------------------------------------------------------

    def run(self, strategy: Strategy, bars: list[Bar]) -> BacktestResult:
        """Execute `strategy` over `bars` and return the raw record of the run."""
        result = self._new_result(strategy)
        self._reset(result)
        if not self._precheck(bars, strategy, result):
            return result

        strategy.on_start(self.instrument)
        prev_ts: Optional[datetime] = None
        last = len(bars) - 1

        for i in range(max(0, strategy.warmup), len(bars)):
            bar = bars[i]
            self._roll_day(bar.ts)

            # 1. Financing for every night since the previous processed bar.
            if self._position is not None and prev_ts is not None:
                self._charge_swap(prev_ts, bar.ts)

            # 2. Protective exits, checked against this bar's OHLC. Excursions
            # are recorded first: every exit path clears the position, so
            # measuring afterwards would skip the bar a trade was stopped out
            # on - which is, by construction, its worst bar.
            if self._position is not None:
                self._update_excursions(bar)
                self._check_protective_exit(bar)

            # 3. The intent formed at the close of bar i-1, filled at this open.
            if self._pending is not None:
                intent, self._pending = self._pending, None
                self._execute_intent(intent, bar)
                # A position opened at this bar's open then lived through the
                # rest of this bar, so its stop and target have to be checked
                # against this bar's range too. Letting them go live only on the
                # next bar backtests a tight stop as though it were not there:
                # the tighter the stop relative to a bar's range, the more of
                # the equity curve is manufactured by the hole.
                if self._position is not None and self._position.entry_ts == bar.ts:
                    self._update_excursions(bar)
                    self._check_protective_exit(bar, from_open=False)

            # 4. Mark to market: age the position and record its excursions.
            if self._position is not None:
                self._position.bars_held += 1
                self._update_excursions(bar)

            # 5. Circuit breakers, evaluated on the marked equity.
            self._apply_risk_halts(bar)

            # 6. The strategy's turn. Its intent belongs to the next bar.
            if not self._halted_early:
                self._call_strategy(strategy, bars, i, bar, is_last=i == last)

            if self._halted_early or i == last:
                self._close_position(bar.close, bar.ts, ExitReason.END_OF_DATA)

            # One point per processed bar, recorded after every event on this
            # bar has settled, so the last point is the account's true final
            # state rather than a mark taken moments before the last exit.
            self._record_equity(bar)
            result.bars_processed += 1
            prev_ts = bar.ts
            if self._halted_early:
                break

        strategy.on_finish()
        result.halted_early = self._halted_early
        return result

    # -- run setup --------------------------------------------------------

    def _new_result(self, strategy: Strategy) -> BacktestResult:
        return BacktestResult(
            initial_equity=self.initial_equity,
            instrument=self.instrument,
            costs=self.costs,
            risk=self.risk,
            strategy_name=strategy.name,
            params=dict(getattr(strategy, "params", {})),
        )

    def _reset(self, result: BacktestResult) -> None:
        self._result = result
        self._balance = self.initial_equity
        self._position: Optional[_OpenPosition] = None
        self._pending: Optional[OrderIntent] = None
        self._peak_equity = self.initial_equity
        self._last_equity = self.initial_equity
        self._day: Optional[date] = None
        self._day_start_equity = self.initial_equity
        self._halted_day = False
        self._halted_early = False

    def _precheck(self, bars: list[Bar], strategy: Strategy, result: BacktestResult) -> bool:
        """Sanity-check the inputs. Returns False when there is nothing to run."""
        if not self.instrument.quote_is_account_currency:
            self._warn(
                "instrument.quote_is_account_currency is False: P&L is an "
                "approximation that ignores the conversion rate"
            )
        if not bars:
            self._warn("no bars supplied; nothing was run")
            return False
        if strategy.warmup >= len(bars):
            self._warn(
                f"strategy warmup of {strategy.warmup} bars needs more history than "
                f"the {len(bars)} bars supplied; nothing was run"
            )
            return False
        for i in range(1, len(bars)):
            if bars[i].ts <= bars[i - 1].ts:
                self._warn(
                    f"bars are not strictly increasing in time at {_stamp(bars[i].ts)}; "
                    "results depend on the order supplied"
                )
                break
        return True

    # -- accounting -------------------------------------------------------

    def _unrealized(self, price: float) -> float:
        """Open-position P&L at `price`: mark plus accrued swap, minus the
        entry leg's costs. Exit costs are not guessed at - they are charged
        when the exit actually happens.

        The mark runs from `entry_mid`, not from the fill. Running it from the
        fill would subtract the entry spread and slippage a second time, and
        mark a position down by twice its entry costs the instant it opens -
        which flows straight into drawdown, Ulcer and Sharpe.
        """
        pos = self._position
        if pos is None:
            return 0.0
        move = (price - pos.entry_mid) * int(pos.direction)
        gross = move * pos.lots * self.instrument.contract_size
        return gross + pos.swap - pos.commission - pos.spread_cost - pos.slippage_cost

    def _equity(self, price: float) -> float:
        return self._balance + self._unrealized(price)

    def _record_equity(self, bar: Bar) -> None:
        equity = self._equity(bar.close)
        if equity > self._peak_equity:
            self._peak_equity = equity
        drawdown = 0.0
        if self._peak_equity > 0.0:
            drawdown = max(0.0, (self._peak_equity - equity) / self._peak_equity)
        exposure = 0.0
        if self._position is not None:
            exposure = self._position.lots * int(self._position.direction)
        self._result.equity_curve.append(
            EquityPoint(
                ts=bar.ts,
                equity=equity,
                balance=self._balance,
                exposure_lots=exposure,
                drawdown=drawdown,
            )
        )
        self._last_equity = equity

    def _roll_day(self, ts: datetime) -> None:
        """Start a new UTC day: reset the daily loss budget and any halt."""
        day = ts.date()
        if day != self._day:
            self._day = day
            self._day_start_equity = self._last_equity
            self._halted_day = False

    def _charge_swap(self, prev_ts: datetime, ts: datetime) -> None:
        pos = self._position
        assert pos is not None
        for instant in self.cost_model.rollovers_between(prev_ts, ts):
            pos.swap += self.cost_model.swap(pos.direction, pos.lots, instant)

    # -- exits ------------------------------------------------------------

    def _check_protective_exit(self, bar: Bar, *, from_open: bool = True) -> None:
        """Fill the stop or the target if this bar could have touched either.

        `from_open=False` is used for a position opened at this bar's own open.
        The gap branches below are skipped in that case: the account has just
        dealt at that open, so it cannot also have gapped through a level on
        the way to it. Only the intrabar range applies.
        """
        pos = self._position
        assert pos is not None
        stop, target = pos.stop_loss, pos.take_profit
        long = pos.direction == Direction.LONG
        stop_reason = ExitReason.TRAILING_STOP if pos.stop_moved else ExitReason.STOP_LOSS

        # The open is the first price the account could have dealt at, so gaps
        # are resolved before anything that happened later in the bar.
        if from_open:
            if stop is not None and ((long and bar.open <= stop) or (not long and bar.open >= stop)):
                self._close_position(bar.open, bar.ts, stop_reason)
                return
            if target is not None and ((long and bar.open >= target) or (not long and bar.open <= target)):
                # Filled at the limit, never at the better gapped open.
                self._close_position(target, bar.ts, ExitReason.TAKE_PROFIT, is_limit=True)
                return

        stop_hit = stop is not None and ((long and bar.low <= stop) or (not long and bar.high >= stop))
        target_hit = target is not None and ((long and bar.high >= target) or (not long and bar.low <= target))
        if stop_hit:
            # Checked first, so a bar that brackets both resolves as a loss.
            self._close_position(stop, bar.ts, stop_reason)
        elif target_hit:
            self._close_position(target, bar.ts, ExitReason.TAKE_PROFIT, is_limit=True)

    def _close_position(
        self,
        mid: float,
        ts: datetime,
        reason: ExitReason,
        *,
        is_limit: bool = False,
    ) -> None:
        """Close the open position at `mid` and realise the whole round turn."""
        pos = self._position
        if pos is None:
            return
        buying = pos.direction == Direction.SHORT  # closing a short is a buy
        fill = self.cost_model.fill(mid, buying, is_limit=is_limit)
        # Measured mid-to-mid, so the cost fields on the Trade are charged once
        # each instead of also being embedded in the two fill prices.
        move = (mid - pos.entry_mid) * int(pos.direction)
        trade = Trade(
            direction=pos.direction,
            lots=pos.lots,
            entry_ts=pos.entry_ts,
            entry_price=pos.entry_price,
            exit_ts=ts,
            exit_price=fill.price,
            exit_reason=reason,
            initial_stop=pos.initial_stop,
            gross_pnl=move * pos.lots * self.instrument.contract_size,
            commission=pos.commission + self.cost_model.commission(pos.lots),
            swap=pos.swap,
            slippage_cost=pos.slippage_cost + self.cost_model.cash(fill.slippage_points, pos.lots),
            spread_cost=pos.spread_cost + self.cost_model.cash(fill.spread_points, pos.lots),
            bars_held=pos.bars_held,
            mae=pos.mae,
            mfe=pos.mfe,
            tag=pos.tag,
            contract_size=self.instrument.contract_size,
        )
        self._position = None
        self._balance += trade.net_pnl
        self._result.trades.append(trade)

    # -- entries ----------------------------------------------------------

    def _execute_intent(self, intent: OrderIntent, bar: Bar) -> None:
        """Act on the previous bar's intent, at this bar's open."""
        pos = self._position
        if intent.direction == Direction.FLAT:
            if pos is not None and intent.close_existing:
                self._close_position(bar.open, bar.ts, ExitReason.SIGNAL)
            return

        if pos is not None:
            if pos.direction == intent.direction:
                # Already positioned the right way round. Paying a round turn to
                # re-enter the same trade is a cost with no economic content, so
                # the intent is read as an update to the protective levels.
                if intent.stop_loss is not None:
                    # A widened stop is allowed - a strategy may have reason -
                    # but the risk it reports must follow it. `initial_stop`
                    # tracks the *widest* stop the trade ever carried, so an
                    # R-multiple is measured against the risk that was really
                    # taken rather than against the tighter one it started
                    # with. Silently keeping the original scales every
                    # R-multiple, expectancy_r and best/worst-R by a factor
                    # nothing recorded.
                    if pos.initial_stop is not None and _widens(
                        pos.direction, pos.entry_mid, pos.initial_stop, intent.stop_loss
                    ):
                        self._warn(
                            f"{_stamp(bar.ts)}: stop widened from {pos.initial_stop:.5f} to "
                            f"{intent.stop_loss:.5f}; risk is now measured against the wider one"
                        )
                        pos.initial_stop = intent.stop_loss
                    pos.stop_loss = intent.stop_loss
                    pos.stop_moved = True
                if intent.take_profit is not None:
                    pos.take_profit = intent.take_profit
                return
            if not intent.close_existing:
                return  # one position at a time; a reversal must close first
            self._close_position(bar.open, bar.ts, ExitReason.SIGNAL)

        self._open_position(intent, bar)

    def _open_position(self, intent: OrderIntent, bar: Bar) -> None:
        if self._halted_day:
            self._warn(f"{_stamp(bar.ts)}: entry skipped, daily risk halt is active")
            return
        buying = intent.direction == Direction.LONG
        fill = self.cost_model.fill(bar.open, buying)
        lots = self._size(intent, fill.price, bar)
        if lots <= 0.0:
            return
        self._position = _OpenPosition(
            direction=intent.direction,
            lots=lots,
            entry_ts=bar.ts,
            entry_price=fill.price,
            entry_mid=bar.open,
            stop_loss=intent.stop_loss,
            take_profit=intent.take_profit,
            initial_stop=intent.stop_loss,
            tag=intent.tag,
            commission=self.cost_model.commission(lots),
            spread_cost=self.cost_model.cash(fill.spread_points, lots),
            slippage_cost=self.cost_model.cash(fill.slippage_points, lots),
        )

    def _size(self, intent: OrderIntent, entry_price: float, bar: Bar) -> float:
        """Lots for a new position, or 0.0 with a warning when it cannot be sized.

        Sizing uses the *filled* entry price, not the bar's open: the risk that
        matters is the distance the account will actually travel to its stop.
        """
        inst = self.instrument
        when = _stamp(bar.ts)
        if intent.lots is not None:
            lots = inst.round_lots(min(intent.lots, self.risk.max_lots))
            if lots <= 0.0:
                self._warn(
                    f"{when}: trade skipped, requested size {intent.lots:.4f} lots rounds "
                    f"to zero at a minimum of {inst.min_lot} lots"
                )
            return lots

        stop = intent.stop_loss
        if stop is None:
            self._warn(
                f"{when}: trade skipped, intent has neither a stop loss nor an explicit "
                "lot size, so risk cannot be sized"
            )
            return 0.0
        long = intent.direction == Direction.LONG
        if (long and stop >= entry_price) or (not long and stop <= entry_price):
            self._warn(
                f"{when}: trade skipped, stop {stop:.5f} is on the wrong side of the "
                f"fill {entry_price:.5f}"
            )
            return 0.0

        # Two different distances, on purpose. The guard asks the question the
        # user configured - "is my stop at least N points from the price?" - so
        # it measures from the bar, not from the fill, which sits a spread and a
        # slippage further away and would quietly let a 47-point stop through a
        # 50-point minimum. Sizing then uses the fill, because the cash the
        # account actually puts at risk is the distance it will really travel.
        quoted_points = inst.points(abs(bar.open - stop))
        if quoted_points < self.risk.min_stop_points:
            self._warn(
                f"{when}: trade skipped, stop is {quoted_points:.1f} points away, inside the "
                f"{self.risk.min_stop_points:.1f}-point minimum"
            )
            return 0.0
        distance = abs(entry_price - stop)
        stop_points = inst.points(distance)

        risk_cash = self._equity(bar.open) * self.risk.risk_per_trade
        lots = risk_cash / (distance * inst.contract_size)
        lots = inst.round_lots(min(lots, self.risk.max_lots))
        if lots <= 0.0:
            self._warn(
                f"{when}: trade skipped, risking {risk_cash:.2f} over {stop_points:.1f} points "
                f"sizes below the {inst.min_lot}-lot minimum"
            )
        return lots

    # -- per-bar bookkeeping ----------------------------------------------

    def _update_excursions(self, bar: Bar) -> None:
        """Track how far the trade went against and in favour, in account
        currency and gross of costs. Both are magnitudes: never negative.

        Called twice per bar - once before the protective-exit check so a
        stopped-out trade records the bar that stopped it, and once at the mark
        so a position opened on this bar is not missed. `max` makes the overlap
        harmless.
        """
        pos = self._position
        assert pos is not None
        value = pos.lots * self.instrument.contract_size
        if pos.direction == Direction.LONG:
            adverse = (pos.entry_mid - bar.low) * value
            favourable = (bar.high - pos.entry_mid) * value
        else:
            adverse = (bar.high - pos.entry_mid) * value
            favourable = (pos.entry_mid - bar.low) * value
        pos.mae = max(pos.mae, adverse, 0.0)
        pos.mfe = max(pos.mfe, favourable, 0.0)

    def _apply_risk_halts(self, bar: Bar) -> None:
        """Daily loss limit first, then the account-level ruin threshold."""
        equity = self._equity(bar.close)
        limit = self.risk.max_daily_loss
        if limit is not None and not self._halted_day:
            budget = self._day_start_equity * limit
            if self._day_start_equity - equity > budget:
                self._close_position(bar.close, bar.ts, ExitReason.RISK_HALT)
                self._halted_day = True
                self._warn(
                    f"{_stamp(bar.ts)}: daily loss of "
                    f"{self._day_start_equity - equity:.2f} exceeds the {budget:.2f} limit; "
                    "no further entries today"
                )
                equity = self._equity(bar.close)

        floor = self.risk.ruin_threshold * self.initial_equity
        if equity < floor:
            self._close_position(bar.close, bar.ts, ExitReason.RISK_HALT)
            self._halted_early = True
            self._warn(
                f"{_stamp(bar.ts)}: equity {self._equity(bar.close):.2f} is below the ruin "
                f"threshold of {floor:.2f}; the run was abandoned here"
            )

    def _call_strategy(
        self,
        strategy: Strategy,
        bars: list[Bar],
        index: int,
        bar: Bar,
        *,
        is_last: bool,
    ) -> None:
        ctx = Context(
            bars=BarWindow(bars, index),
            position=self._position_view(bar.close),
            equity=self._equity(bar.close),
            balance=self._balance,
            instrument=self.instrument,
        )
        intent = strategy.on_bar(ctx)
        # A requested stop move takes effect now, so it guards every later bar
        # but cannot rewrite what this bar already did.
        if ctx._stop_move is not None and self._position is not None:
            self._position.stop_loss = ctx._stop_move
            self._position.stop_moved = True
        # An intent formed on the final bar has no bar left to execute on.
        self._pending = None if is_last else intent

    def _position_view(self, price: float) -> Optional[PositionView]:
        pos = self._position
        if pos is None:
            return None
        return PositionView(
            direction=pos.direction,
            lots=pos.lots,
            entry_price=pos.entry_price,
            entry_ts=pos.entry_ts,
            stop_loss=pos.stop_loss,
            take_profit=pos.take_profit,
            bars_held=pos.bars_held,
            unrealized_pnl=self._unrealized(price),
            mae=pos.mae,
            mfe=pos.mfe,
        )

    def _warn(self, message: str) -> None:
        warnings = self._result.warnings
        if len(warnings) < _MAX_WARNINGS:
            warnings.append(message)
        elif len(warnings) == _MAX_WARNINGS:
            warnings.append(f"further warnings suppressed after {_MAX_WARNINGS}")


def _stamp(ts: datetime) -> str:
    """Timestamps in warnings are for a human reading a report, not for parsing."""
    return ts.strftime("%Y-%m-%d %H:%M")


def _widens(direction: Direction, entry: float, old_stop: float, new_stop: float) -> bool:
    """True when `new_stop` sits further from entry than `old_stop`, i.e. the
    trade is being given more room and is therefore risking more."""
    return abs(new_stop - entry) > abs(old_stop - entry) and (
        (direction == Direction.LONG and new_stop < old_stop)
        or (direction == Direction.SHORT and new_stop > old_stop)
    )
