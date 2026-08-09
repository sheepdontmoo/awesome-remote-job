# tradelab — internal build contract

This file is the contract every module builds against. Read it together with
`tradelab/types.py` and `tradelab/strategy.py`, which are already written and
are **authoritative**. Do not change them without a very good reason; other
modules depend on their exact shapes.

## Hard rules

1. **Python 3.9+ stdlib only.** No numpy, pandas, scipy, matplotlib, requests.
   `from __future__ import annotations` at the top of every module so modern
   type syntax works on older interpreters.
2. **No look-ahead, ever.** A decision made on bar `i` executes at the open of
   bar `i+1`. Indicators are causal: a value at index `i` uses only data at
   indices `<= i`.
3. **Costs are never optional.** Every fill pays spread, slippage and
   commission. Every held night pays swap.
4. **Pessimistic tie-breaking.** When a bar could plausibly have hit either the
   stop or the target, the stop wins. When a bar gaps through a stop, the fill
   is at the gapped open, not the stop price.
5. **No promises.** Docstrings and reports describe what was measured on
   historical data. Nothing in this package claims a future return.

## Execution model (engine.py)

For each bar index `i` from `warmup` to `N-1`:

1. **Rollover.** If the bar's timestamp crosses `CostConfig.rollover_hour_utc`
   and a position is open, charge swap (triple on `triple_swap_weekday`).
2. **Protective exits.** Check the open position against its stop / target
   using this bar's OHLC, in this order:
   - Gap check on `bar.open`: if the open is already through the stop, fill at
     `bar.open` (plus adverse slippage). Same for the target, but a gapped
     target fills at the target price, never better — the favourable half of a
     gap is not a gift the engine hands out.
   - Otherwise if both stop and target lie within `[low, high]`, the **stop**
     fills.
   - Otherwise whichever of stop/target lies within `[low, high]` fills.
3. **Pending intent from bar `i-1`.** Execute at `bar.open`:
   - Closing an existing position and opening a new one both pay full costs.
   - Size via `RiskConfig.risk_per_trade` unless the intent names `lots`.
4. **Mark to market** at `bar.close`; append an `EquityPoint`; update the
   position's MAE/MFE using the bar's extremes.
5. **Risk halts.** If the day's loss exceeds `RiskConfig.max_daily_loss`, close
   any position (`ExitReason.RISK_HALT`) and refuse new entries until the next
   UTC day. If equity falls below `ruin_threshold * initial_equity`, stop the
   run and set `BacktestResult.halted_early`.
6. **Strategy call.** Build a `Context` over `BarWindow(bars, i)` and call
   `strategy.on_bar(ctx)`. Store the returned intent for bar `i+1`. Apply any
   `ctx.move_stop(...)` to the live position immediately.

At the end of data, close any open position at the final close with
`ExitReason.END_OF_DATA`.

### Fill pricing

With `price_is_mid=True`, `half = spread_points / 2`:

| Action           | Base price | Adjustment                          |
|------------------|-----------|--------------------------------------|
| Buy (enter long / exit short)  | mid | `+ half*point + slippage*point` |
| Sell (enter short / exit long) | mid | `- half*point - slippage*point` |

Limit-style fills (the take-profit) pay the half-spread but **no** slippage.
Stop fills pay both — a stop becomes a market order when it triggers.

Attribute the half-spread to `Trade.spread_cost` and the slippage to
`Trade.slippage_cost`, in account currency, summed over both legs.

### Sizing

```
stop_distance = abs(entry_price - stop_loss)
if points(stop_distance) < risk.min_stop_points: skip the trade (warn)
risk_cash = equity * risk.risk_per_trade
lots      = risk_cash / (stop_distance * instrument.contract_size)
lots      = instrument.round_lots(min(lots, risk.max_lots))
if lots == 0: skip the trade (warn)
```

Gross P&L for a closed round turn:
`(exit_price - entry_price) * int(direction) * lots * contract_size`.

The engine **must** set `Trade.contract_size` from the instrument when it opens
a position. `Trade.risk_amount` and `Trade.r_multiple` depend on it; leaving the
default in place on a non-standard instrument silently scales every R-multiple.

**Invariant the tests enforce:**
`final_equity == initial_equity + sum(t.net_pnl for t in trades)` to within
1e-6, and every `EquityPoint.equity` equals balance plus open-position P&L.

## Module ownership

| File | Responsibility |
|------|----------------|
| `tradelab/types.py` | **written** — value types |
| `tradelab/strategy.py` | **written** — `Strategy`, `Context`, `BarWindow` |
| `tradelab/indicators.py` | causal SMA/EMA/ATR/stdev/Donchian/zscore/RSI |
| `tradelab/costs.py` | `CostModel`: fill pricing, commission, swap accrual |
| `tradelab/engine.py` | `Backtester.run(...) -> BacktestResult` |
| `tradelab/metrics.py` | `summarize(result) -> Metrics` |
| `tradelab/validation.py` | walk-forward, permutation tests, MC drawdown, sensitivity |
| `tradelab/data.py` | CSV loaders + synthetic generators |
| `tradelab/report.py` | text report rendering |
| `tradelab/strategies/*.py` | concrete strategies |
| `tradelab/cli.py` | `python -m tradelab ...` |

## Style

Match the existing two files: module docstring explaining *why*, dataclasses
over dicts, explicit names, comments only where the reasoning is non-obvious.
Type-annotate public functions. Keep functions short enough to read in one
screen. No emoji in code or output.
