# HonestEA — the MetaTrader 5 half of tradelab

`HonestEA.mq5` runs the same Donchian breakout rules as
`tradelab/strategies/donchian.py`, on a live IC Markets price feed, against a
**demo** account.

It exists for one reason: the Python backtest models spread, slippage,
commission and swap as configured constants, and a real broker does not. There
is no way to find out how wrong those constants are by running more backtests.
The only way is to put the same rules in front of the same feed and diff the
fills. Everything else in this file follows from that.

It is not a product, it is not tuned, and it is not expected to make money.

---

## Install

1. In MetaTrader 5: **File → Open Data Folder**. This opens the terminal's data
   directory, which is *not* where the program is installed.
2. Copy `HonestEA.mq5` into `MQL5/Experts/`.
3. Back in the terminal, press **F4** to open MetaEditor (or **Ctrl+N** in the
   Navigator to refresh), find `HonestEA.mq5` under *Experts*, and press **F7**
   to compile.
4. Fix nothing. If it does not compile cleanly on your build, say so — a
   silently patched EA is a different EA from the one that was tested.
5. In the terminal's Navigator, drag **HonestEA** onto a EURUSD H1 chart.
6. In the dialog, **Common → Allow Algo Trading**, and make sure the toolbar
   *Algo Trading* button is green.

The EA writes to the **Experts** tab of the Toolbox. Read it. Every skipped
signal, failed order and killswitch trip is logged there with a reason, and the
skipped signals are usually more informative than the taken ones.

### It will refuse to start on a live account

`OnInit` reads `AccountInfoInteger(ACCOUNT_TRADE_MODE)`. If the account is
`ACCOUNT_TRADE_MODE_REAL` and the input `AllowLiveTrading` is `false` — the
default — the EA prints an explanation and returns `INIT_FAILED`. It does not
trade, does not create an indicator handle, and does not sit there half-alive.

That check exists because attaching an EA to the wrong chart is a thing people
do, including people who are careful. Flipping `AllowLiveTrading` to `true` is
a deliberate act with a specific consequence, which is the point.

The Strategy Tester reports `ACCOUNT_TRADE_MODE_DEMO`, so this never blocks a
backtest.

---

## Inputs that matter

| Input | Default | Why |
|---|---|---|
| `AllowLiveTrading` | `false` | The live-account guard. Leave it. |
| `RiskPercent` | `0.5` | Risk per trade as a percent of equity, converted to lots from the stop distance. |
| `MaxDailyLossPct` | `3.0` | Killswitch: flatten and stand down for the rest of the server day after this drawdown from the day-open equity. |
| `MaxSpreadPoints` | `20` | Skip entries when the spread is wider than this. |
| `MinStopPoints` | `50` | Refuse to size a stop closer than this. Sizing off a near-zero stop turns 0.5% risk into 50%. |
| `EntryLookback` / `ExitLookback` / `AtrPeriod` / `AtrStopMult` | 20 / 10 / 14 / 2.0 | Must match whatever you backtested. |

The spread filter and the killswitch are not decoration. Breakouts fire when the
market is moving, which is exactly when the spread is worst, so the filter
removes a meaningful and *biased* share of signals — the log tells you how many.
Note that this makes the forward test diverge from the Python run by
construction: the engine has no spread filter, because it has no varying spread
to filter. That divergence is data, not a bug.

Sizing uses `SYMBOL_TRADE_TICK_VALUE / SYMBOL_TRADE_TICK_SIZE` to get the
account-currency value of a 1.0 price move, then floors the result onto
`SYMBOL_VOLUME_STEP` and clamps it to `SYMBOL_VOLUME_MIN` / `SYMBOL_VOLUME_MAX`.
It floors rather than rounds, because rounding up risks more than the number you
typed in.

---

## Running the Strategy Tester

**Ctrl+R** opens the tester. Set:

- **Expert**: `HonestEA`
- **Symbol**: the exact symbol from your broker (`EURUSD` on one broker is
  `EURUSD.a` or `EURUSD-Raw` on another; the contract specs differ)
- **Period**: the timeframe you backtested in Python
- **Modelling**: **Every tick based on real ticks**
- **Deposit / leverage**: whatever you will actually forward-test with
- **Optimisation**: off

### Why "Every tick based on real ticks", and not the open-price model

The tester offers four modelling modes, and three of them will lie to you about
this strategy.

- **Open prices only** evaluates once per bar, at the open. It cannot model a
  stop being hit *inside* a bar at all — it can only notice afterwards that the
  bar's range contained the stop, and then it has to guess the fill price. For a
  strategy whose entire loss distribution is defined by intrabar stop hits, that
  guess is the result.
- **1 minute OHLC** and **Every tick** both *synthesise* ticks by interpolating
  between OHLC points. The interpolation is smooth. Real price is not. The
  moments where the difference matters most — a gap, a spike through a stop, a
  news candle — are precisely the moments the interpolation smooths away.
- **Every tick based on real ticks** replays the broker's stored tick history,
  including the real bid/ask at each tick and therefore the real spread as it
  widened. It is slow. It is also the only mode whose stop fills mean anything.

Even this mode is not reality: it replays *your broker's recorded* ticks, which
is a filtered view, and it fills you at those prices without asking whether
there was size there.

Download the tick history first: **View → Symbols →** select the symbol **→
Ticks →** request the date range. Without it, "real ticks" quietly degrades.

### Spread and commission in the tester

- **Spread**: set the tester's spread field to **Current** only if you are using
  real ticks — in that mode the spread comes from the tick data. In any other
  mode, do **not** leave it on `Current`; type in a fixed value at or above your
  broker's *average* spread, not the advertised best case. IC Markets Raw Spread
  EURUSD averages around 1.0 point (0.1 pip) in liquid hours and multiples of
  that around news. A backtest run at the tightest spread ever advertised is a
  marketing document.
- **Commission**: the tester does **not** apply commission from the symbol
  settings on most builds. You must set it in the **custom symbol settings**, or
  accept that the tester's net figures are gross of commission and subtract it
  yourself. IC Markets Raw Spread is roughly USD 3.50 per lot per side, i.e. USD
  7 round turn per lot. On a strategy trading 0.1 lots that is USD 0.70 a trade
  — small until you count the trades.
- **Swap**: comes from the symbol settings, and the tester uses today's values
  for the entire historical period. Real swaps changed. A breakout system holds
  positions for weeks, so this is not a rounding error on the long side.

---

## The tester results will still flatter reality

Assume the tester's number is the best case, not the expected case. Concretely:

- **No requotes, no rejections, no partial fills.** Every order in the tester
  gets done. Live, an order sent into a fast market can come back
  `TRADE_RETCODE_REQUOTE` or `TRADE_RETCODE_PRICE_OFF`, and the EA's logged
  failure is a trade that did not happen.
- **No latency.** The tester fills at the tick it decided on. Your order takes
  tens to hundreds of milliseconds to reach the broker, and during a breakout
  the price at arrival is worse — systematically worse, in the direction you are
  going.
- **Stop fills are optimistic.** A stop becomes a market order when it triggers.
  In the tester it is filled at the trigger price plus the modelled spread. Live,
  it is filled at whatever is available after the move that triggered it.
- **Tick history is incomplete.** Gaps in the broker's recorded ticks are
  interpolated. Weekend gaps and the Sunday open are modelled crudely everywhere.
- **The killswitch and spread filter are untested by definition.** They fire on
  conditions the tester under-represents. Their real behaviour only shows up
  forward.
- **Survivorship in your own head.** You will run the tester more than once. Each
  re-run on the same data is another draw from the same lottery. Count them, and
  deflate the result accordingly (`tradelab.validation.deflated_sharpe`).

---

## The demo-first protocol

Follow it in order. Each step is allowed to end the project, and most projects
should end at step 2.

**1. Backtest in Python first.**
```bash
python3 -m tradelab run --strategy donchian --csv data/EURUSD_H1.csv
python3 -m tradelab run --strategy donchian --csv data/EURUSD_H1.csv --stress 2.0
python3 -m tradelab validate --strategy donchian --csv data/EURUSD_H1.csv
```
If the edge does not survive doubled costs, walk-forward, and a permutation
test, stop. Do not open MetaTrader. The forward test cannot rescue a strategy
that has no edge; it can only take longer to tell you the same thing.

**2. Reconcile the tester against Python.** Run the MT5 Strategy Tester on the
same symbol, timeframe and date range, with real ticks. Trade counts should be
close. They will not be identical — the spread filter, the server-day killswitch
boundary and tick-level stop hits all differ from the bar-level engine — but a
2x difference in trade count means the two implementations have diverged and one
of them is wrong. Find out which before continuing.

**3. Forward-test on a demo account for a minimum of two to three months.**
Not two weeks. A breakout system takes perhaps 20-40 trades a quarter on one
symbol at H1, and its returns come from a handful of them. A month is a sample
of noise. Leave it running, do not touch the inputs, and do not restart it
because it is losing — that is the sample you came for.

**4. The window must span at least one high-volatility event.** NFP, a CPI
print, an FOMC or ECB decision, or a genuine shock. These are the events where
the spread filter fires, where stops slip by amounts your cost model does not
contain, and where the difference between the Python assumptions and reality is
actually visible. A quiet three months tells you almost nothing about the tail
that will define your live results.

**5. Diff the demo fills against the Python assumptions.** Export the demo
account's history. For every trade, compute:
- the actual spread paid, versus `CostConfig.spread_points`;
- the slippage on each stop-out — the fill price versus the stop price — versus
  `CostConfig.slippage_points`;
- the commission actually charged, versus
  `Instrument.commission_per_lot_per_side`;
- the swap actually charged per night, versus `swap_long_points` /
  `swap_short_points`;
- the number of signals the EA logged as skipped, and why.

Then widen the Python cost model to match what you measured, and **re-run step
1**. This is the step the whole exercise exists for, and it is the step people
skip. If the strategy dies at the real costs, it died — the demo run just told
you something three months of backtesting could not.

**6. A demo account fills better than a live one.** State it plainly and plan
for it. A demo fills against the broker's quoted price with no counterparty on
the other side: no rejections, no requotes, no partial fills, no last-look, no
market impact, and latency that does not include a dealing desk deciding
anything. Live, all of those exist and every one of them costs you rather than
pays you. Treat the demo result as an **upper bound**, not an estimate.

**7. Only then, real money, at the smallest size the broker allows.** Set
`AllowLiveTrading = true` deliberately, once, and run at `SYMBOL_VOLUME_MIN` —
0.01 lots — regardless of what the sizing maths would permit. The first months
of live trading are not an income stream; they are the final measurement, the
one that includes rejections and slippage and your own behaviour under a real
drawdown. Run at minimum size for at least as long as the demo test. Only size
up if the live results match the demo results, and even then, slowly.

If at any point you find yourself skipping a step because the results so far
look good, that is the specific feeling this document was written against.

---

## Known differences from the Python engine

These are deliberate. Listing them is cheaper than rediscovering them as
"bugs" during reconciliation.

- **Day boundary.** The killswitch resets on the **broker server day**
  (GMT+2/+3 on IC Markets); `RiskConfig.max_daily_loss` resets on the **UTC**
  day. Halt windows are offset by a couple of hours.
- **Spread filter.** The EA skips entries above `MaxSpreadPoints`. The engine
  has a single constant spread and no filter, so it takes those trades.
- **Rollover.** The EA pays whatever the broker charges, whenever the broker
  charges it. The engine charges a fixed points-per-night at
  `CostConfig.rollover_hour_utc`, tripled on `triple_swap_weekday`.
- **Intrabar path.** The EA's stops are on the broker's server and fill against
  ticks. The engine sees only OHLC and resolves ambiguity pessimistically
  (stop before target, gaps fill at the gapped open). The engine's rule is a
  mitigation, not a model.
- **Margin.** The EA checks `OrderCalcMargin` against free margin and skips the
  trade if it does not fit. The engine has no margin concept.
- **One position.** Both sides trade one position at a time on one symbol. No
  pyramiding, no hedging.

---

## Disclaimer

This EA is research tooling for measuring the gap between a simulated cost model
and a broker's actual behaviour. It is not financial advice and makes no claim
about future returns. Leveraged CFD and forex trading loses money for the
substantial majority of retail accounts; your broker publishes that percentage
and it is worth reading. Run it on a demo account. Trade only money you can
afford to lose entirely.
