# tradelab

A zero-dependency backtesting lab for retail forex and CFD strategies, built to
be **hard to fool yourself with**.

This exists because of a screenshot: a viral post claiming a 19-year-old turned
$68 into $750,000 in a few days with a bot built in 48 hours, with a
"comment / like / follow and I'll DM you the setup" call to action. That is a
follower-farming funnel, not a trading system. One of the screenshots in that
very thread has the word `SIMULATED` printed across the chart.

The honest version of that idea is not a money printer. It is a research
harness that tells you, as coldly as possible, whether a rule set has an edge
that survives spread, commission, swap, slippage, and the multiple-testing
problem. Usually the answer is no. A tool that can say no is the only kind
worth having.

---

## Install

There is nothing to install.

```bash
cd trading-lab
python3 -m tradelab demo
```

Python 3.9+. Standard library only — no numpy, no pandas. That is deliberate:
you can read every line that produces a number, and the whole package runs
anywhere Python does.

## Use it

```bash
# Run a strategy on synthetic data (no data files needed)
python3 -m tradelab run --strategy donchian

# Run on your own MT5 export
python3 -m tradelab run --strategy donchian --csv data/EURUSD_H1.csv

# Same run, with every cost doubled — the test that matters
python3 -m tradelab run --strategy donchian --csv data/EURUSD_H1.csv --stress 2.0

# Tune parameters, then ask whether the tuning meant anything
python3 -m tradelab validate --strategy donchian --csv data/EURUSD_H1.csv
```

```python
from tradelab import Backtester, CostConfig, Instrument, RiskConfig, summarize, render_report
from tradelab.data import load_mt5_csv
from tradelab.strategies import DonchianBreakout

bars = load_mt5_csv("data/EURUSD_H1.csv")
bt = Backtester(
    instrument=Instrument(symbol="EURUSD", commission_per_lot_per_side=3.50),
    costs=CostConfig(spread_points=1.0, slippage_points=2.0, price_is_mid=False),
    risk=RiskConfig(risk_per_trade=0.005),
    initial_equity=10_000.0,
)
result = bt.run(DonchianBreakout(entry_lookback=20, atr_stop_mult=2.0), bars)
print(render_report(result, summarize(result)))
```

---

## What makes this different from a backtest that lies to you

**Look-ahead is impossible, not merely discouraged.** A strategy is handed a
`BarWindow` that ends at the bar which just closed. Reaching past it raises
`LookAheadError` instead of quietly returning tomorrow's price. Decisions made
on bar *i* execute at the **open of bar *i+1***, never at bar *i*'s close.
Donchian channels exclude the current bar, because including it makes a
breakout system look prescient.

**Costs are charged on every fill, and broken out.** Half-spread on each leg,
commission per side, slippage on market and stop orders (but not limits),
swap every night with the triple-swap weekday. The report shows gross P&L and
net P&L side by side, so the gap between the pretty number and the real one is
impossible to miss. `CostConfig.stress(2.0)` doubles everything — an edge that
dies there was a rounding error with good marketing.

**Ties break against you.** If a bar spans both your stop and your target, the
stop fills. If price gaps through your stop, you fill at the gapped open, not
at your stop price. A gapped take-profit never fills better than its limit.
Real brokers are not generous, and neither is this.

**The null hypothesis is runnable.** `RandomEntry` takes coin-flip trades with
the same risk profile and cost structure as a real strategy. If your system
can't clearly beat a monkey with a stop-loss on your own data, it has nothing.
`synthetic_random_walk()` generates a series with **zero exploitable edge by
construction** — whatever your strategy scores on it is the score of luck, and
that number is your bar for "impressive-looking."

**Overfitting gets measured, not assumed away.**

| Tool | Question it answers |
|---|---|
| `walk_forward` | Do the parameters chosen in-sample survive out-of-sample? |
| `permutation_test` | Could this result have come from a series with the signal destroyed? |
| `monte_carlo_drawdown` | How bad could the drawdown have been if the trades arrived in a different order? |
| `parameter_sensitivity` | Is the edge a broad plateau, or a lone spike surrounded by losers? |
| `deflated_sharpe` | What's the Sharpe worth after accounting for how many configurations you tried? |

That last one is the quiet killer. Test 200 parameter sets on pure noise and
you are *guaranteed* an impressive winner. Every backtester finds it; almost
none of them haircut it.

**The report argues with you.** `Metrics.caveats()` emits plain-English
warnings the numbers themselves justify — fewer than 30 trades, cost drag
above 50% of gross, a losing streak longer than most people can actually sit
through. When nothing triggers, it says *"No automatic caveats triggered. This
is not an endorsement."*

---

## Layout

```
trading-lab/
├── tradelab/
│   ├── types.py         Bar, Instrument, CostConfig, RiskConfig, Trade, ...
│   ├── strategy.py      Strategy base, Context, BarWindow (the no-lookahead guarantee)
│   ├── indicators.py    causal SMA/EMA/ATR/RSI/Donchian/z-score
│   ├── costs.py         fill pricing, commission, swap accrual
│   ├── engine.py        the bar loop
│   ├── metrics.py       performance stats + caveats()
│   ├── validation.py    walk-forward, permutation, Monte Carlo, deflated Sharpe
│   ├── data.py          CSV/MT5 loaders + synthetic generators
│   ├── report.py        ASCII report with equity curve and underwater plot
│   └── strategies/      donchian, mean reversion, and the baselines
├── mt5/
│   ├── HonestEA.mq5     MQL5 forward-test EA (refuses to start on a live account)
│   └── README.md        demo-first protocol
├── tests/               unittest suite; run: python3 -m unittest discover -s tests -t .
└── SPEC.md              the build contract
```

## Getting real data

MetaTrader 5 → **View → Symbols → Bars/Ticks → Export**. Load with
`load_mt5_csv()`. MT5 exports **bid** prices, so set
`CostConfig(price_is_mid=False)` or you will double-count half the spread.

Get your actual costs from your account, not from a marketing page: the
contract specifications for your symbol, your commission tier, and — best of
all — a real statement showing fills. Guessing your costs is the most common
way a backtest flatters itself.

## The honest workflow

1. Backtest in Python. Assume the result is too good.
2. Re-run at `--stress 2.0`. If the edge dies, stop here. Most do.
3. Compare against `RandomEntry` and against `synthetic_random_walk` data.
4. Walk-forward and permutation test. Deflate the Sharpe by the number of
   configurations you tried — count *all* of them, including the ones you
   abandoned.
5. Forward-test on an IC Markets **demo** account with `mt5/HonestEA.mq5` for
   two to three months, spanning at least one high-volatility event (NFP, CPI,
   a central bank decision).
6. Compare demo fills against your Python assumptions. Demo accounts fill
   better than live ones. Widen your cost model to match reality, then re-run
   step 1.
7. Only then, real money, at the smallest size your broker allows, for months
   before sizing up.

Most strategies die at step 2. That is the tool working.

---

## Limitations (read these)

- One position at a time. No pyramiding, no hedging, no portfolio of symbols.
- No partial fills, requotes, or rejected orders.
- Bar data hides the intrabar path. The engine's pessimistic tie-breaking is a
  mitigation, not a fix. Tick data is strictly better, and still not reality.
- Account currency is assumed equal to the quote currency (correct for EURUSD
  on a USD account; an approximation otherwise).
- Swap is modelled as a fixed points-per-night value. Real swaps change daily.
- No slippage model that widens with volatility or size — it's a flat
  configurable constant, which is optimistic during news.

## Disclaimer

This is research software for evaluating trading rules against historical data.
It is not financial advice, it is not a recommendation to trade, and it makes
no claim about future returns. Historical performance — especially simulated
historical performance — does not predict future results. Leveraged CFD and
forex trading loses money for the substantial majority of retail accounts; your
broker is required to publish that percentage, and it is worth reading before
you write a single strategy. Trade only money you can afford to lose entirely.
