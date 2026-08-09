//+------------------------------------------------------------------+
//|                                                     HonestEA.mq5 |
//|                                                                  |
//| Donchian channel breakout, MQL5 side of the tradelab package.    |
//|                                                                  |
//| WHY THIS FILE EXISTS                                             |
//|                                                                  |
//| It is not here to make money. It is here to answer one question  |
//| that a Python backtest structurally cannot answer: what does a   |
//| real broker actually do with these orders?                       |
//|                                                                  |
//| tradelab models spread, commission, swap and slippage as         |
//| configured constants. A live feed does not. Spread on EURUSD     |
//| goes from 0.1 pip to 3 pips in the second before a CPI print,    |
//| stop fills slip by amounts no flat constant describes, and       |
//| orders get rejected for reasons the engine has no concept of.    |
//| The only way to find out how far the model is from reality is to |
//| run the same rules against a live price feed and diff the fills. |
//| That is this EA's entire job.                                    |
//|                                                                  |
//| The logic mirrors tradelab/strategies/donchian.py exactly:       |
//|   * entry when the CLOSED bar closes beyond the N-bar channel    |
//|     measured over the N bars BEFORE it (the channel excludes the |
//|     signal bar - see indicators.donchian for why that is not a   |
//|     detail),                                                     |
//|   * initial stop at atr_stop_mult ATRs from the signal close,    |
//|   * position size derived from that stop distance,               |
//|   * exit on the opposite exit-channel break,                     |
//|   * optional trailing stop ratcheted to the exit channel, which  |
//|     may only ever move towards the market.                       |
//|                                                                  |
//| Any divergence between this file and donchian.py makes the       |
//| forward test meaningless, because you would be comparing two     |
//| different strategies. If you change one, change the other.       |
//|                                                                  |
//| The safety code below - the live-account refusal, the daily loss |
//| killswitch, the spread filter, the risk-derived sizing - is not  |
//| boilerplate wrapped around the interesting part. On a forward    |
//| test it IS the interesting part. A breakout EA left unattended   |
//| on a widening feed with a sizing bug is how accounts end.        |
//+------------------------------------------------------------------+
#property copyright "tradelab"
#property version   "1.00"
#property description "Donchian breakout forward-test EA. Refuses to start on a live account unless explicitly permitted. Research tool, not a money printer."

#include <Trade/Trade.mqh>

//--- safety -----------------------------------------------------------------
input group "Safety"
input bool   AllowLiveTrading   = false;   // Permit running on a REAL account (leave false)
input double RiskPercent        = 0.5;     // Risk per trade, percent of equity
input double MaxLots            = 5.0;     // Hard cap on position size
input double MaxDailyLossPct    = 3.0;     // Halt for the day after this loss, percent of day-open equity
input int    MaxSpreadPoints    = 20;      // Skip entries when spread exceeds this, in points
input int    MinStopPoints      = 50;      // Refuse to size a stop closer than this, in points
input int    MaxSlippagePoints  = 20;      // Deviation allowed on market orders, in points

//--- strategy ---------------------------------------------------------------
input group "Strategy"
input int    EntryLookback      = 20;      // Donchian entry channel, bars (excludes the signal bar)
input int    ExitLookback       = 10;      // Donchian exit channel, bars
input int    AtrPeriod          = 14;      // ATR period for the initial stop
input double AtrStopMult        = 2.0;     // Initial stop distance, in ATRs
input bool   UseTrailingStop    = true;    // Ratchet the stop to the exit channel
input bool   LongOnly           = false;   // Disable shorts

//--- housekeeping -----------------------------------------------------------
input group "Housekeeping"
input long   MagicNumber        = 20240719; // Identifies this EA's positions
input bool   VerboseLog         = true;     // Log skipped signals and their reasons

//--- state ------------------------------------------------------------------
CTrade    g_trade;
int       g_atr_handle    = INVALID_HANDLE;
datetime  g_last_bar_time = 0;      // time of the last bar this EA acted on
int       g_halt_day      = -1;     // day-of-year currently under a killswitch halt
int       g_equity_day    = -1;     // day-of-year g_day_open_equity belongs to
double    g_day_open_equity = 0.0;

//+------------------------------------------------------------------+
//| Refuse to start where we should not be running at all.           |
//|                                                                  |
//| ACCOUNT_TRADE_MODE_REAL is the only signal MT5 gives that money  |
//| is real. Checking it here, before a single handle is created, is |
//| the difference between "I attached the wrong chart" being an     |
//| annoyance and it being a loss. The Strategy Tester reports       |
//| ACCOUNT_TRADE_MODE_DEMO, so this check never blocks a backtest.  |
//+------------------------------------------------------------------+
int OnInit()
  {
   const ENUM_ACCOUNT_TRADE_MODE mode =
      (ENUM_ACCOUNT_TRADE_MODE)AccountInfoInteger(ACCOUNT_TRADE_MODE);

   if(mode == ACCOUNT_TRADE_MODE_REAL && !AllowLiveTrading)
     {
      Print("HonestEA: REFUSING TO START.");
      Print("  This is a LIVE account (ACCOUNT_TRADE_MODE_REAL), account #",
            AccountInfoInteger(ACCOUNT_LOGIN), " at ", AccountInfoString(ACCOUNT_COMPANY), ".");
      Print("  AllowLiveTrading is false, which is the default and the correct setting");
      Print("  until this strategy has been forward-tested on a demo account for at");
      Print("  least two to three months across a high-volatility event, and the demo");
      Print("  fills have been compared against the Python cost model.");
      Print("  If you have genuinely done that and accept the risk, set AllowLiveTrading");
      Print("  to true deliberately. Nothing in this EA predicts a profit.");
      return(INIT_FAILED);
     }

   if(mode == ACCOUNT_TRADE_MODE_REAL)
      Print("HonestEA: running on a LIVE account because AllowLiveTrading was set true. ",
            "Risk per trade ", DoubleToString(RiskPercent, 2), "%.");

   if(!ValidateInputs())
      return(INIT_PARAMETERS_INCORRECT);

   g_atr_handle = iATR(_Symbol, PERIOD_CURRENT, AtrPeriod);
   if(g_atr_handle == INVALID_HANDLE)
     {
      Print("HonestEA: iATR failed to create a handle, error ", GetLastError());
      return(INIT_FAILED);
     }

   g_trade.SetExpertMagicNumber((ulong)MagicNumber);
   g_trade.SetDeviationInPoints((ulong)MaxSlippagePoints);
   g_trade.SetTypeFillingBySymbol(_Symbol);
   g_trade.LogLevel(LOG_LEVEL_ERRORS);

   g_last_bar_time = 0;
   g_halt_day      = -1;
   g_equity_day    = -1;

   PrintFormat("HonestEA on %s %s: entry=%d exit=%d atr=%d x%.2f trail=%s long_only=%s "
               "risk=%.2f%% daily_loss_halt=%.2f%% max_spread=%d pts",
               _Symbol, EnumToString((ENUM_TIMEFRAMES)Period()),
               EntryLookback, ExitLookback, AtrPeriod, AtrStopMult,
               (UseTrailingStop ? "true" : "false"), (LongOnly ? "true" : "false"),
               RiskPercent, MaxDailyLossPct, MaxSpreadPoints);
   return(INIT_SUCCEEDED);
  }

//+------------------------------------------------------------------+
//| Release the indicator handle. Leaking handles across             |
//| re-initialisations (a timeframe change is one) eventually        |
//| exhausts the terminal's indicator pool and CopyBuffer starts     |
//| failing for reasons that look like a data problem.               |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   if(g_atr_handle != INVALID_HANDLE)
     {
      IndicatorRelease(g_atr_handle);
      g_atr_handle = INVALID_HANDLE;
     }
   Print("HonestEA: deinit, reason ", reason);
  }

//+------------------------------------------------------------------+
//| Input sanity. Failing here is far cheaper than discovering a     |
//| zero-width stop by dividing by it during position sizing.        |
//+------------------------------------------------------------------+
bool ValidateInputs()
  {
   bool ok = true;
   if(EntryLookback < 1 || ExitLookback < 1 || AtrPeriod < 1)
     { Print("HonestEA: lookbacks and AtrPeriod must be >= 1."); ok = false; }
   if(AtrStopMult <= 0.0)
     { Print("HonestEA: AtrStopMult must be positive; a zero-width stop cannot be sized."); ok = false; }
   if(RiskPercent <= 0.0 || RiskPercent > 5.0)
     { Print("HonestEA: RiskPercent must be in (0, 5]. Above ~1% per trade, a normal "
             "losing streak for a breakout system is an account-ending event."); ok = false; }
   if(MaxDailyLossPct <= 0.0 || MaxDailyLossPct > 50.0)
     { Print("HonestEA: MaxDailyLossPct must be in (0, 50]."); ok = false; }
   if(MaxSpreadPoints <= 0)
     { Print("HonestEA: MaxSpreadPoints must be positive."); ok = false; }
   if(MinStopPoints < 0)
     { Print("HonestEA: MinStopPoints must be >= 0."); ok = false; }
   if(MaxLots <= 0.0)
     { Print("HonestEA: MaxLots must be positive."); ok = false; }
   return(ok);
  }

//+------------------------------------------------------------------+
//| Tick handler.                                                    |
//|                                                                  |
//| Two different clocks run here on purpose.                        |
//|                                                                  |
//| The killswitch is evaluated on every tick, because the loss it   |
//| guards against does not wait for the bar to close. Everything    |
//| else runs once per completed bar, because the strategy is        |
//| defined on closed bars and acting intrabar would be a different  |
//| strategy from the one that was backtested - one whose results    |
//| the Python numbers say nothing about.                            |
//+------------------------------------------------------------------+
void OnTick()
  {
   if(!EnforceDailyLossLimit())
      return;

   const datetime bar_time = (datetime)SeriesInfoInteger(_Symbol, PERIOD_CURRENT, SERIES_LASTBAR_DATE);
   if(bar_time == 0 || bar_time == g_last_bar_time)
      return;                       // no new bar has formed yet

   if(Bars(_Symbol, PERIOD_CURRENT) < RequiredBars())
     {
      g_last_bar_time = bar_time;
      if(VerboseLog)
         Print("HonestEA: waiting for history, need ", RequiredBars(), " bars, have ",
               Bars(_Symbol, PERIOD_CURRENT));
      return;
     }

   g_last_bar_time = bar_time;
   ProcessClosedBar();
  }

//+------------------------------------------------------------------+
//| Bars of history needed before the first decision. Mirrors        |
//| DonchianBreakout.warmup, plus the shift for the excluded bar.    |
//+------------------------------------------------------------------+
int RequiredBars()
  {
   int longest = EntryLookback;
   if(ExitLookback > longest) longest = ExitLookback;
   if(AtrPeriod > longest)    longest = AtrPeriod;
   return(longest + AtrPeriod + 4);
  }

//+------------------------------------------------------------------+
//| One decision, taken on the bar that just closed (shift 1) and    |
//| executed now, at what is effectively the open of the forming bar |
//| (shift 0). That is the same one-bar delay the Python engine      |
//| enforces, and it is the reason the entry price is always worse   |
//| than the signal price.                                           |
//+------------------------------------------------------------------+
void ProcessClosedBar()
  {
   double exit_upper = 0.0, exit_lower = 0.0;
   if(!DonchianChannel(ExitLookback, exit_upper, exit_lower))
      return;

   if(HasOpenPosition())
     {
      ManagePosition(exit_upper, exit_lower);
      return;
     }

   double entry_upper = 0.0, entry_lower = 0.0;
   if(!DonchianChannel(EntryLookback, entry_upper, entry_lower))
      return;

   TryEntry(entry_upper, entry_lower);
  }

//+------------------------------------------------------------------+
//| Donchian channel over the `period` bars ENDING AT SHIFT 2.       |
//|                                                                  |
//| Shift 1 is the bar that just closed and produces the signal; it  |
//| is deliberately not in the window. A breakout measured against a |
//| level that includes the bar doing the breaking is not a signal,  |
//| it is a definition - `close <= high <= max(highs)` makes a       |
//| strict test structurally unable to fire, and a non-strict one    |
//| reduces to "is this the highest bar of the last N", which is     |
//| true by construction. The Python side documents the measured     |
//| consequences at length in indicators.donchian.                   |
//+------------------------------------------------------------------+
bool DonchianChannel(const int period, double &upper, double &lower)
  {
   double highs[], lows[];
   ArraySetAsSeries(highs, true);
   ArraySetAsSeries(lows,  true);

   const int got_h = CopyHigh(_Symbol, PERIOD_CURRENT, 2, period, highs);
   const int got_l = CopyLow(_Symbol, PERIOD_CURRENT, 2, period, lows);
   if(got_h < period || got_l < period)
     {
      if(VerboseLog)
         Print("HonestEA: CopyHigh/CopyLow returned ", got_h, "/", got_l,
               " of ", period, " bars, error ", GetLastError(), " - skipping this bar.");
      return(false);
     }

   const int hi = ArrayMaximum(highs, 0, period);
   const int lo = ArrayMinimum(lows, 0, period);
   if(hi < 0 || lo < 0)
      return(false);

   upper = highs[hi];
   lower = lows[lo];
   return(true);
  }

//+------------------------------------------------------------------+
//| ATR of the bar that just closed.                                 |
//+------------------------------------------------------------------+
bool CurrentAtr(double &value)
  {
   double buf[];
   ArraySetAsSeries(buf, true);
   const int got = CopyBuffer(g_atr_handle, 0, 1, 1, buf);
   if(got < 1)
     {
      if(VerboseLog)
         Print("HonestEA: CopyBuffer on the ATR handle returned ", got,
               ", error ", GetLastError(), " - skipping this bar.");
      return(false);
     }
   if(buf[0] <= 0.0 || !MathIsValidNumber(buf[0]))
      return(false);
   value = buf[0];
   return(true);
  }

//+------------------------------------------------------------------+
//| Entry: close of the signal bar beyond the entry channel.         |
//+------------------------------------------------------------------+
void TryEntry(const double upper, const double lower)
  {
   if(IsHalted())
      return;

   const double close = iClose(_Symbol, PERIOD_CURRENT, 1);
   if(close <= 0.0)
      return;

   int direction = 0;
   if(close > upper)
      direction = 1;
   else
      if(close < lower && !LongOnly)
         direction = -1;

   if(direction == 0)
      return;

   // The spread filter sits after the signal test so the log tells you how
   // many real signals the feed's own widening cost you. On a breakout system
   // that is not a rounding error: breakouts happen when the market is moving,
   // which is exactly when the spread is worst.
   const int spread = CurrentSpreadPoints();
   if(spread > MaxSpreadPoints)
     {
      Print("HonestEA: skipping ", (direction > 0 ? "long" : "short"),
            " signal - spread ", spread, " points exceeds MaxSpreadPoints ", MaxSpreadPoints, ".");
      return;
     }

   double atr_value = 0.0;
   if(!CurrentAtr(atr_value))
      return;

   const double stop_distance = atr_value * AtrStopMult;
   const double stop_price = (direction > 0) ? close - stop_distance : close + stop_distance;

   OpenPosition(direction, stop_price, stop_distance);
  }

//+------------------------------------------------------------------+
//| Size from the stop distance and send the order.                  |
//|                                                                  |
//| The stop is anchored to the SIGNAL close, not to the price we    |
//| are about to get. If the market has already moved through it,    |
//| the trade is refused rather than opened with an instantly-hit or |
//| retroactively-widened stop - the Python strategy refuses the     |
//| same trade for the same reason.                                  |
//+------------------------------------------------------------------+
void OpenPosition(const int direction, const double stop_price, const double stop_distance)
  {
   MqlTick tick;
   if(!SymbolInfoTick(_Symbol, tick))
     {
      Print("HonestEA: SymbolInfoTick failed, error ", GetLastError(), " - no order sent.");
      return;
     }

   const double point = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
   if(point <= 0.0)
     {
      Print("HonestEA: SYMBOL_POINT is ", point, " - cannot size anything. No order sent.");
      return;
     }

   if(stop_distance / point < (double)MinStopPoints)
     {
      Print("HonestEA: skipping entry - stop distance ",
            DoubleToString(stop_distance / point, 1), " points is below MinStopPoints ",
            MinStopPoints, ". Sizing off a near-zero stop is how a 0.5% risk becomes 50%.");
      return;
     }

   const double entry = (direction > 0) ? tick.ask : tick.bid;
   if(entry <= 0.0)
      return;

   if((direction > 0 && entry <= stop_price) || (direction < 0 && entry >= stop_price))
     {
      Print("HonestEA: skipping entry - price has already moved through the stop level ",
            DoubleToString(stop_price, _Digits), " (current ", DoubleToString(entry, _Digits), ").");
      return;
     }

   // The broker forbids stops closer than SYMBOL_TRADE_STOPS_LEVEL. Sending one
   // anyway returns an invalid-stops retcode and leaves you flat, which is
   // survivable; sending the order and then failing to attach the stop is not.
   const long stops_level = SymbolInfoInteger(_Symbol, SYMBOL_TRADE_STOPS_LEVEL);
   if(MathAbs(entry - stop_price) / point < (double)stops_level)
     {
      Print("HonestEA: skipping entry - stop is inside the broker's stops level of ",
            stops_level, " points.");
      return;
     }

   const double lots = LotsForRisk(MathAbs(entry - stop_price));
   if(lots <= 0.0)
      return;

   const double sl = NormalizeDouble(stop_price, _Digits);
   const string comment = (direction > 0) ? "donchian_long" : "donchian_short";

   bool ok = false;
   if(direction > 0)
      ok = g_trade.Buy(lots, _Symbol, 0.0, sl, 0.0, comment);
   else
      ok = g_trade.Sell(lots, _Symbol, 0.0, sl, 0.0, comment);

   if(!ok || (g_trade.ResultRetcode() != TRADE_RETCODE_DONE &&
              g_trade.ResultRetcode() != TRADE_RETCODE_PLACED))
     {
      PrintFormat("HonestEA: %s %.2f lots FAILED, retcode %u (%s), broker comment '%s'.",
                  (direction > 0 ? "BUY" : "SELL"), lots,
                  g_trade.ResultRetcode(), g_trade.ResultRetcodeDescription(),
                  g_trade.ResultComment());
      return;
     }

   PrintFormat("HonestEA: %s %.2f lots at %s, stop %s (%.1f points, %.2f ATR), "
               "signal close %s, spread %d points, deal %I64u.",
               (direction > 0 ? "BUY" : "SELL"), g_trade.ResultVolume(),
               DoubleToString(g_trade.ResultPrice(), _Digits), DoubleToString(sl, _Digits),
               MathAbs(entry - sl) / point, AtrStopMult,
               DoubleToString(iClose(_Symbol, PERIOD_CURRENT, 1), _Digits),
               CurrentSpreadPoints(), g_trade.ResultDeal());
  }

//+------------------------------------------------------------------+
//| Risk-percent sizing.                                             |
//|                                                                  |
//| SYMBOL_TRADE_TICK_VALUE / SYMBOL_TRADE_TICK_SIZE gives the       |
//| account-currency value of a 1.0 price move for one lot. Deriving |
//| it this way, rather than assuming "one pip is ten dollars", is   |
//| what makes the sizing correct on a JPY cross, on gold, on an     |
//| index CFD, and on a EUR-denominated account. Getting it wrong is |
//| silent: the trades still open, they are just the wrong size, and |
//| the equity curve is the wrong shape by a constant factor.        |
//|                                                                  |
//| The volume is floored to SYMBOL_VOLUME_STEP, never rounded up.   |
//| Rounding up risks more than the stated percentage.               |
//+------------------------------------------------------------------+
double LotsForRisk(const double stop_distance)
  {
   if(stop_distance <= 0.0)
      return(0.0);

   const double tick_value = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
   const double tick_size  = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
   if(tick_value <= 0.0 || tick_size <= 0.0)
     {
      Print("HonestEA: SYMBOL_TRADE_TICK_VALUE=", tick_value, " SYMBOL_TRADE_TICK_SIZE=",
            tick_size, " - cannot convert price distance to money. No order sent.");
      return(0.0);
     }

   const double value_per_price_unit = tick_value / tick_size;   // per 1.0 lot
   const double risk_cash = AccountInfoDouble(ACCOUNT_EQUITY) * (RiskPercent / 100.0);
   if(risk_cash <= 0.0)
      return(0.0);

   double lots = risk_cash / (stop_distance * value_per_price_unit);
   lots = MathMin(lots, MaxLots);
   lots = NormaliseVolume(lots);

   if(lots <= 0.0)
     {
      Print("HonestEA: computed size rounds to zero at ", DoubleToString(RiskPercent, 2),
            "% risk over a ", DoubleToString(stop_distance, _Digits),
            " stop. The account is too small for this stop distance; skipping the trade ",
            "rather than trading the minimum lot and risking more than stated.");
      return(0.0);
     }

   // Margin check. ORDER_TYPE_BUY is used as the probe for both directions:
   // on FX the two sides require the same margin, and where they do not, the
   // buy side is the conservative one to test against.
   const double margin_free = AccountInfoDouble(ACCOUNT_MARGIN_FREE);
   double margin_needed = 0.0;
   if(OrderCalcMargin(ORDER_TYPE_BUY, _Symbol, lots, SymbolInfoDouble(_Symbol, SYMBOL_ASK), margin_needed))
     {
      if(margin_needed > margin_free)
        {
         Print("HonestEA: skipping trade - ", DoubleToString(lots, 2), " lots needs ",
               DoubleToString(margin_needed, 2), " margin, free margin is ",
               DoubleToString(margin_free, 2), ".");
         return(0.0);
        }
     }

   return(lots);
  }

//+------------------------------------------------------------------+
//| Snap a volume down onto the broker's grid.                       |
//+------------------------------------------------------------------+
double NormaliseVolume(const double lots)
  {
   double step = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
   const double vmin = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   const double vmax = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
   if(step <= 0.0)
      step = 0.01;

   double stepped = MathFloor(lots / step + 1e-9) * step;
   stepped = NormalizeDouble(stepped, 8);

   if(vmax > 0.0 && stepped > vmax)
      stepped = vmax;
   if(stepped < vmin - 1e-9)
      return(0.0);

   return(stepped);
  }

//+------------------------------------------------------------------+
//| Manage an open position: opposite-channel exit, then the trail.  |
//+------------------------------------------------------------------+
void ManagePosition(const double upper, const double lower)
  {
   if(!SelectPosition())
      return;

   const long type = PositionGetInteger(POSITION_TYPE);
   const bool is_long = (type == POSITION_TYPE_BUY);
   const double close = iClose(_Symbol, PERIOD_CURRENT, 1);
   if(close <= 0.0)
      return;

   if(is_long && close < lower)
     {
      ClosePosition("opposite channel break (long)");
      return;
     }
   if(!is_long && close > upper)
     {
      ClosePosition("opposite channel break (short)");
      return;
     }

   if(UseTrailingStop)
      Ratchet(is_long, close, upper, lower);
  }

//+------------------------------------------------------------------+
//| Move the stop towards the market, never away from it.            |
//|                                                                  |
//| The one-directional comparison is the whole safety property. A   |
//| trailing stop allowed to widen is a strategy quietly increasing  |
//| its risk after the trade has gone against it, and it will look   |
//| like an improvement in the equity curve right up until it does   |
//| not.                                                             |
//+------------------------------------------------------------------+
void Ratchet(const bool is_long, const double close, const double upper, const double lower)
  {
   const double current = PositionGetDouble(POSITION_SL);
   const double point = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
   if(point <= 0.0)
      return;

   double candidate = 0.0;
   if(is_long)
     {
      candidate = lower;
      if(candidate >= close)
         return;                       // already through the level; the exit test owns this bar
      if(current > 0.0 && candidate <= current)
         return;                       // would loosen the stop
     }
   else
     {
      candidate = upper;
      if(candidate <= close)
         return;
      if(current > 0.0 && candidate >= current)
         return;
     }

   // The broker will not accept a stop inside the stops/freeze level. Leaving
   // the existing stop in place is the safe failure here.
   MqlTick tick;
   if(!SymbolInfoTick(_Symbol, tick))
      return;
   const double market = is_long ? tick.bid : tick.ask;
   const long stops_level = SymbolInfoInteger(_Symbol, SYMBOL_TRADE_STOPS_LEVEL);
   if(MathAbs(market - candidate) / point < (double)stops_level)
      return;
   if((is_long && candidate >= market) || (!is_long && candidate <= market))
      return;

   const double sl = NormalizeDouble(candidate, _Digits);
   if(MathAbs(sl - current) < point / 2.0)
      return;                          // nothing meaningful to change

   const ulong ticket = PositionGetInteger(POSITION_TICKET);
   if(!g_trade.PositionModify(ticket, sl, PositionGetDouble(POSITION_TP)))
     {
      PrintFormat("HonestEA: trailing stop modify to %s FAILED, retcode %u (%s).",
                  DoubleToString(sl, _Digits), g_trade.ResultRetcode(),
                  g_trade.ResultRetcodeDescription());
      return;
     }

   if(VerboseLog)
      PrintFormat("HonestEA: trailed stop %s -> %s.",
                  DoubleToString(current, _Digits), DoubleToString(sl, _Digits));
  }

//+------------------------------------------------------------------+
//| Close this EA's position, logging the retcode either way.        |
//+------------------------------------------------------------------+
bool ClosePosition(const string reason)
  {
   if(!SelectPosition())
      return(false);

   const ulong ticket = PositionGetInteger(POSITION_TICKET);
   const double profit = PositionGetDouble(POSITION_PROFIT);

   if(!g_trade.PositionClose(ticket, (ulong)MaxSlippagePoints))
     {
      PrintFormat("HonestEA: close of #%I64u FAILED (%s), retcode %u (%s), comment '%s'. "
                  "The position is still open.",
                  ticket, reason, g_trade.ResultRetcode(),
                  g_trade.ResultRetcodeDescription(), g_trade.ResultComment());
      return(false);
     }

   PrintFormat("HonestEA: closed #%I64u - %s. Floating P&L at the decision was %.2f.",
               ticket, reason, profit);
   return(true);
  }

//+------------------------------------------------------------------+
//| Select this EA's position on this symbol, if any.                |
//+------------------------------------------------------------------+
bool SelectPosition()
  {
   for(int i = PositionsTotal() - 1; i >= 0; i--)
     {
      const ulong ticket = PositionGetTicket(i);   // also selects it
      if(ticket == 0)
         continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol)
         continue;
      if(PositionGetInteger(POSITION_MAGIC) != MagicNumber)
         continue;
      return(true);
     }
   return(false);
  }

bool HasOpenPosition()
  {
   return(SelectPosition());
  }

//+------------------------------------------------------------------+
//| Current spread in points, from the tick rather than from         |
//| SYMBOL_SPREAD, which some feeds report as a rounded integer that |
//| lags the quote.                                                  |
//+------------------------------------------------------------------+
int CurrentSpreadPoints()
  {
   MqlTick tick;
   const double point = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
   if(point <= 0.0 || !SymbolInfoTick(_Symbol, tick))
      return((int)SymbolInfoInteger(_Symbol, SYMBOL_SPREAD));
   if(tick.ask <= 0.0 || tick.bid <= 0.0)
      return((int)SymbolInfoInteger(_Symbol, SYMBOL_SPREAD));
   return((int)MathRound((tick.ask - tick.bid) / point));
  }

//+------------------------------------------------------------------+
//| Daily loss killswitch.                                           |
//|                                                                  |
//| Returns false when the EA should do nothing further this tick.   |
//|                                                                  |
//| The day boundary here is the BROKER SERVER day, not UTC. The     |
//| Python engine uses UTC. On IC Markets the server runs at         |
//| GMT+2/+3, so the two halt windows are offset by a couple of      |
//| hours and the trade sequences can legitimately diverge around    |
//| the boundary. That is a known, deliberate difference; the server |
//| day is what the account statement will be cut on, and the        |
//| statement is what you will be reconciling against.               |
//+------------------------------------------------------------------+
bool EnforceDailyLossLimit()
  {
   MqlDateTime now;
   TimeToStruct(TimeCurrent(), now);
   const int day = now.day_of_year;

   if(day != g_equity_day)
     {
      g_equity_day = day;
      g_day_open_equity = AccountInfoDouble(ACCOUNT_EQUITY);
      if(g_halt_day != -1 && g_halt_day != day)
        {
         Print("HonestEA: new trading day, killswitch released. Day-open equity ",
               DoubleToString(g_day_open_equity, 2), ".");
         g_halt_day = -1;
        }
     }

   if(g_halt_day == day)
      return(false);

   if(g_day_open_equity <= 0.0)
      return(true);

   const double equity = AccountInfoDouble(ACCOUNT_EQUITY);
   const double loss_fraction = (g_day_open_equity - equity) / g_day_open_equity;
   if(loss_fraction < MaxDailyLossPct / 100.0)
      return(true);

   PrintFormat("HonestEA: DAILY LOSS KILLSWITCH. Equity %.2f is %.2f%% below the day-open "
               "equity %.2f (limit %.2f%%). Flattening and standing down until the next "
               "server day.",
               equity, loss_fraction * 100.0, g_day_open_equity, MaxDailyLossPct);

   if(HasOpenPosition())
      ClosePosition("daily loss killswitch");

   g_halt_day = day;
   return(false);
  }

bool IsHalted()
  {
   MqlDateTime now;
   TimeToStruct(TimeCurrent(), now);
   return(g_halt_day == now.day_of_year);
  }
//+------------------------------------------------------------------+
