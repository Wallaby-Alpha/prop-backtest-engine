# Institutional Backtesting & Prop Firm Rules Engine (Antigravity Standards)

Whenever the user asks you to backtest, evaluate, or optimize a trading strategy in this project, **NEVER start from scratch**. You MUST strictly follow the institutional backtesting methodology and prop firm simulation standards established in this codebase:

---

## 1. Core Backtest Methodology & Zero-Lookahead Rules
1. **CME Session Rollover:**
   - Any intraday bar with Eastern Time $\ge$ 18:00 belongs to the **next business day's session** (`session_date_et`).
2. **Strict Causal Lag (No Lookahead Bias):**
   - Higher timeframe indicators (Daily EMAs, Daily High/Low, Daily Volume Profiles) must **strictly shift by 1 day** (`.shift(1)`).
   - Never use the current day's closing price, daily high/low, or today's developing volume profile to decide today's intraday entry.
3. **Execution & Real-World Friction:**
   - Default slippage: **0.50 pts per side** on NQ / MNQ (total 1.0 pt round trip = $2.00/contract on MNQ, $20.00/contract on NQ).
   - Default commissions: **$0.62 per contract per side** on MNQ ($1.24 RT), **$1.60 per contract per side** on NQ ($3.20 RT).
4. **Out-of-Sample (OOS) Holdout:**
   - Always maintain a strict chronological split (default 70% In-Sample training, 30% Out-of-Sample holdout).

---

## 2. Institutional Position Sizing (Concrete Dollar Risk)
- **Concrete Dollar Risk per Trade:** Always size positions dynamically based on each setup's stop-loss distance:
  $$\text{Contracts} = \max\left(1, \left\lfloor \frac{\text{Target Dollar Risk (\$) favours}}{\text{Stop Distance (pts)} \times \text{Multiplier} + \text{Friction}} \right\rfloor\right)$$
- Standard baseline: **$250 risk per trade** on a $50,000 account (risks 12.5% of the $2,000 max drawdown buffer).
- **Two-Stage Prop Sizing:**
  - **Evaluation Phase:** Target $250 - $350 risk/trade to reach the profit target.
  - **Funded Stage:** Cut risk in half ($125 - $175 risk/trade) to preserve the drawdown buffer and capture recurring payouts.

---

## 3. Prop Firm Simulation Standards
1. **Real-World Calendar Cadence (Anti-Time-Compression):**
   - **Never compress trade days:** Backtests must align trade results against the complete CME session calendar (`all_sessions`), including zero-trade days.
   - Always report **Trade Cadence** (trades per month and market days per trade).
2. **Dual Pass Rate Reporting:**
   - **Realistic 60-Day Challenge Pass Rate:** Probability of reaching the target within a realistic 3-month (60 market session) evaluation window.
   - **Open-Ended Pass Rate:** Probability of reaching the target if the evaluation is renewed indefinitely.
   - **Est. Calendar Months & Subscription Cost to Pass:** (e.g. at $49/month Topstep fee).
3. **Topstep $50k Rule Defaults:**
   - Account Size: $50,000
   - Profit Target: $3,000
   - Max Drawdown: $2,000 (End-of-day trailing, locks at $50,100 once reached)
   - Daily Loss Limit: $1,000 — **Enforced as a SOFT BREACH (Daily Lockout)**. Auto-flattens position, trading resumes next session. Account is NOT blown.

---

## 4. How to Run Backtests
Whenever the user requests a backtest or strategy tweak:
- Use the modular engine in `engine/` (`data_manager.py`, `strategies.py`, `execution_engine.py`, `prop_simulator.py`).
- Or execute the CLI runner: `python run_prop_backtest.py --strategy "Value Area Reclaim" --start 10:00 --end 12:00 --risk 250`.
- Present results in clean GitHub markdown tables with:
  - In-Sample vs. Out-of-Sample Net P&L, Win Rate, Profit Factor, Max Drawdown
  - Trade Cadence (trades/mo)
  - Realistic 60-day Challenge Pass Rate vs. Open-Ended Pass Rate
  - Funded Stage Payout Rate & Net Expectancy
