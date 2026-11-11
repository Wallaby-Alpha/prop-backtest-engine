import pandas as pd
import numpy as np

PROP_PRESETS = {
    "Topstep $50k": {
        "account_size": 50000,
        "profit_target": 3000,
        "max_drawdown": 2000,
        "dd_type": "eod_trailing", # End-of-day trailing
        "daily_loss_limit": 1000,
        "dll_type": "soft",        # Soft daily lockout (auto-flattens position, does NOT blow account)
        "min_days": 1,
        "challenge_fee": 49.0,
        "payout_buffer": 2000,     # Must maintain balance above $52,000 for payout
        "payout_amount": 1000
    },
    "Topstep $100k": {
        "account_size": 100000,
        "profit_target": 6000,
        "max_drawdown": 3000,
        "dd_type": "eod_trailing",
        "daily_loss_limit": 2000,
        "dll_type": "soft",
        "min_days": 1,
        "challenge_fee": 99.0,
        "payout_buffer": 3000,
        "payout_amount": 1500
    },
    "Topstep $150k": {
        "account_size": 150000,
        "profit_target": 9000,
        "max_drawdown": 4500,
        "dd_type": "eod_trailing",
        "daily_loss_limit": 3000,
        "dll_type": "soft",
        "min_days": 1,
        "challenge_fee": 149.0,
        "payout_buffer": 4500,
        "payout_amount": 2000
    },
    "Apex $50k": {
        "account_size": 50000,
        "profit_target": 3000,
        "max_drawdown": 2500,
        "dd_type": "intraday_trailing", # Real-time peak trailing
        "daily_loss_limit": None,       # No daily loss limit
        "dll_type": None,
        "min_days": 1,
        "challenge_fee": 35.0,
        "payout_buffer": 2600,
        "payout_amount": 1000
    },
    "Static Drawdown $50k": {
        "account_size": 50000,
        "profit_target": 3000,
        "max_drawdown": 2000,
        "dd_type": "static",            # Fixed at $48,000
        "daily_loss_limit": 1000,
        "dll_type": "soft",
        "min_days": 1,
        "challenge_fee": 50.0,
        "payout_buffer": 2000,
        "payout_amount": 1000
    }
}

def simulate_prop_evaluation_single_path(
    daily_pnls,
    rules,
    risk_scaler=1.0,
    consistency_threshold=0.50 # e.g. no single day > 50% of total profit
):
    """
    Simulates a single evaluation attempt given a sequence of daily P&Ls.
    
    rules dict:
      - account_size: float
      - profit_target: float
      - max_drawdown: float
      - dd_type: 'eod_trailing', 'intraday_trailing', 'static'
      - daily_loss_limit: float or None
      - dll_type: 'soft' (lockout for remainder of day) or 'hard' (instant account termination)
      - min_days: int
    """
    start_bal = rules["account_size"]
    target = rules["profit_target"]
    max_dd = rules["max_drawdown"]
    dd_type = rules.get("dd_type", "eod_trailing")
    dll = rules.get("daily_loss_limit", None)
    dll_type = rules.get("dll_type", "soft")
    min_days = rules.get("min_days", 1)
    
    current_bal = start_bal
    peak_bal = start_bal
    # Trailing DD threshold starts at start_bal - max_dd
    dd_threshold = start_bal - max_dd
    
    days_traded = 0
    day_gains = []
    soft_lockout_count = 0
    
    for day_pnl in daily_pnls:
        scaled_pnl = day_pnl * risk_scaler
        days_traded += 1
        
        # Check Daily Loss Limit (DLL)
        if dll is not None and scaled_pnl <= -dll:
            if dll_type == "hard":
                return {
                    'status': 'BLOWN',
                    'reason': 'DAILY_LOSS_LIMIT',
                    'days': days_traded,
                    'final_bal': current_bal + scaled_pnl,
                    'peak_profit': peak_bal - start_bal,
                    'soft_lockouts': soft_lockout_count
                }
            else:
                # Topstep Soft Lockout: auto-flatten position, cap daily loss at DLL, resume next day
                scaled_pnl = -float(dll)
                soft_lockout_count += 1
                
        day_gains.append(scaled_pnl)
        current_bal += scaled_pnl
        if current_bal > peak_bal:
            peak_bal = current_bal
            
        # Update trailing drawdown threshold
        if dd_type == 'eod_trailing':
            new_thresh = peak_bal - max_dd
            # Topstep locks trailing DD at starting balance + $100 once reached
            if new_thresh > start_bal + 100:
                new_thresh = start_bal + 100
            if new_thresh > dd_threshold:
                dd_threshold = new_thresh
        elif dd_type == 'intraday_trailing':
            new_thresh = peak_bal - max_dd
            if new_thresh > dd_threshold:
                dd_threshold = new_thresh
        elif dd_type == 'static':
            dd_threshold = start_bal - max_dd
            
        # Check Max Drawdown Breach
        if current_bal <= dd_threshold:
            return {
                'status': 'BLOWN',
                'reason': 'MAX_DRAWDOWN',
                'days': days_traded,
                'final_bal': current_bal,
                'peak_profit': peak_bal - start_bal,
                'soft_lockouts': soft_lockout_count
            }
            
        # Check Profit Target Hit
        if (current_bal - start_bal) >= target and days_traded >= min_days:
            # Check consistency rule if required
            tot_profit = current_bal - start_bal
            max_single_day = max(day_gains) if len(day_gains) > 0 else 0
            if consistency_threshold is not None and tot_profit > 0:
                if (max_single_day / tot_profit) > consistency_threshold:
                    continue # keep trading until consistency met
                    
            return {
                'status': 'PASSED',
                'reason': 'TARGET_REACHED',
                'days': days_traded,
                'final_bal': current_bal,
                'peak_profit': tot_profit,
                'soft_lockouts': soft_lockout_count
            }
            
    return {
        'status': 'ONGOING',
        'reason': 'RUN_ENDED',
        'days': days_traded,
        'final_bal': current_bal,
        'peak_profit': peak_bal - start_bal,
        'soft_lockouts': soft_lockout_count
    }

def simulate_funded_stage(
    daily_pnls,
    rules,
    funded_risk_scaler=0.50, # Dropping risk in funded stage to protect buffer
    max_days=250
):
    """
    Simulates a Funded Account that passed evaluation.
    Evaluates how many payouts are captured before blowing the account.
    """
    start_bal = rules["account_size"]
    max_dd = rules["max_drawdown"]
    dll = rules.get("daily_loss_limit", None)
    buffer_req = rules.get("payout_buffer", max_dd)
    payout_amt = rules.get("payout_amount", 1000)
    
    # In funded account, you start with the base account balance
    current_bal = start_bal
    peak_bal = start_bal
    dd_threshold = start_bal - max_dd
    
    total_payouts = 0
    payouts_count = 0
    days_held = 0
    days_since_payout = 0
    
    dll_type = rules.get("dll_type", "soft")
    
    for day_pnl in daily_pnls[:max_days]:
        scaled_pnl = day_pnl * funded_risk_scaler
        days_held += 1
        days_since_payout += 1
        
        if dll is not None and scaled_pnl <= -dll:
            if dll_type == "hard":
                return {
                    'status': 'BLOWN',
                    'reason': 'DAILY_LOSS_LIMIT',
                    'payouts_count': payouts_count,
                    'total_payouts_usd': total_payouts,
                    'days_survived': days_held
                }
            else:
                scaled_pnl = -float(dll)
            
        current_bal += scaled_pnl
        if current_bal > peak_bal:
            peak_bal = current_bal
            
        new_thresh = peak_bal - max_dd
        if new_thresh > start_bal + 100:
            new_thresh = start_bal + 100
        if new_thresh > dd_threshold:
            dd_threshold = new_thresh
            
        if current_bal <= dd_threshold:
            return {
                'status': 'BLOWN',
                'reason': 'MAX_DRAWDOWN',
                'payouts_count': payouts_count,
                'total_payouts_usd': total_payouts,
                'days_survived': days_held
            }
            
        # Check Payout Request Eligibility (every 10-15 trading days above buffer)
        profit_above_start = current_bal - start_bal
        if profit_above_start >= (buffer_req + payout_amt) and days_since_payout >= 10:
            current_bal -= payout_amt
            total_payouts += payout_amt
            payouts_count += 1
            days_since_payout = 0
            # Reset peak balance after payout deduction
            peak_bal = current_bal
            
    return {
        'status': 'ACTIVE',
        'payouts_count': payouts_count,
        'total_payouts_usd': total_payouts,
        'days_survived': days_held
    }

def run_monte_carlo_prop_simulation(
    trades_df,
    rules,
    eval_risk_scaler=1.0,
    funded_risk_scaler=0.5,
    num_simulations=1000,
    consistency_threshold=0.50,
    all_sessions=None,
    challenge_window_sessions=60 # 60 trading days = approx 3 calendar months
):
    """
    Runs multi-path Monte Carlo simulation with calendar-accurate cadence.
    Distinguishes between:
      1. Realistic Challenge Window Pass Rate (e.g. 60 market days / 3 months)
      2. Unbounded Pass Rate (if renewed indefinitely)
      3. Trade Frequency (trades per month & days per trade)
      4. Estimated Subscription Fees needed to pass
    """
    if len(trades_df) == 0:
        return {}
        
    daily_pnls = trades_df.groupby('date')['net_pnl'].sum().values
    if len(daily_pnls) < 5:
        return {}
        
    np.random.seed(42)
    n_days = len(daily_pnls)
    
    # Calendar Session Alignment
    if all_sessions is not None and len(all_sessions) > 0:
        full_daily_series = trades_df.groupby('date')['net_pnl'].sum().reindex(all_sessions, fill_value=0.0).values
        total_market_sessions = len(all_sessions)
    else:
        full_daily_series = daily_pnls
        total_market_sessions = max(len(daily_pnls), 750)
        
    trades_count = len(trades_df)
    trades_per_month = (trades_count / max(1, total_market_sessions)) * 21.0 # 21 sessions per month
    cadence_days_per_trade = total_market_sessions / max(1, trades_count)
    
    eval_results = []
    funded_results = []
    window_eval_results = []
    
    for _ in range(num_simulations):
        # 1. Realistic Challenge Window Simulation (sample contiguous 60 market sessions)
        if len(full_daily_series) > challenge_window_sessions:
            w_start = np.random.randint(0, len(full_daily_series) - challenge_window_sessions)
            window_days = full_daily_series[w_start : w_start + challenge_window_sessions]
            res_w = simulate_prop_evaluation_single_path(
                window_days,
                rules,
                risk_scaler=eval_risk_scaler,
                consistency_threshold=consistency_threshold
            )
            window_eval_results.append(res_w)
        else:
            window_eval_results.append({'status': 'ONGOING', 'reason': 'SHORT_DATA'})
            
        # 2. Unbounded Active-Days Simulation
        sim_days = np.random.choice(daily_pnls, size=min(120, n_days), replace=True)
        res_eval = simulate_prop_evaluation_single_path(
            sim_days,
            rules,
            risk_scaler=eval_risk_scaler,
            consistency_threshold=consistency_threshold
        )
        eval_results.append(res_eval)
        
        # 3. Funded Stage Simulation
        if res_eval['status'] == 'PASSED':
            funded_days = np.random.choice(daily_pnls, size=200, replace=True)
            res_funded = simulate_funded_stage(
                funded_days,
                rules,
                funded_risk_scaler=funded_risk_scaler
            )
            funded_results.append(res_funded)
        else:
            funded_results.append({'status': 'DID_NOT_PASS', 'payouts_count': 0, 'total_payouts_usd': 0, 'days_survived': 0})
            
    eval_df = pd.DataFrame(eval_results)
    funded_df = pd.DataFrame(funded_results)
    window_df = pd.DataFrame(window_eval_results)
    
    # Metrics
    passed_mask = eval_df['status'] == 'PASSED'
    unbounded_pass_rate = passed_mask.mean() * 100.0
    
    window_pass_mask = window_df['status'] == 'PASSED'
    realistic_challenge_pass_rate = window_pass_mask.mean() * 100.0
    
    blown_dd_rate = (eval_df['reason'] == 'MAX_DRAWDOWN').mean() * 100.0
    blown_dll_rate = (eval_df['reason'] == 'DAILY_LOSS_LIMIT').mean() * 100.0
    soft_lockout_rate = (eval_df['soft_lockouts'] > 0).mean() * 100.0 if 'soft_lockouts' in eval_df.columns else 0.0
    
    passed_days = eval_df.loc[passed_mask, 'days']
    median_trades_to_pass = passed_days.median() if len(passed_days) > 0 else 0
    p90_trades_to_pass = passed_days.quantile(0.90) if len(passed_days) > 0 else 0
    
    # Real-world calendar time & cost modeling
    est_calendar_months_to_pass = median_trades_to_pass / max(0.05, trades_per_month)
    fee = rules.get("challenge_fee", 49.0)
    est_subscription_cost = max(fee, est_calendar_months_to_pass * fee)
    
    payout_mask = funded_df['payouts_count'] > 0
    payout_rate = payout_mask.mean() * 100.0
    funded_payout_rate = (funded_df.loc[passed_mask, 'payouts_count'] > 0).mean() * 100.0 if passed_mask.sum() > 0 else 0
    
    avg_total_payout = funded_df['total_payouts_usd'].mean()
    net_ev_per_challenge = avg_total_payout - est_subscription_cost
    
    return {
        'total_simulations': num_simulations,
        'unbounded_pass_rate_pct': unbounded_pass_rate,
        'realistic_challenge_pass_rate_pct': realistic_challenge_pass_rate,
        'trades_per_month': trades_per_month,
        'cadence_days_per_trade': cadence_days_per_trade,
        'est_calendar_months_to_pass': est_calendar_months_to_pass,
        'est_subscription_cost_usd': est_subscription_cost,
        'blown_max_dd_pct': blown_dd_rate,
        'blown_dll_pct': blown_dll_rate,
        'soft_lockout_pct': soft_lockout_rate,
        'median_days_to_pass': median_trades_to_pass,
        'p90_days_to_pass': p90_trades_to_pass,
        'payout_rate_overall_pct': payout_rate,
        'funded_stage_payout_rate_pct': funded_payout_rate,
        'avg_payouts_usd': avg_total_payout,
        'challenge_fee': fee,
        'net_ev_per_challenge': net_ev_per_challenge,
        'eval_df': eval_df,
        'funded_df': funded_df,
        'window_df': window_df
    }

def calculate_risk_sensitivity_matrix(trades_df, rules, all_sessions=None, risk_multipliers=[0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]):
    """
    Evaluates Pass Rate, Cadence, and Net EV across different risk scalers.
    """
    matrix = []
    base_risk = trades_df['actual_risk_usd'].mean() if ('actual_risk_usd' in trades_df.columns and len(trades_df) > 0) else 250.0
    
    for r in risk_multipliers:
        sim = run_monte_carlo_prop_simulation(
            trades_df,
            rules,
            eval_risk_scaler=r,
            funded_risk_scaler=max(0.25, r * 0.5),
            num_simulations=500,
            all_sessions=all_sessions
        )
        if sim:
            matrix.append({
                'Risk Scaler': f"{r:.2f}x (${base_risk * r:,.0f}/trade)",
                '60-Day Pass Rate': f"{sim['realistic_challenge_pass_rate_pct']:.1f}%",
                'Open-Ended Pass': f"{sim['unbounded_pass_rate_pct']:.1f}%",
                'Est. Months': f"{sim['est_calendar_months_to_pass']:.1f} mos",
                'Funded Payout Rate': f"{sim['funded_stage_payout_rate_pct']:.1f}%",
                'Avg Payout': f"${sim['avg_payouts_usd']:,.2f}",
                'Net EV': f"${sim['net_ev_per_challenge']:+,.2f}"
            })
    return pd.DataFrame(matrix)
