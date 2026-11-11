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
    consistency_threshold=0.50
):
    """
    Runs multi-path Monte Carlo simulation across historical daily P&Ls.
    Returns:
      - Pass Rate %
      - Fail Rate % (DD breach vs DLL breach)
      - Median Days to Pass
      - Payout Rate % (probability of reaching >= 1 payout)
      - Expected Payouts ($)
      - Net Expected Value per Evaluation Challenge ($)
    """
    if len(trades_df) == 0:
        return {}
        
    # Aggregate trades by date to get realistic daily P&L sequence
    daily_pnls = trades_df.groupby('date')['net_pnl'].sum().values
    if len(daily_pnls) < 10:
        return {}
        
    np.random.seed(42)
    n_days = len(daily_pnls)
    
    eval_results = []
    funded_results = []
    
    for _ in range(num_simulations):
        # Sample with block replacement / random start index
        start_idx = np.random.randint(0, max(1, n_days - 30))
        # Draw a synthetic or shuffled block of trading days
        sim_days = np.random.choice(daily_pnls, size=min(120, n_days), replace=True)
        
        # 1. Evaluation Simulation
        res_eval = simulate_prop_evaluation_single_path(
            sim_days,
            rules,
            risk_scaler=eval_risk_scaler,
            consistency_threshold=consistency_threshold
        )
        eval_results.append(res_eval)
        
        # 2. If Passed, Simulate Funded Stage
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
    
    passed_mask = eval_df['status'] == 'PASSED'
    pass_rate = passed_mask.mean() * 100.0
    
    blown_dd_rate = (eval_df['reason'] == 'MAX_DRAWDOWN').mean() * 100.0
    blown_dll_rate = (eval_df['reason'] == 'DAILY_LOSS_LIMIT').mean() * 100.0
    soft_lockout_rate = (eval_df['soft_lockouts'] > 0).mean() * 100.0 if 'soft_lockouts' in eval_df.columns else 0.0
    
    passed_days = eval_df.loc[passed_mask, 'days']
    median_days_to_pass = passed_days.median() if len(passed_days) > 0 else 0
    p90_days_to_pass = passed_days.quantile(0.90) if len(passed_days) > 0 else 0
    
    payout_mask = funded_df['payouts_count'] > 0
    payout_rate = payout_mask.mean() * 100.0 # of all challenges
    funded_payout_rate = (funded_df.loc[passed_mask, 'payouts_count'] > 0).mean() * 100.0 if passed_mask.sum() > 0 else 0
    
    avg_total_payout = funded_df['total_payouts_usd'].mean()
    fee = rules.get("challenge_fee", 49.0)
    net_ev_per_challenge = avg_total_payout - fee
    
    return {
        'total_simulations': num_simulations,
        'pass_rate_pct': pass_rate,
        'blown_max_dd_pct': blown_dd_rate,
        'blown_dll_pct': blown_dll_rate,
        'soft_lockout_pct': soft_lockout_rate,
        'median_days_to_pass': median_days_to_pass,
        'p90_days_to_pass': p90_days_to_pass,
        'payout_rate_overall_pct': payout_rate,
        'funded_stage_payout_rate_pct': funded_payout_rate,
        'avg_payouts_usd': avg_total_payout,
        'challenge_fee': fee,
        'net_ev_per_challenge': net_ev_per_challenge,
        'eval_df': eval_df,
        'funded_df': funded_df
    }

def calculate_risk_sensitivity_matrix(trades_df, rules, risk_multipliers=[0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]):
    """
    Evaluates Pass Rate, Payout Rate, and Net EV across different risk scalers.
    """
    matrix = []
    for r in risk_multipliers:
        sim = run_monte_carlo_prop_simulation(
            trades_df,
            rules,
            eval_risk_scaler=r,
            funded_risk_scaler=max(0.25, r * 0.5), # Scale down at funded
            num_simulations=500
        )
        if sim:
            matrix.append({
                'Risk Scaler': f"{r:.2f}x",
                'Pass Rate (%)': f"{sim['pass_rate_pct']:.1f}%",
                'Median Days': f"{sim['median_days_to_pass']:.0f}",
                'Funded Payout Rate (%)': f"{sim['funded_stage_payout_rate_pct']:.1f}%",
                'Avg Payout ($)': f"${sim['avg_payouts_usd']:,.2f}",
                'Net EV per Challenge ($)': f"${sim['net_ev_per_challenge']:+,.2f}"
            })
    return pd.DataFrame(matrix)
