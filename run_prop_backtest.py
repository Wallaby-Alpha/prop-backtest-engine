"""
Institutional Prop Backtesting CLI Runner
Executes backtests with concrete dollar risk sizing and calendar-accurate prop evaluation.
Can be executed directly from command line or called programmatically by Antigravity.
"""

import argparse
import sys
import pandas as pd
import numpy as np

# Prevent Windows console cp1252 encoding crashes on emojis
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

from engine.data_manager import load_asset_data, compute_daily_volume_profiles, split_in_sample_out_sample, normalize_time_str
from engine.strategies import StrategyRegistry
from engine.execution_engine import process_trades
from engine.prop_simulator import PROP_PRESETS, run_monte_carlo_prop_simulation, calculate_risk_sensitivity_matrix
from engine.llm_parser import compile_user_prompt_to_setup

def run_backtest(
    asset="MNQ",
    timeframe="15m",
    strategy="Value Area Reclaim",
    params=None,
    dollar_risk=250.0,
    prop_preset="Topstep $50k",
    oos_ratio=0.30
):
    if params is None:
        params = {}
        
    print(f"\n=======================================================")
    print(f"🚀 Running Backtest: {strategy} on {asset} ({timeframe})")
    print(f"💵 Sizing: ${dollar_risk:,.0f} Risk per Trade | Prop Preset: {prop_preset}")
    print(f"=======================================================\n")
    
    df, spec = load_asset_data(asset, timeframe)
    profiles = compute_daily_volume_profiles(df)
    is_mask, oos_mask, split_date = split_in_sample_out_sample(df, oos_ratio=oos_ratio)
    
    strat_fn = StrategyRegistry.get(strategy)
    signals = strat_fn(df, profiles, params)
    
    if len(signals) == 0:
        print("❌ No signals generated with the provided parameters.")
        return
        
    trades, metrics = process_trades(
        signals,
        spec,
        oos_split_date=split_date,
        sizing_mode="dollar_risk",
        dollar_risk_per_trade=dollar_risk
    )
    
    all_sess = sorted(list(df['session_date_et'].unique()))
    rules = PROP_PRESETS.get(prop_preset, PROP_PRESETS["Topstep $50k"])
    sim = run_monte_carlo_prop_simulation(trades, rules, all_sessions=all_sess)
    
    # 1. Performance Overview
    m_all = metrics.get('all', {})
    m_is = metrics.get('in_sample', {})
    m_oos = metrics.get('out_sample', {})
    
    print("📊 STRATEGY PERFORMANCE:")
    print(f"  • Total Trades:       {m_all.get('trades', 0)}")
    print(f"  • Total Net P&L:      ${m_all.get('net_pnl', 0):,.2f}")
    print(f"  • Win Rate:           {m_all.get('win_rate', 0):.1f}%")
    print(f"  • Profit Factor:      {m_all.get('pf', 0):.2f}")
    print(f"  • Max Drawdown:       ${m_all.get('max_dd', 0):,.2f}")
    print(f"  • Avg Risk / Trade:   ${m_all.get('avg_risk_usd', 0):,.2f} ({m_all.get('avg_contracts', 1):.1f} contracts avg)")
    print(f"\n🔬 ROBUSTNESS (OOS Holdout split at {split_date}):")
    print(f"  • In-Sample P&L:      ${m_is.get('net_pnl', 0):,.2f} | WR: {m_is.get('win_rate', 0):.1f}% | PF: {m_is.get('pf', 0):.2f}")
    print(f"  • Out-of-Sample P&L:  ${m_oos.get('net_pnl', 0):,.2f} | WR: {m_oos.get('win_rate', 0):.1f}% | PF: {m_oos.get('pf', 0):.2f}")
    
    # 2. Prop Firm Evaluation
    if sim:
        print(f"\n🎯 PROP EVALUATION ({prop_preset}):")
        print(f"  • Target: ${rules['profit_target']:,} | Trailing DD: ${rules['max_drawdown']:,} | DLL: ${rules['daily_loss_limit']:,} ({rules['dll_type']})")
        print(f"  • Trade Cadence:                  {sim['trades_per_month']:.2f} trades/month (1 trade every {sim['cadence_days_per_trade']:.1f} market days)")
        print(f"  • Realistic 60-Day Pass Rate:     {sim['realistic_challenge_pass_rate_pct']:.1f}%  (within 3-month challenge window)")
        print(f"  • Open-Ended (Indefinite) Pass:   {sim['unbounded_pass_rate_pct']:.1f}%")
        print(f"  • Median Trades to Pass:          {sim['median_days_to_pass']:.0f} trades")
        print(f"  • Est. Calendar Months to Pass:   {sim['est_calendar_months_to_pass']:.1f} months")
        print(f"  • Est. Subscription Cost:         ${sim['est_subscription_cost_usd']:,.2f}")
        print(f"  • Blown by Max Drawdown:          {sim['blown_max_dd_pct']:.1f}%")
        print(f"  • Daily Lockouts Hit:             {sim.get('soft_lockout_pct', 0.0):.1f}%")
        print(f"  • Funded Stage Payout Rate:       {sim['funded_stage_payout_rate_pct']:.1f}%")
        print(f"  • Avg Payout Collected:           ${sim['avg_payouts_usd']:,.2f}")
        print(f"  • Net EV per Challenge:           ${sim['net_ev_per_challenge']:+,.2f}")
        
        print("\n📈 SIZING SENSITIVITY MATRIX:")
        sens = calculate_risk_sensitivity_matrix(trades, rules, all_sessions=all_sess)
        print(sens.to_string(index=False))
        
    return trades, metrics, sim

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Institutional Prop Backtest")
    parser.add_argument("--prompt", type=str, default="", help="Natural language strategy prompt")
    parser.add_argument("--asset", type=str, default="MNQ", choices=["MNQ", "NQ", "BTC"])
    parser.add_argument("--timeframe", type=str, default="15m", choices=["15m", "5m"])
    parser.add_argument("--strategy", type=str, default="Value Area Reclaim")
    parser.add_argument("--start", type=str, default="10:00")
    parser.add_argument("--end", type=str, default="12:00")
    parser.add_argument("--target-mode", type=str, default="opposite_va")
    parser.add_argument("--ny-filter", type=str, default="none")
    parser.add_argument("--trend-filter", type=str, default="none")
    parser.add_argument("--risk", type=float, default=250.0)
    parser.add_argument("--prop", type=str, default="Topstep $50k")
    
    args = parser.parse_args()
    
    if args.prompt:
        compiled = compile_user_prompt_to_setup(args.prompt)
        print(f"🤖 AI Parsed Prompt: {compiled['explanation']}")
        run_backtest(
            asset=compiled.get("asset", args.asset),
            timeframe=compiled.get("timeframe", args.timeframe),
            strategy=compiled.get("strategy", args.strategy),
            params=compiled.get("params", {}),
            dollar_risk=args.risk,
            prop_preset=compiled.get("prop_preset", args.prop)
        )
    else:
        params = {
            "start_time": normalize_time_str(args.start),
            "end_time": normalize_time_str(args.end),
            "target_mode": args.target_mode,
            "ny_open_filter": args.ny_filter,
            "trend_filter": args.trend_filter
        }
        run_backtest(
            asset=args.asset,
            timeframe=args.timeframe,
            strategy=args.strategy,
            params=params,
            dollar_risk=args.risk,
            prop_preset=args.prop
        )
