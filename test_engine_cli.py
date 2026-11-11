import os
import pandas as pd
from engine.data_manager import load_asset_data, compute_daily_volume_profiles, split_in_sample_out_sample
from engine.strategies import StrategyRegistry
from engine.execution_engine import process_trades
from engine.prop_simulator import PROP_PRESETS, run_monte_carlo_prop_simulation, calculate_risk_sensitivity_matrix

def main():
    print("==================================================================", flush=True)
    print("         END-TO-END ENGINE TEST: NQ / MNQ VALUE AREA RECLAIM       ", flush=True)
    print("==================================================================", flush=True)
    
    # 1. Load Data
    symbol = "MNQ"
    print(f"Loading {symbol} 15m dataset...", flush=True)
    df, spec = load_asset_data(symbol=symbol, timeframe="15m")
    print(f"Loaded {len(df):,} bars from {df['date_et'].min()} to {df['date_et'].max()}")
    
    # 2. Daily Volume Profiles
    print("Computing Daily Volume Profiles...", flush=True)
    daily_profiles = compute_daily_volume_profiles(df)
    print(f"Computed {len(daily_profiles):,} prior-day profiles.")
    
    # 3. In-Sample / Out-of-Sample Split (70% IS / 30% OOS)
    is_mask, oos_mask, split_date = split_in_sample_out_sample(df, oos_ratio=0.30)
    print(f"In-Sample / Out-of-Sample split date: {split_date}")
    
    # 4. Strategy Execution: Value Area Reclaim between 10:00 and 12:00 ET
    strategy_fn = StrategyRegistry.get("Value Area Reclaim")
    params = {
        "start_time": "10:00",
        "end_time": "12:00",
        "target_mode": "opposite_va", # Full rotation to opposite boundary
        "stop_loss_buffer_atr": 0.5,
        "require_vol_surge": False,
        "eod_exit_time": "15:55"
    }
    
    print(f"\nRunning Strategy: Value Area Reclaim ({params['start_time']} - {params['end_time']} ET)...", flush=True)
    raw_signals = strategy_fn(df, daily_profiles, params)
    print(f"Raw Signals Generated: {len(raw_signals):,} trades.")
    
    # 5. Process Trades with Slippage and Commissions (trading 2 MNQ contracts)
    contracts = 2
    trades_df, metrics = process_trades(
        raw_signals,
        spec,
        oos_split_date=split_date,
        contracts=contracts
    )
    
    print("\n--- LIVE ACCOUNT METRICS ---")
    m_all = metrics['all']
    m_is = metrics['in_sample']
    m_oos = metrics['out_sample']
    
    print(f"Total Trades:      {m_all['trades']} (IS: {m_is.get('trades',0)}, OOS: {m_oos.get('trades',0)})")
    print(f"Win Rate:          {m_all['win_rate']:.1f}% (IS: {m_is.get('win_rate',0):.1f}%, OOS: {m_oos.get('win_rate',0):.1f}%)")
    print(f"Profit Factor:     {m_all['pf']:.2f} (IS: {m_is.get('pf',0):.2f}, OOS: {m_oos.get('pf',0):.2f})")
    print(f"Total Net P&L:     ${m_all['net_pnl']:,.2f} (IS: ${m_is.get('net_pnl',0):,.2f}, OOS: ${m_oos.get('net_pnl',0):,.2f})")
    print(f"Max Drawdown:      ${m_all['max_dd']:,.2f}")
    print(f"Avg Win / Loss:    ${m_all['avg_win']:,.2f} / ${m_all['avg_loss']:,.2f} (Payoff: {m_all['payoff']:.2f}:1)")
    print(f"Expected Value:    ${m_all['ev']:,.2f} per trade")
    
    # 6. Prop Firm Simulation: Topstep $50k Rule
    prop_preset_name = "Topstep $50k"
    rules = PROP_PRESETS[prop_preset_name]
    print(f"\n==================================================================", flush=True)
    print(f"         PROP FIRM SIMULATION: {prop_preset_name.upper()}         ", flush=True)
    print(f"==================================================================", flush=True)
    print(f"Target: ${rules['profit_target']:,} | Max DD: ${rules['max_drawdown']:,} ({rules['dd_type']}) | Daily Loss Limit: ${rules['daily_loss_limit']:,}")
    
    sim_res = run_monte_carlo_prop_simulation(
        trades_df,
        rules,
        eval_risk_scaler=1.0,     # Standard 2 MNQ contracts
        funded_risk_scaler=0.5,   # Dropping to 1 MNQ in funded stage to protect buffer!
        num_simulations=1000
    )
    
    print(f"\n[Monte Carlo Evaluation Results (1,000 Trials)]")
    print(f"  Evaluation Pass Rate:            {sim_res['pass_rate_pct']:.1f}%")
    print(f"  Blown by Max Trailing DD:        {sim_res['blown_max_dd_pct']:.1f}%")
    print(f"  Blown by Daily Loss Limit:       {sim_res['blown_dll_pct']:.1f}%")
    print(f"  Median Days to Pass:             {sim_res['median_days_to_pass']:.0f} trading days")
    print(f"  90th Percentile Days to Pass:    {sim_res['p90_days_to_pass']:.0f} trading days")
    
    print(f"\n[Funded Stage & Payout Results]")
    print(f"  Overall Payout Rate:             {sim_res['payout_rate_overall_pct']:.1f}% (of all attempts)")
    print(f"  Funded Account Payout Rate:      {sim_res['funded_stage_payout_rate_pct']:.1f}% (of passed accounts)")
    print(f"  Average Total Payout Collected:  ${sim_res['avg_payouts_usd']:,.2f}")
    print(f"  Challenge Fee:                   ${sim_res['challenge_fee']:.2f}")
    print(f"  Net Expected Value (EV):         ${sim_res['net_ev_per_challenge']:+,.2f} per evaluation purchased")
    
    # 7. Risk Sensitivity Matrix
    print(f"\n[Risk Sensitivity Matrix (Evaluation vs Funded Scaling)]")
    sens_df = calculate_risk_sensitivity_matrix(trades_df, rules)
    print(sens_df.to_string(index=False))

if __name__ == '__main__':
    main()
