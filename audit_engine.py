import os
import pandas as pd
import numpy as np
from engine.data_manager import load_asset_data, compute_daily_volume_profiles
from engine.strategies import StrategyRegistry
from engine.execution_engine import process_trades
from engine.prop_simulator import PROP_PRESETS, simulate_prop_evaluation_single_path

def audit_sessions_and_profiles(df):
    print("--- 1. AUDITING CME FUTURES SESSIONS & VOLUME PROFILES ---")
    # Check Sunday vs Monday dates
    df_sun = df[df['time_et'].dt.day_name() == 'Sunday']
    print(f"Total Sunday 15m bars in dataset: {len(df_sun)}")
    if len(df_sun) > 0:
        sun_dates = df_sun['date_et'].unique()
        print(f"Sunday calendar dates count: {len(sun_dates)} (e.g. {sun_dates[:3]})")
        print("Note: In CME futures, Sunday 18:00 to Monday 17:00 ET is ONE single trading session ('Monday' trade date).")
        print("Calendar date grouping splits Sunday evening from Monday, creating an artificial Sunday profile!")

def audit_strategy_fills(df, dp):
    print("\n--- 2. AUDITING STRATEGY EXECUTION & FILL LOGIC ---")
    strat = StrategyRegistry.get("Value Area Reclaim")
    params = {"start_time": "10:00", "end_time": "12:00", "target_mode": "opposite_va", "eod_exit_time": "15:45"}
    signals = strat(df, dp, params)
    
    print(f"Total Signals: {len(signals)}")
    # Check if entry_time < exit_time
    time_order_valid = (signals['exit_time'] >= signals['entry_time']).all()
    print(f"Fill timestamp integrity (exit >= entry): {time_order_valid}")
    
    # Check same-bar exits vs multi-bar exits
    same_bar = (signals['bars'] == 0).sum()
    print(f"Same-bar entry & exit count: {same_bar} (should be 0 because exits are checked starting at bar i+1)")
    
    # Check stop vs target prices
    signals['diff'] = signals['exit_price'] - signals['entry_price']
    longs = signals[signals['side'] == 'LONG']
    shorts = signals[signals['side'] == 'SHORT']
    
    print(f"Longs count: {len(longs)}, Shorts count: {len(shorts)}")
    print(f"Longs avg win pts: {(longs[longs['diff'] > 0]['diff']).mean():.2f}, avg loss pts: {(longs[longs['diff'] <= 0]['diff']).mean():.2f}")
    print(f"Shorts avg win pts: {(-shorts[shorts['diff'] < 0]['diff']).mean():.2f}, avg loss pts: {(-shorts[shorts['diff'] >= 0]['diff']).mean():.2f}")

def audit_prop_rules():
    print("\n--- 3. AUDITING PROP FIRM RULES (TOPSTEP VS REALITY) ---")
    rules = PROP_PRESETS["Topstep $50k"]
    print(f"Configured rules: {rules}")
    print("AUDIT FINDING ON DAILY LOSS LIMIT (DLL):")
    print("In Topstep, the $1,000 Daily Loss Limit is a DAILY LOCKOUT (soft breach), NOT an account termination.")
    print("If an account loses $1,000 in a day, Topstep flattens the trader and locks the platform until 18:00 ET.")
    print("The account is ONLY blown if total balance dips below the $48,000 Max Drawdown threshold!")
    print("In our initial implementation, hitting DLL was marked as BLOWN (instant fail), which artificially crushed pass rates!")

if __name__ == '__main__':
    df, spec = load_asset_data('MNQ', '15m')
    dp = compute_daily_volume_profiles(df)
    audit_sessions_and_profiles(df)
    audit_strategy_fills(df, dp)
    audit_prop_rules()
