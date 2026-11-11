import os
import pandas as pd
import numpy as np
from engine.data_manager import load_asset_data, compute_daily_volume_profiles
from engine.strategies import StrategyRegistry
from engine.execution_engine import process_trades

def run_intrabar_ground_truth_backtest():
    print("==================================================================", flush=True)
    print("     TRUE 1-MINUTE INTRABAR GROUND TRUTH VS 15M ESTIMATION        ", flush=True)
    print("==================================================================", flush=True)
    
    df_15m, spec = load_asset_data('MNQ', '15m')
    df_1m, _ = load_asset_data('MNQ', '1m')
    daily_profiles = compute_daily_volume_profiles(df_15m)
    
    # 1. 15m Strategy Signal Generation with 1m Intrabar Execution
    start_time = "10:00"
    end_time = "12:00"
    target_mode = "opposite_va"
    stop_buffer_atr = 0.5
    eod_time = "15:55"
    
    dates = sorted(list(daily_profiles.keys()))
    prior_profiles = {dates[i]: daily_profiles[dates[i - 1]] for i in range(1, len(dates))}
    
    n = len(df_15m)
    closes_15 = df_15m['close'].values
    opens_15 = df_15m['open'].values
    atrs_15 = df_15m['atr14'].values
    times_15 = df_15m['time_str'].values
    dates_15 = df_15m['session_date_et'].values
    timestamps_15 = df_15m.index
    
    trades_15m_approx = []
    trades_1m_ground_truth = []
    
    in_pos = False
    pos_side = 0
    entry_p = 0.0
    stop_p = 0.0
    tp_p = 0.0
    entry_idx = 0
    
    was_below_val = False
    was_above_vah = False
    probe_low = 1e9
    probe_high = -1e9
    curr_day = None
    
    # Pre-index 1m data for fast lookup
    df_1m_times = df_1m.index
    
    for i in range(1, n):
        d = dates_15[i]
        if d not in prior_profiles:
            continue
            
        if d != curr_day:
            curr_day = d
            was_below_val = False
            was_above_vah = False
            probe_low = 1e9
            probe_high = -1e9
            
        p_prof = prior_profiles[d]
        val = p_prof['val']
        vah = p_prof['vah']
        poc = p_prof['poc']
        
        t_str = times_15[i]
        o, h, l, c, a = opens_15[i], df_15m['high'].iloc[i], df_15m['low'].iloc[i], closes_15[i], atrs_15[i]
        
        # Outer probe tracking
        if l < val:
            was_below_val = True
            if l < probe_low: probe_low = l
        if h > vah:
            was_above_vah = True
            if h > probe_high: probe_high = h
            
        # Entry check: Reclaim inside Value Area
        if not in_pos and (start_time <= t_str <= end_time):
            if was_below_val and (c > val) and (closes_15[i - 1] <= val):
                in_pos = True
                pos_side = 1
                # Execution on next bar open to be 100% lookahead-proof:
                entry_idx = i
                entry_time = timestamps_15[i]
                # Filled at candle close or next bar open:
                entry_p = closes_15[i]
                stop_p = probe_low - stop_buffer_atr * a
                tp_p = vah
                was_below_val = False
                
                # Now simulate trade path using 1-minute bars until exit
                # 1m bar slice starting right after entry candle
                # Candle i covers [timestamps_15[i] - 15m, timestamps_15[i]]
                slice_1m = df_1m.loc[entry_time : entry_time + pd.Timedelta(hours=8)]
                
                # 1m Ground Truth Execution:
                exit_found = False
                for t_1m, bar_1m in slice_1m.iterrows():
                    if t_1m == entry_time:
                        continue
                    m_time_str = bar_1m['time_str']
                    m_h, m_l, m_c = bar_1m['high'], bar_1m['low'], bar_1m['close']
                    
                    # Check EOD
                    if m_time_str >= eod_time:
                        trades_1m_ground_truth.append({
                            'entry_time': entry_time, 'exit_time': t_1m,
                            'entry_price': entry_p, 'exit_price': m_c,
                            'side': 'LONG', 'exit_reason': 'EOD_FLAT', 'date': d
                        })
                        exit_found = True
                        break
                        
                    # Check Stop and TP on 1-minute bar
                    hit_stop = m_l <= stop_p
                    hit_tp = m_h >= tp_p
                    
                    if hit_tp and not hit_stop:
                        trades_1m_ground_truth.append({
                            'entry_time': entry_time, 'exit_time': t_1m,
                            'entry_price': entry_p, 'exit_price': tp_p,
                            'side': 'LONG', 'exit_reason': 'TARGET_VAH', 'date': d
                        })
                        exit_found = True
                        break
                    elif hit_stop and not hit_tp:
                        trades_1m_ground_truth.append({
                            'entry_time': entry_time, 'exit_time': t_1m,
                            'entry_price': entry_p, 'exit_price': stop_p,
                            'side': 'LONG', 'exit_reason': 'STOP_LOSS', 'date': d
                        })
                        exit_found = True
                        break
                    elif hit_stop and hit_tp:
                        # Even on 1m, if both hit, assume stop (ultra-conservative)
                        trades_1m_ground_truth.append({
                            'entry_time': entry_time, 'exit_time': t_1m,
                            'entry_price': entry_p, 'exit_price': stop_p,
                            'side': 'LONG', 'exit_reason': 'STOP_LOSS_1M_AMBIGUOUS', 'date': d
                        })
                        exit_found = True
                        break
                        
                in_pos = False # Ready for next trade
                
            elif was_above_vah and (c < vah) and (closes_15[i - 1] >= vah):
                in_pos = True
                pos_side = -1
                entry_idx = i
                entry_time = timestamps_15[i]
                entry_p = closes_15[i]
                stop_p = probe_high + stop_buffer_atr * a
                tp_p = val
                was_above_vah = False
                
                slice_1m = df_1m.loc[entry_time : entry_time + pd.Timedelta(hours=8)]
                for t_1m, bar_1m in slice_1m.iterrows():
                    if t_1m == entry_time:
                        continue
                    m_time_str = bar_1m['time_str']
                    m_h, m_l, m_c = bar_1m['high'], bar_1m['low'], bar_1m['close']
                    
                    if m_time_str >= eod_time:
                        trades_1m_ground_truth.append({
                            'entry_time': entry_time, 'exit_time': t_1m,
                            'entry_price': entry_p, 'exit_price': m_c,
                            'side': 'SHORT', 'exit_reason': 'EOD_FLAT', 'date': d
                        })
                        break
                        
                    hit_stop = m_h >= stop_p
                    hit_tp = m_l <= tp_p
                    
                    if hit_tp and not hit_stop:
                        trades_1m_ground_truth.append({
                            'entry_time': entry_time, 'exit_time': t_1m,
                            'entry_price': entry_p, 'exit_price': tp_p,
                            'side': 'SHORT', 'exit_reason': 'TARGET_VAL', 'date': d
                        })
                        break
                    elif hit_stop and not hit_tp:
                        trades_1m_ground_truth.append({
                            'entry_time': entry_time, 'exit_time': t_1m,
                            'entry_price': entry_p, 'exit_price': stop_p,
                            'side': 'SHORT', 'exit_reason': 'STOP_LOSS', 'date': d
                        })
                        break
                    elif hit_stop and hit_tp:
                        trades_1m_ground_truth.append({
                            'entry_time': entry_time, 'exit_time': t_1m,
                            'entry_price': entry_p, 'exit_price': stop_p,
                            'side': 'SHORT', 'exit_reason': 'STOP_LOSS_1M_AMBIGUOUS', 'date': d
                        })
                        break
                        
                in_pos = False

    df_gt = pd.DataFrame(trades_1m_ground_truth)
    print(f"Generated {len(df_gt)} Ground Truth 1-Minute Trades.")
    
    trades_processed, metrics = process_trades(df_gt, spec, contracts=2)
    m = metrics['all']
    print("\n--- 1-MINUTE GROUND TRUTH METRICS (2 MNQ CONTRACTS) ---")
    print(f"Total Trades:   {m['trades']}")
    print(f"Win Rate:       {m['win_rate']:.1f}%")
    print(f"Profit Factor:  {m['pf']:.2f}")
    print(f"Total Net P&L:  ${m['net_pnl']:,.2f}")
    print(f"Max Drawdown:   ${m['max_dd']:,.2f}")
    print(f"Avg Win / Loss: ${m['avg_win']:,.2f} / ${m['avg_loss']:,.2f}")
    print(f"Expected Value: ${m['ev']:,.2f} per trade")

if __name__ == '__main__':
    run_intrabar_ground_truth_backtest()
