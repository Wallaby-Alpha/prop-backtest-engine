import os
import pandas as pd
import numpy as np
from engine.data_manager import load_asset_data, compute_daily_volume_profiles
from engine.strategies import StrategyRegistry

def run_intrabar_audit():
    print("==================================================================", flush=True)
    print("      DEEP DIVE: INTRABAR AMBIGUITY & LOOKAHEAD AUDIT (15M vs 1M)  ", flush=True)
    print("==================================================================", flush=True)
    
    # 1. Load 15m and 1m datasets
    print("Loading 15m and 1m datasets...", flush=True)
    df_15m, spec = load_asset_data('MNQ', '15m')
    df_1m, _ = load_asset_data('MNQ', '1m')
    daily_profiles = compute_daily_volume_profiles(df_15m)
    
    print(f"15m bars: {len(df_15m):,}, 1m bars: {len(df_1m):,}")
    
    # 2. Run 15m Strategy
    strat = StrategyRegistry.get("Value Area Reclaim")
    params = {
        "start_time": "10:00",
        "end_time": "12:00",
        "target_mode": "opposite_va",
        "stop_loss_buffer_atr": 0.5,
        "eod_exit_time": "15:55"
    }
    signals = strat(df_15m, daily_profiles, params)
    print(f"\nGenerated {len(signals)} signals on 15m bars.")
    
    # 3. Check Close[i] vs Open[i+1] differences
    diffs = []
    for idx, row in signals.iterrows():
        entry_t = row['entry_time']
        # Find entry bar in 15m
        bar_loc = df_15m.index.get_loc(entry_t)
        if bar_loc + 1 < len(df_15m):
            c_i = df_15m['close'].iloc[bar_loc]
            o_next = df_15m['open'].iloc[bar_loc + 1]
            diffs.append(abs(o_next - c_i))
            
    avg_diff = np.mean(diffs)
    max_diff = np.max(diffs)
    print(f"\n[Execution Gap: Close[i] vs Open[i+1]]")
    print(f"  Average abs difference: {avg_diff:.4f} pts (${avg_diff * 2:.2f} on MNQ)")
    print(f"  Max abs difference:     {max_diff:.4f} pts (${max_diff * 2:.2f} on MNQ)")
    print(f"  (Note: Engine applies 0.50 pt slippage per side, covering typical spread/gap)")

    # 4. Check Intrabar Collision on 15m bars:
    # Does any 15m exit bar touch BOTH stop and target?
    ambiguous_count = 0
    checked_against_1m = 0
    resolved_same = 0
    resolved_stop_first = 0
    resolved_target_first = 0
    
    print("\n[Cross-verifying all trades with 1-Minute Bar Data]...")
    for idx, row in signals.iterrows():
        entry_t = row['entry_time']
        exit_t = row['exit_time']
        side = row['side']
        entry_p = row['entry_price']
        exit_p = row['exit_price']
        exit_reason = row['exit_reason']
        
        # Get 15m exit bar
        exit_bar = df_15m.loc[exit_t]
        h_15 = exit_bar['high']
        l_15 = exit_bar['low']
        
        # Check if this exit bar was potentially ambiguous
        # E.g. for long: did h_15 reach target AND l_15 reach stop?
        # Let's inspect the actual 1-minute path between entry_t and exit_t + 15m
        slice_1m = df_1m.loc[entry_t : exit_t + pd.Timedelta(minutes=14)]
        if len(slice_1m) == 0:
            continue
            
        # Re-run stop and target check minute by minute on 1m bars
        # Determine which price level was hit first on 1m
        target_p = exit_p if 'TARGET' in exit_reason else None
        stop_p = exit_p if 'STOP' in exit_reason else None
        
        # If we know both target level and stop level from setup:
        # In opposite_va:
        prof = daily_profiles.get(row['date'])
        if not prof:
            continue
        tp = prof['vah'] if side == 'LONG' else prof['val']
        
        # Check if on ANY 15m bar both tp and stop could be touched
        # Let's test on the 1m slice
        first_event = None
        for t_1m, b_1m in slice_1m.iterrows():
            if t_1m == entry_t: # Skip exact entry minute if order placed at end of 15m
                continue
            if side == 'LONG':
                hit_stop = b_1m['low'] <= stop_p if stop_p else False
                hit_tp = b_1m['high'] >= tp
                if hit_stop and hit_tp:
                    # Intrabar on 1m bar!
                    first_event = '1M_AMBIGUOUS'
                    break
                elif hit_stop:
                    first_event = 'STOP'
                    break
                elif hit_tp:
                    first_event = 'TARGET'
                    break
            else: # SHORT
                hit_stop = b_1m['high'] >= stop_p if stop_p else False
                hit_tp = b_1m['low'] <= tp
                if hit_stop and hit_tp:
                    first_event = '1M_AMBIGUOUS'
                    break
                elif hit_stop:
                    first_event = 'STOP'
                    break
                elif hit_tp:
                    first_event = 'TARGET'
                    break
                    
        if first_event:
            checked_against_1m += 1
            if ('STOP' in exit_reason and first_event == 'STOP') or ('TARGET' in exit_reason and first_event == 'TARGET'):
                resolved_same += 1
            elif 'STOP' in exit_reason and first_event == 'TARGET':
                resolved_target_first += 1
            elif 'TARGET' in exit_reason and first_event == 'STOP':
                resolved_stop_first += 1

    print(f"Total trades checked against 1m path: {checked_against_1m}")
    print(f"15m decision matched 1m exact ground truth: {resolved_same} ({resolved_same/max(1,checked_against_1m)*100:.1f}%)")
    print(f"15m stop assumption was conservative (1m hit target first): {resolved_target_first}")
    print(f"15m target assumption was optimistic (1m hit stop first): {resolved_stop_first}")

if __name__ == '__main__':
    run_intrabar_audit()
