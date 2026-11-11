import os
import pandas as pd
import numpy as np
from datetime import datetime, time
import pytz

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data_cache")

ASSET_SPECS = {
    "MNQ": {
        "name": "Micro E-mini Nasdaq-100 Futures",
        "multiplier": 2.0,      # $2.00 per full point
        "tick_size": 0.25,      # 0.25 pts = $0.50 per tick
        "default_commission": 0.62, # $0.62 per side ($1.24 round trip)
        "default_slippage_pts": 0.25, # 1 tick slippage
        "type": "futures",
        "file_prefix": "mnq"
    },
    "NQ": {
        "name": "E-mini Nasdaq-100 Futures",
        "multiplier": 20.0,     # $20.00 per full point
        "tick_size": 0.25,      # 0.25 pts = $5.00 per tick
        "default_commission": 2.50, # $2.50 per side ($5.00 round trip)
        "default_slippage_pts": 0.25, # 1 tick slippage
        "type": "futures",
        "file_prefix": "mnq"    # derived from MNQ data
    },
    "BTC": {
        "name": "Bitcoin USD (Crypto Continuous)",
        "multiplier": 1.0,      # $1.00 per dollar move
        "tick_size": 0.10,
        "default_commission": 0.0005, # 5 bps maker/taker
        "default_slippage_pts": 5.0,  # $5.00 slippage
        "type": "crypto",
        "file_prefix": "btc"
    }
}

def load_asset_data(symbol="MNQ", timeframe="15m"):
    """
    Load standardized OHLCV dataframe with Eastern Time timezone index and helper columns.
    """
    spec = ASSET_SPECS.get(symbol, ASSET_SPECS["MNQ"])
    prefix = spec["file_prefix"]
    fpath = os.path.join(CACHE_DIR, f"{prefix}_{timeframe}.parquet")
    
    if not os.path.exists(fpath):
        raise FileNotFoundError(f"Data file not found: {fpath}. Please run data prep first.")
        
    df = pd.read_parquet(fpath)
    df = df.sort_index()
    
    # Project timestamps to Eastern Time (ET) for session logic
    ny_tz = pytz.timezone('America/New_York')
    df['time_et'] = df.index.tz_convert(ny_tz)
    df['date_et'] = df['time_et'].dt.date
    df['time_str'] = df['time_et'].dt.strftime('%H:%M')
    df['hour_et'] = df['time_et'].dt.hour
    df['minute_et'] = df['time_et'].dt.minute
    
    # CME Futures Session Trade Date:
    # Sunday 18:00 ET belongs to Monday trade date. Monday 18:00 belongs to Tuesday, etc.
    if spec["type"] == "futures":
        shifted = df['time_et'] + pd.to_timedelta(np.where(df['hour_et'] >= 18, 6, 0), unit='h')
        df['session_date_et'] = shifted.dt.date
    else:
        df['session_date_et'] = df['date_et']
    
    # Compute ATR(14)
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift(1)).abs(),
        (df['low'] - df['close'].shift(1)).abs()
    ], axis=1).max(axis=1)
    df['atr14'] = tr.rolling(14).mean().bfill()
    df['vol_sma20'] = df['volume'].rolling(20).mean().bfill()
    
    return df, spec

def compute_daily_volume_profiles(df, value_area_pct=0.70, num_bins=60):
    """
    Computes prior-day Volume Profile (VAH, VAL, POC) for each trading day/session.
    """
    daily_profiles = {}
    grouped = df.groupby('session_date_et')
    
    for d, group in grouped:
        if len(group) < 5 or group['volume'].sum() == 0:
            continue
            
        p_low = group['low'].min()
        p_high = group['high'].max()
        if p_high <= p_low:
            continue
            
        bin_size = (p_high - p_low) / num_bins
        bins = np.linspace(p_low, p_high, num_bins + 1)
        bin_centers = (bins[:-1] + bins[1:]) / 2.0
        
        # Approximate volume profile by assigning bar volume across high-low span
        bin_volumes = np.zeros(num_bins)
        for _, row in group.iterrows():
            b_low, b_high, vol = row['low'], row['high'], row['volume']
            if vol == 0 or b_high <= b_low:
                continue
            idx_start = max(0, min(num_bins - 1, int((b_low - p_low) / bin_size)))
            idx_end = max(0, min(num_bins - 1, int((b_high - p_low) / bin_size)))
            span_bins = idx_end - idx_start + 1
            vol_per_bin = vol / span_bins
            bin_volumes[idx_start:idx_end + 1] += vol_per_bin
            
        total_vol = bin_volumes.sum()
        if total_vol == 0:
            continue
            
        poc_idx = np.argmax(bin_volumes)
        poc = bin_centers[poc_idx]
        
        target_va_vol = total_vol * value_area_pct
        va_vol = bin_volumes[poc_idx]
        low_idx = poc_idx
        high_idx = poc_idx
        
        while va_vol < target_va_vol:
            next_up = bin_volumes[high_idx + 1] if high_idx + 1 < num_bins else -1
            next_dn = bin_volumes[low_idx - 1] if low_idx - 1 >= 0 else -1
            if next_up == -1 and next_dn == -1:
                break
            if next_up >= next_dn:
                high_idx += 1
                va_vol += next_up
            else:
                low_idx -= 1
                va_vol += next_dn
                
        vah = bin_centers[high_idx]
        val = bin_centers[low_idx]
        
        daily_profiles[d] = {
            'vah': vah,
            'val': val,
            'poc': poc,
            'range_high': p_high,
            'range_low': p_low
        }
        
    return daily_profiles

def split_in_sample_out_sample(df, oos_ratio=0.30):
    """
    Splits dataset into In-Sample (training/discovery) and Out-of-Sample (validation)
    based on chronological session trade date without lookahead.
    """
    date_col = 'session_date_et' if 'session_date_et' in df.columns else 'date_et'
    unique_dates = sorted(list(set(df[date_col])))
    split_idx = int(len(unique_dates) * (1.0 - oos_ratio))
    split_date = unique_dates[split_idx]
    
    is_mask = df[date_col] < split_date
    oos_mask = df[date_col] >= split_date
    
    return is_mask, oos_mask, split_date
