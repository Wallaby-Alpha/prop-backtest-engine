import os
import glob
import pandas as pd
import numpy as np

DOWNLOADS_DIR = r"C:\Users\phkim\Downloads"
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

OUT_1M = os.path.join(CACHE_DIR, "mnq_1m.parquet")
OUT_5M = os.path.join(CACHE_DIR, "mnq_5m.parquet")
OUT_15M = os.path.join(CACHE_DIR, "mnq_15m.parquet")

def main():
    csv_files = sorted(glob.glob(os.path.join(DOWNLOADS_DIR, "mnq-*.csv")))
    print(f"Found {len(csv_files)} quarterly MNQ CSV files in {DOWNLOADS_DIR}...")
    
    dfs = []
    for f in csv_files:
        print(f"  Reading and filtering {os.path.basename(f)}...", flush=True)
        # Read with symbol column to filter out calendar spreads
        df = pd.read_csv(f, usecols=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'symbol'])
        
        # 1. Exclude calendar spreads (contain hyphen, e.g. MNQZ2-MNQH3)
        df = df[~df['symbol'].str.contains('-', na=False)]
        
        # 2. Convert timestamp
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        
        # 3. For any duplicate minute timestamps across contracts, keep the front-month (highest volume)
        df = df.sort_values('volume', ascending=False).drop_duplicates(subset=['timestamp'])
        
        dfs.append(df[['timestamp', 'open', 'high', 'low', 'close', 'volume']])
        
    print("Concatenating and sorting continuous outrights...", flush=True)
    full_df = pd.concat(dfs, ignore_index=True)
    full_df = full_df.sort_values('volume', ascending=False).drop_duplicates(subset=['timestamp']).sort_values('timestamp')
    full_df = full_df.set_index('timestamp')
    
    print(f"Merged Outright 1-minute dataset: {len(full_df):,} bars from {full_df.index.min()} to {full_df.index.max()}")
    print(f"Price Range: min={full_df['low'].min():.2f}, max={full_df['high'].max():.2f}")
    
    # Save 1m parquet
    full_df.to_parquet(OUT_1M)
    print(f"Saved 1M parquet: {OUT_1M} ({os.path.getsize(OUT_1M)/(1024*1024):.2f} MB)")
    
    # Resample to 5m
    print("Resampling to 5-minute bars...", flush=True)
    df_5m = full_df.resample('5min', closed='left', label='left').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    df_5m.to_parquet(OUT_5M)
    print(f"Saved 5M parquet: {OUT_5M} ({len(df_5m):,} bars, {os.path.getsize(OUT_5M)/(1024*1024):.2f} MB)")

    # Resample to 15m
    print("Resampling to 15-minute bars...", flush=True)
    df_15m = full_df.resample('15min', closed='left', label='left').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    df_15m.to_parquet(OUT_15M)
    print(f"Saved 15M parquet: {OUT_15M} ({len(df_15m):,} bars, {os.path.getsize(OUT_15M)/(1024*1024):.2f} MB)")

    print("\nClean Outright MNQ Data Preparation Complete!")

if __name__ == '__main__':
    main()
