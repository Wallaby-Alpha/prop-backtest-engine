import pandas as pd
import numpy as np
from engine.data_manager import normalize_time_str

class StrategyRegistry:
    _strategies = {}

    @classmethod
    def register(cls, name):
        def decorator(fn):
            cls._strategies[name] = fn
            return fn
        return decorator

    @classmethod
    def get(cls, name):
        if name not in cls._strategies:
            raise KeyError(f"Strategy '{name}' not found. Available: {list(cls._strategies.keys())}")
        return cls._strategies[name]

    @classmethod
    def list_available(cls):
        return list(cls._strategies.keys())


@StrategyRegistry.register("Value Area Reclaim")
def value_area_reclaim_strategy(df, daily_profiles, params):
    """
    Volume Profile Value Area Reclaim Strategy.
    Probes outside prior-day Value Area, fails to accept, and closes back inside.
    
    Params:
      - start_time: str, "HH:MM" in ET (e.g. "10:00")
      - end_time: str, "HH:MM" in ET (e.g. "12:00")
      - target_mode: str, 'opposite_va' (full rotation), 'poc' (conservative), or 'scale_50_50'
      - stop_loss_buffer_atr: float, buffer below/above probe extreme in ATR(14)
      - require_vol_surge: bool, require volume > 20-period volume SMA on reclaim bar
      - eod_exit_time: str, "HH:MM" in ET (e.g. "15:55")
    """
    start_time = normalize_time_str(params.get("start_time", "10:00"), default="10:00")
    end_time = normalize_time_str(params.get("end_time", "12:00"), default="12:00")
    target_mode = params.get("target_mode", "opposite_va")
    stop_buffer_atr = params.get("stop_loss_buffer_atr", 0.5)
    require_vol_surge = params.get("require_vol_surge", False)
    eod_time = normalize_time_str(params.get("eod_exit_time", "15:55"), default="15:55")
    
    # Advanced Filters
    ny_open_filter = params.get("ny_open_filter", "none") # 'none', 'outside_range', 'outside_va'
    trend_filter = params.get("trend_filter", "none")     # 'none', 'daily_ema50', 'daily_ema200'
    
    dates = sorted(list(daily_profiles.keys()))
    prior_profiles = {dates[i]: daily_profiles[dates[i - 1]] for i in range(1, len(dates))}
    
    n = len(df)
    closes = df['close'].values
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    volumes = df['volume'].values
    vol_smas = df['vol_sma20'].values
    atrs = df['atr14'].values
    times = df['time_str'].values
    dates_arr = df['session_date_et'].values if 'session_date_et' in df.columns else df['date_et'].values
    timestamps = df.index
    
    ema50_arr = df['daily_ema50'].values if 'daily_ema50' in df.columns else np.zeros(n)
    ema200_arr = df['daily_ema200'].values if 'daily_ema200' in df.columns else np.zeros(n)
    
    signals = [] # list of trade events
    
    in_pos = False
    pos_side = 0 # 1: Long, -1: Short
    entry_price = 0.0
    stop_price = 0.0
    tp1_price = 0.0
    tp2_price = 0.0
    entry_idx = 0
    pos_stage = 1
    
    was_below_val = False
    was_above_vah = False
    probe_low = 0.0
    probe_high = 0.0
    curr_day = None
    ny_open_price = None
    
    for i in range(1, n):
        d = dates_arr[i]
        if d not in prior_profiles:
            continue
            
        if d != curr_day:
            curr_day = d
            was_below_val = False
            was_above_vah = False
            probe_low = 1e9
            probe_high = -1e9
            ny_open_price = None
            
        p_prof = prior_profiles[d]
        val = p_prof['val']
        vah = p_prof['vah']
        poc = p_prof['poc']
        p_r_high = p_prof.get('range_high', vah)
        p_r_low = p_prof.get('range_low', val)
        
        t_str = times[i]
        o, h, l, c, v, vs, a = opens[i], highs[i], lows[i], closes[i], volumes[i], vol_smas[i], atrs[i]
        
        # Track 09:30 ET New York Open Price
        if "09:30" <= t_str <= "09:45" and ny_open_price is None:
            ny_open_price = o
        
        # Track outer probes
        if l < val:
            was_below_val = True
            if l < probe_low: probe_low = l
        if h > vah:
            was_above_vah = True
            if h > probe_high: probe_high = h
            
        # ---------------- Check Position Exits ----------------
        if in_pos:
            held = i - entry_idx
            # End of day flat rule
            if t_str >= eod_time or (i < n - 1 and dates_arr[i+1] != d):
                blended_exit = c if pos_stage == 1 else (tp1_price + c) / 2.0
                signals.append({
                    'entry_time': timestamps[entry_idx], 'exit_time': timestamps[i],
                    'entry_price': entry_price, 'exit_price': blended_exit,
                    'side': 'LONG' if pos_side == 1 else 'SHORT',
                    'bars': held, 'exit_reason': 'EOD_FLAT',
                    'date': d
                })
                in_pos = False
                continue
                
            if pos_side == 1: # LONG
                # Stop hit
                if l <= stop_price:
                    blended_exit = stop_price if pos_stage == 1 else (tp1_price + stop_price) / 2.0
                    reason = 'STOP_LOSS' if pos_stage == 1 else 'STOP_BE_AFTER_TP1'
                    signals.append({
                        'entry_time': timestamps[entry_idx], 'exit_time': timestamps[i],
                        'entry_price': entry_price, 'exit_price': blended_exit,
                        'side': 'LONG', 'bars': held, 'exit_reason': reason,
                        'date': d
                    })
                    in_pos = False
                # Target hit
                elif target_mode == 'poc' and h >= tp1_price:
                    signals.append({
                        'entry_time': timestamps[entry_idx], 'exit_time': timestamps[i],
                        'entry_price': entry_price, 'exit_price': tp1_price,
                        'side': 'LONG', 'bars': held, 'exit_reason': 'TARGET_POC',
                        'date': d
                    })
                    in_pos = False
                elif target_mode == 'opposite_va' and h >= tp2_price:
                    signals.append({
                        'entry_time': timestamps[entry_idx], 'exit_time': timestamps[i],
                        'entry_price': entry_price, 'exit_price': tp2_price,
                        'side': 'LONG', 'bars': held, 'exit_reason': 'TARGET_VAH',
                        'date': d
                    })
                    in_pos = False
                elif target_mode == 'scale_50_50':
                    if pos_stage == 1 and h >= tp1_price:
                        # Move stop to BE
                        stop_price = entry_price
                        pos_stage = 2
                    elif pos_stage == 2 and h >= tp2_price:
                        signals.append({
                            'entry_time': timestamps[entry_idx], 'exit_time': timestamps[i],
                            'entry_price': entry_price, 'exit_price': (tp1_price + tp2_price) / 2.0,
                            'side': 'LONG', 'bars': held, 'exit_reason': 'TARGET_SCALE_FULL',
                            'date': d
                        })
                        in_pos = False
                        
            elif pos_side == -1: # SHORT
                if h >= stop_price:
                    blended_exit = stop_price if pos_stage == 1 else (tp1_price + stop_price) / 2.0
                    reason = 'STOP_LOSS' if pos_stage == 1 else 'STOP_BE_AFTER_TP1'
                    signals.append({
                        'entry_time': timestamps[entry_idx], 'exit_time': timestamps[i],
                        'entry_price': entry_price, 'exit_price': blended_exit,
                        'side': 'SHORT', 'bars': held, 'exit_reason': reason,
                        'date': d
                    })
                    in_pos = False
                elif target_mode == 'poc' and l <= tp1_price:
                    signals.append({
                        'entry_time': timestamps[entry_idx], 'exit_time': timestamps[i],
                        'entry_price': entry_price, 'exit_price': tp1_price,
                        'side': 'SHORT', 'bars': held, 'exit_reason': 'TARGET_POC',
                        'date': d
                    })
                    in_pos = False
                elif target_mode == 'opposite_va' and l <= tp2_price:
                    signals.append({
                        'entry_time': timestamps[entry_idx], 'exit_time': timestamps[i],
                        'entry_price': entry_price, 'exit_price': tp2_price,
                        'side': 'SHORT', 'bars': held, 'exit_reason': 'TARGET_VAL',
                        'date': d
                    })
                    in_pos = False
                elif target_mode == 'scale_50_50':
                    if pos_stage == 1 and l <= tp1_price:
                        stop_price = entry_price
                        pos_stage = 2
                    elif pos_stage == 2 and l <= tp2_price:
                        signals.append({
                            'entry_time': timestamps[entry_idx], 'exit_time': timestamps[i],
                            'entry_price': entry_price, 'exit_price': (tp1_price + tp2_price) / 2.0,
                            'side': 'SHORT', 'bars': held, 'exit_reason': 'TARGET_SCALE_FULL',
                            'date': d
                        })
                        in_pos = False
                        
        # ---------------- Check Entries ----------------
        if not in_pos:
            # Check active trading window in Eastern Time
            if not (start_time <= t_str <= end_time):
                continue
            if require_vol_surge and (v < vs):
                continue
                
            # Check NY Open context filter
            if ny_open_filter == "outside_range":
                if ny_open_price is None or (p_r_low <= ny_open_price <= p_r_high):
                    continue
            elif ny_open_filter == "outside_va":
                if ny_open_price is None or (val <= ny_open_price <= vah):
                    continue
                
            # Long Setup: Probed below VAL, then candle closes back inside > VAL
            if was_below_val and (c > val) and (closes[i - 1] <= val):
                if trend_filter == "daily_ema50" and c <= ema50_arr[i]:
                    was_below_val = False
                    continue
                if trend_filter == "daily_ema200" and c <= ema200_arr[i]:
                    was_below_val = False
                    continue
                    
                in_pos = True
                pos_side = 1
                entry_price = c
                stop_price = probe_low - stop_buffer_atr * a
                tp1_price = poc
                tp2_price = vah
                entry_idx = i
                pos_stage = 1
                was_below_val = False # consumed
                
            # Short Setup: Probed above VAH, then candle closes back inside < VAH
            elif was_above_vah and (c < vah) and (closes[i - 1] >= vah):
                if trend_filter == "daily_ema50" and c >= ema50_arr[i]:
                    was_above_vah = False
                    continue
                if trend_filter == "daily_ema200" and c >= ema200_arr[i]:
                    was_above_vah = False
                    continue
                    
                in_pos = True
                pos_side = -1
                entry_price = c
                stop_price = probe_high + stop_buffer_atr * a
                tp1_price = poc
                tp2_price = val
                entry_idx = i
                pos_stage = 1
                was_above_vah = False # consumed

    return pd.DataFrame(signals)


@StrategyRegistry.register("Opening Range Breakout (ORB)")
def orb_strategy(df, daily_profiles, params):
    """
    Opening Range Breakout Strategy (e.g. 09:30 - 10:00 ET).
    """
    orb_start = normalize_time_str(params.get("orb_start", "09:30"), default="09:30")
    orb_end = normalize_time_str(params.get("orb_end", "10:00"), default="10:00")
    trade_end = normalize_time_str(params.get("trade_end", "12:00"), default="12:00")
    r_mult = params.get("r_target_mult", 2.0)
    eod_time = normalize_time_str(params.get("eod_exit_time", "15:55"), default="15:55")
    
    n = len(df)
    closes = df['close'].values
    highs = df['high'].values
    lows = df['low'].values
    times = df['time_str'].values
    dates_arr = df['session_date_et'].values if 'session_date_et' in df.columns else df['date_et'].values
    timestamps = df.index
    
    signals = []
    curr_day = None
    orb_h = -1e9
    orb_l = 1e9
    orb_done = False
    in_pos = False
    pos_side = 0
    entry_price = 0.0
    stop_price = 0.0
    target_price = 0.0
    entry_idx = 0
    
    for i in range(1, n):
        d = dates_arr[i]
        t_str = times[i]
        
        if d != curr_day:
            curr_day = d
            orb_h = -1e9
            orb_l = 1e9
            orb_done = False
            
        # Build ORB
        if orb_start <= t_str <= orb_end:
            if highs[i] > orb_h: orb_h = highs[i]
            if lows[i] < orb_l: orb_l = lows[i]
        elif t_str > orb_end and not orb_done and orb_h > orb_l:
            orb_done = True
            
        # Check Exit
        if in_pos:
            held = i - entry_idx
            if t_str >= eod_time or (i < n - 1 and dates_arr[i+1] != d):
                signals.append({
                    'entry_time': timestamps[entry_idx], 'exit_time': timestamps[i],
                    'entry_price': entry_price, 'exit_price': closes[i],
                    'side': 'LONG' if pos_side == 1 else 'SHORT',
                    'bars': held, 'exit_reason': 'EOD_FLAT', 'date': d
                })
                in_pos = False
                continue
                
            if pos_side == 1:
                if lows[i] <= stop_price:
                    signals.append({
                        'entry_time': timestamps[entry_idx], 'exit_time': timestamps[i],
                        'entry_price': entry_price, 'exit_price': stop_price,
                        'side': 'LONG', 'bars': held, 'exit_reason': 'STOP_LOSS', 'date': d
                    })
                    in_pos = False
                elif highs[i] >= target_price:
                    signals.append({
                        'entry_time': timestamps[entry_idx], 'exit_time': timestamps[i],
                        'entry_price': entry_price, 'exit_price': target_price,
                        'side': 'LONG', 'bars': held, 'exit_reason': 'TARGET_HIT', 'date': d
                    })
                    in_pos = False
            elif pos_side == -1:
                if highs[i] >= stop_price:
                    signals.append({
                        'entry_time': timestamps[entry_idx], 'exit_time': timestamps[i],
                        'entry_price': entry_price, 'exit_price': stop_price,
                        'side': 'SHORT', 'bars': held, 'exit_reason': 'STOP_LOSS', 'date': d
                    })
                    in_pos = False
                elif lows[i] <= target_price:
                    signals.append({
                        'entry_time': timestamps[entry_idx], 'exit_time': timestamps[i],
                        'entry_price': entry_price, 'exit_price': target_price,
                        'side': 'SHORT', 'bars': held, 'exit_reason': 'TARGET_HIT', 'date': d
                    })
                    in_pos = False

        # Check Entry
        if not in_pos and orb_done and (orb_end < t_str <= trade_end):
            if closes[i] > orb_h and closes[i-1] <= orb_h:
                in_pos = True
                pos_side = 1
                entry_price = closes[i]
                risk = orb_h - orb_l
                stop_price = orb_l
                target_price = entry_price + r_mult * risk
                entry_idx = i
            elif closes[i] < orb_l and closes[i-1] >= orb_l:
                in_pos = True
                pos_side = -1
                entry_price = closes[i]
                risk = orb_h - orb_l
                stop_price = orb_h
                target_price = entry_price - r_mult * risk
                entry_idx = i
                
    return pd.DataFrame(signals)
