import pandas as pd
import numpy as np

def process_trades(
    signals_df,
    spec,
    oos_split_date=None,
    sizing_mode="dollar_risk", # 'dollar_risk' or 'fixed_contracts'
    contracts=2,
    dollar_risk_per_trade=250.0,
    slippage_pts=None,
    commission_per_side=None
):
    """
    Applies asset specifications, concrete position sizing (fixed contracts or fixed dollar risk per trade),
    slippage, and commissions to raw trade signals to produce exact accounting and performance metrics.
    """
    if signals_df is None or len(signals_df) == 0:
        return pd.DataFrame(), {}
        
    if isinstance(signals_df, list):
        df = pd.DataFrame(signals_df)
    else:
        df = signals_df.copy()
    mult = spec["multiplier"]
    slip = slippage_pts if slippage_pts is not None else spec["default_slippage_pts"]
    comm = commission_per_side if commission_per_side is not None else spec["default_commission"]
    
    # Calculate points per 1 contract (subtracting slippage on entry and exit)
    is_long = df['side'] == 'LONG'
    df['raw_pts'] = np.where(
        is_long,
        df['exit_price'] - df['entry_price'],
        df['entry_price'] - df['exit_price']
    )
    df['net_pts'] = df['raw_pts'] - (2 * slip)
    
    # Stop distance in points (if available)
    if 'stop_price' in df.columns:
        stop_dist = (pd.to_numeric(df['entry_price'], errors='coerce') - pd.to_numeric(df['stop_price'], errors='coerce')).abs()
    else:
        stop_dist = pd.Series(25.0, index=df.index)
        
    # Safeguard against any missing, NaN, or non-positive stop distances
    stop_dist = pd.to_numeric(stop_dist, errors='coerce').fillna(25.0)
    stop_dist = np.where(stop_dist <= 0, 25.0, stop_dist)
        
    risk_pts = stop_dist + (2 * slip)
    risk_per_1c = (risk_pts * mult) + (2 * comm)
    risk_per_1c = np.maximum(1.0, np.nan_to_num(risk_per_1c, nan=50.0))
    
    # Position Sizing: Concrete Dollar Risk vs Fixed Contracts
    if sizing_mode == "dollar_risk":
        target_risk = float(dollar_risk_per_trade) if dollar_risk_per_trade else 250.0
        c_calc = np.floor(target_risk / risk_per_1c)
        c_calc = np.nan_to_num(c_calc, nan=1.0)
        df['contracts'] = np.maximum(1, c_calc.astype(int))
        df['intended_risk_usd'] = target_risk
        df['actual_risk_usd'] = df['contracts'] * risk_per_1c
    else:
        c_fixed = int(contracts) if contracts else 1
        df['contracts'] = max(1, c_fixed)
        df['intended_risk_usd'] = df['contracts'] * risk_per_1c
        df['actual_risk_usd'] = df['contracts'] * risk_per_1c
        
    # Dollar P&L
    if spec["type"] == "futures":
        df['gross_pnl'] = df['raw_pts'] * mult * df['contracts']
        df['total_commission'] = 2 * comm * df['contracts']
        df['net_pnl'] = (df['net_pts'] * mult * df['contracts']) - df['total_commission']
    else: # Crypto percentage-based
        df['ret_pct'] = df['net_pts'] / df['entry_price']
        df['net_pnl'] = (df['net_pts'] * mult * df['contracts']) - (2 * comm * df['entry_price'] * df['contracts'])

    # In-Sample vs Out-of-Sample tagging
    if oos_split_date is not None:
        df['sample_type'] = np.where(df['date'] < oos_split_date, 'In-Sample', 'Out-of-Sample')
    else:
        df['sample_type'] = 'All'
        
    df['cum_pnl'] = df['net_pnl'].cumsum()
    df['peak_pnl'] = df['cum_pnl'].cummax()
    df['drawdown'] = df['cum_pnl'] - df['peak_pnl']
    
    stats_all = compute_metrics(df)
    stats_is = compute_metrics(df[df['sample_type'] == 'In-Sample']) if oos_split_date else {}
    stats_oos = compute_metrics(df[df['sample_type'] == 'Out-of-Sample']) if oos_split_date else {}
    
    return df, {
        'all': stats_all,
        'in_sample': stats_is,
        'out_sample': stats_oos
    }

def compute_metrics(trades_df):
    if len(trades_df) == 0:
        return {
            'trades': 0, 'win_rate': 0.0, 'net_pnl': 0.0, 'pf': 0.0,
            'max_dd': 0.0, 'ev': 0.0, 'avg_win': 0.0, 'avg_loss': 0.0,
            'payoff': 0.0, 'sharpe': 0.0, 'sortino': 0.0,
            'avg_contracts': 0.0, 'avg_risk_usd': 0.0, 'max_risk_usd': 0.0
        }
        
    n = len(trades_df)
    wins = trades_df[trades_df['net_pnl'] > 0]
    losses = trades_df[trades_df['net_pnl'] <= 0]
    
    win_rate = len(wins) / n * 100.0
    net_pnl = trades_df['net_pnl'].sum()
    sum_w = wins['net_pnl'].sum()
    sum_l = abs(losses['net_pnl'].sum()) if len(losses) > 0 else 1e-10
    pf = sum_w / sum_l if sum_l > 0 else 0.0
    
    avg_w = wins['net_pnl'].mean() if len(wins) > 0 else 0.0
    avg_l = abs(losses['net_pnl'].mean()) if len(losses) > 0 else 0.0
    payoff = avg_w / avg_l if avg_l > 0 else 0.0
    ev = trades_df['net_pnl'].mean()
    
    cum = trades_df['net_pnl'].cumsum()
    max_dd = (cum - cum.cummax()).min()
    
    std_pnl = trades_df['net_pnl'].std() if n > 1 else 1.0
    sharpe = (ev / (std_pnl + 1e-8)) * np.sqrt(252) # annualized estimate
    
    downside_std = losses['net_pnl'].std() if len(losses) > 1 else 1.0
    sortino = (ev / (downside_std + 1e-8)) * np.sqrt(252)
    
    avg_contracts = trades_df['contracts'].mean() if 'contracts' in trades_df.columns else 1.0
    avg_risk = trades_df['actual_risk_usd'].mean() if 'actual_risk_usd' in trades_df.columns else 0.0
    max_risk = trades_df['actual_risk_usd'].max() if 'actual_risk_usd' in trades_df.columns else 0.0
    
    return {
        'trades': n,
        'win_rate': win_rate,
        'net_pnl': net_pnl,
        'pf': pf,
        'max_dd': max_dd,
        'ev': ev,
        'avg_win': avg_w,
        'avg_loss': avg_l,
        'payoff': payoff,
        'sharpe': sharpe,
        'sortino': sortino,
        'avg_contracts': avg_contracts,
        'avg_risk_usd': avg_risk,
        'max_risk_usd': max_risk
    }
