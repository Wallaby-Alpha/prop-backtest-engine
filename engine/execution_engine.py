import pandas as pd
import numpy as np

def process_trades(signals_df, spec, oos_split_date=None, contracts=1, slippage_pts=None, commission_per_side=None):
    """
    Applies asset specifications, contract position sizing, slippage, and commissions
    to raw trade signals to produce exact accounting and performance metrics.
    """
    if signals_df is None or len(signals_df) == 0:
        return pd.DataFrame(), {}
        
    df = signals_df.copy()
    mult = spec["multiplier"]
    slip = slippage_pts if slippage_pts is not None else spec["default_slippage_pts"]
    comm = commission_per_side if commission_per_side is not None else spec["default_commission"]
    
    # Calculate points and raw P&L
    is_long = df['side'] == 'LONG'
    
    # Points gained/lost per 1 contract (subtracting slippage on entry and exit)
    df['raw_pts'] = np.where(
        is_long,
        df['exit_price'] - df['entry_price'],
        df['entry_price'] - df['exit_price']
    )
    df['net_pts'] = df['raw_pts'] - (2 * slip)
    
    # Dollar P&L
    if spec["type"] == "futures":
        df['gross_pnl'] = df['raw_pts'] * mult * contracts
        df['total_commission'] = 2 * comm * contracts
        df['net_pnl'] = (df['net_pts'] * mult * contracts) - df['total_commission']
    else: # Crypto percentage-based
        df['ret_pct'] = df['net_pts'] / df['entry_price']
        df['net_pnl'] = df['net_pts'] * mult * contracts - (2 * comm * df['entry_price'] * contracts)

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
            'payoff': 0.0, 'sharpe': 0.0, 'sortino': 0.0
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
        'sortino': sortino
    }
