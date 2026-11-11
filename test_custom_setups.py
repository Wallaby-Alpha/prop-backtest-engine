import pandas as pd
from engine.data_manager import load_asset_data, compute_daily_volume_profiles
from engine.strategies import StrategyRegistry
from engine.execution_engine import process_trades
from engine.prop_simulator import PROP_PRESETS, run_monte_carlo_prop_simulation

def main():
    print("==================================================================", flush=True)
    print("        TESTING SPECIFIC USER SETUPS: NY OPEN & TREND FILTERS     ", flush=True)
    print("==================================================================", flush=True)
    
    df, spec = load_asset_data("MNQ", "15m")
    daily_profiles = compute_daily_volume_profiles(df)
    strat_fn = StrategyRegistry.get("Value Area Reclaim")
    rules = PROP_PRESETS["Topstep $50k"]
    
    setups = [
        {
            "name": "1. Baseline (No Filters)",
            "params": {
                "start_time": "10:00", "end_time": "12:00",
                "target_mode": "opposite_va",
                "ny_open_filter": "none",
                "trend_filter": "none"
            }
        },
        {
            "name": "2. NY Open Outside Prior Range (Targeting Opposite VA)",
            "params": {
                "start_time": "10:00", "end_time": "12:00",
                "target_mode": "opposite_va",
                "ny_open_filter": "outside_range",
                "trend_filter": "none"
            }
        },
        {
            "name": "3. NY Open Outside Range + Longs Above 50d EMA / Shorts Below",
            "params": {
                "start_time": "10:00", "end_time": "12:00",
                "target_mode": "opposite_va",
                "ny_open_filter": "outside_range",
                "trend_filter": "daily_ema50"
            }
        },
        {
            "name": "4. NY Open Outside Range + Longs Above 200d EMA / Shorts Below",
            "params": {
                "start_time": "10:00", "end_time": "12:00",
                "target_mode": "opposite_va",
                "ny_open_filter": "outside_range",
                "trend_filter": "daily_ema200"
            }
        }
    ]
    
    results = []
    for s in setups:
        sig = strat_fn(df, daily_profiles, s["params"])
        if len(sig) == 0:
            results.append({"Setup": s["name"], "Trades": 0})
            continue
        trades_df, m = process_trades(sig, spec, contracts=2)
        m_all = m['all']
        
        sim = run_monte_carlo_prop_simulation(trades_df, rules, num_simulations=500) if len(trades_df) >= 10 else {}
        
        results.append({
            "Setup": s["name"],
            "Trades": m_all['trades'],
            "Win Rate": f"{m_all['win_rate']:.1f}%",
            "Profit Factor": f"{m_all['pf']:.2f}",
            "Net P&L": f"${m_all['net_pnl']:,.2f}",
            "Avg Win": f"${m_all['avg_win']:.2f}",
            "Avg Loss": f"${m_all['avg_loss']:.2f}",
            "Max DD": f"${m_all['max_dd']:,.2f}",
            "Topstep Pass Rate": f"{sim.get('pass_rate_pct', 0):.1f}%" if sim else "N/A"
        })
        
    res_df = pd.DataFrame(results)
    print("\n" + res_df.to_string(index=False))

if __name__ == '__main__':
    main()
