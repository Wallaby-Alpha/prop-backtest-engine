import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from engine.data_manager import load_asset_data, compute_daily_volume_profiles, split_in_sample_out_sample, ASSET_SPECS
from engine.strategies import StrategyRegistry
from engine.execution_engine import process_trades
from engine.prop_simulator import PROP_PRESETS, run_monte_carlo_prop_simulation, calculate_risk_sensitivity_matrix

# Page Configuration
st.set_page_config(
    page_title="Antigravity Prop & Backtesting Engine",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #1e222d;
        border-radius: 8px;
        padding: 15px;
        border: 1px solid #2a2e39;
        margin-bottom: 10px;
    }
    .metric-val {
        font-size: 24px;
        font-weight: 700;
        color: #f0b90b;
    }
    .metric-lbl {
        font-size: 13px;
        color: #9aa0a6;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- SIDEBAR CONTROLS -----------------
st.sidebar.title("🛠️ Strategy & Prop Config")

# 1. Asset & Timeframe
st.sidebar.subheader("1. Asset & Data")
selected_asset = st.sidebar.selectbox("Asset Symbol", options=["MNQ", "NQ", "BTC"], index=0)
selected_tf = st.sidebar.selectbox("Timeframe", options=["15m", "5m"], index=0)
contracts = st.sidebar.number_input("Contracts Traded", min_value=1, max_value=20, value=2, step=1)
oos_ratio = st.sidebar.slider("Out-of-Sample (OOS) Holdout %", min_value=10, max_value=50, value=30, step=5) / 100.0

# 2. Strategy Selection
st.sidebar.subheader("2. Trading Strategy")
available_strats = StrategyRegistry.list_available()
selected_strat = st.sidebar.selectbox("Strategy Setup", options=available_strats, index=0)

strat_params = {}
if selected_strat == "Value Area Reclaim":
    col_t1, col_t2 = st.sidebar.columns(2)
    with col_t1:
        strat_params["start_time"] = st.text_input("Start Time (ET)", value="10:00")
    with col_t2:
        strat_params["end_time"] = st.text_input("End Time (ET)", value="12:00")
    strat_params["target_mode"] = st.sidebar.selectbox("Target Mode", options=["opposite_va", "poc", "scale_50_50"], index=0, format_func=lambda x: {
        "opposite_va": "Opposite VA (Full Rotation)",
        "poc": "POC Only (Conservative)",
        "scale_50_50": "50% at POC + 50% at VAH/VAL"
    }[x])
    strat_params["stop_loss_buffer_atr"] = st.sidebar.slider("Stop Loss ATR Buffer", 0.0, 2.0, 0.5, 0.25)
    strat_params["require_vol_surge"] = st.sidebar.checkbox("Require Volume Surge (> SMA)", value=False)
    strat_params["eod_exit_time"] = st.sidebar.text_input("EOD Flat Time (ET)", value="15:55")

elif selected_strat == "Opening Range Breakout (ORB)":
    col_o1, col_o2 = st.sidebar.columns(2)
    with col_o1:
        strat_params["orb_start"] = st.text_input("ORB Start (ET)", value="09:30")
    with col_o2:
        strat_params["orb_end"] = st.text_input("ORB End (ET)", value="10:00")
    strat_params["trade_end"] = st.sidebar.text_input("Entry Window End (ET)", value="12:00")
    strat_params["r_target_mult"] = st.sidebar.slider("Target (R-Multiple)", 1.0, 5.0, 2.0, 0.5)
    strat_params["eod_exit_time"] = st.sidebar.text_input("EOD Flat Time (ET)", value="15:55")

# 3. Prop Firm Settings
st.sidebar.subheader("3. Prop Firm Optimization")
selected_preset = st.sidebar.selectbox("Prop Preset", options=list(PROP_PRESETS.keys()), index=0)
base_rules = PROP_PRESETS[selected_preset].copy()

# Override controls
custom_eval_risk = st.sidebar.slider("Eval Risk Multiplier", 0.25, 3.0, 1.0, 0.25, help="Scale trade sizing in evaluation phase to balance speed vs survival")
custom_funded_risk = st.sidebar.slider("Funded Risk Multiplier", 0.10, 2.0, 0.5, 0.10, help="Scale trade sizing down once funded to protect the drawdown buffer")
consistency_toggle = st.sidebar.checkbox("Enforce Consistency Rule (Max 50% on 1 Day)", value=True)

# ----------------- MAIN EXECUTION & CACHING -----------------
@st.cache_data(show_spinner=False)
def get_cached_data(symbol, timeframe):
    return load_asset_data(symbol=symbol, timeframe=timeframe)

@st.cache_data(show_spinner=False)
def get_cached_profiles(df):
    return compute_daily_volume_profiles(df)

st.title("🚀 Institutional Backtesting & Prop Firm Engine")
st.caption(f"Testing **{selected_strat}** on **{selected_asset}** with **{selected_preset}** Evaluation Rules")

with st.spinner("Loading dataset and volume profiles..."):
    df, spec = get_cached_data(selected_asset, selected_tf)
    daily_profiles = get_cached_profiles(df)
    is_mask, oos_mask, split_date = split_in_sample_out_sample(df, oos_ratio=oos_ratio)

# Run Strategy
strategy_fn = StrategyRegistry.get(selected_strat)
raw_signals = strategy_fn(df, daily_profiles, strat_params)

trades_df, metrics = process_trades(
    raw_signals,
    spec,
    oos_split_date=split_date,
    contracts=contracts
)

# ----------------- TABS DISPLAY -----------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Live Strategy Performance",
    "🎯 Prop Firm Evaluation (Pass Rate)",
    "💰 Funded Account & Payouts",
    "📜 Trade Log & Analytics"
])

# ================= TAB 1: LIVE STRATEGY PERFORMANCE =================
with tab1:
    m_all = metrics.get('all', {})
    m_is = metrics.get('in_sample', {})
    m_oos = metrics.get('out_sample', {})
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(f"<div class='metric-card'><div class='metric-lbl'>Total Net P&L</div><div class='metric-val'>${m_all.get('net_pnl', 0):,.2f}</div></div>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<div class='metric-card'><div class='metric-lbl'>Win Rate</div><div class='metric-val'>{m_all.get('win_rate', 0):.1f}%</div></div>", unsafe_allow_html=True)
    with col3:
        st.markdown(f"<div class='metric-card'><div class='metric-lbl'>Profit Factor</div><div class='metric-val'>{m_all.get('pf', 0):.2f}</div></div>", unsafe_allow_html=True)
    with col4:
        st.markdown(f"<div class='metric-card'><div class='metric-lbl'>Max Drawdown</div><div class='metric-val'>${m_all.get('max_dd', 0):,.2f}</div></div>", unsafe_allow_html=True)
    with col5:
        st.markdown(f"<div class='metric-card'><div class='metric-lbl'>Payoff Ratio</div><div class='metric-val'>{m_all.get('payoff', 0):.2f}:1</div></div>", unsafe_allow_html=True)
        
    st.markdown("### In-Sample vs. Out-of-Sample Robustness")
    col_comp1, col_comp2, col_comp3 = st.columns(3)
    with col_comp1:
        st.info(f"**Split Date:** {split_date} ({100-int(oos_ratio*100)}% IS / {int(oos_ratio*100)}% OOS)")
    with col_comp2:
        st.success(f"**In-Sample P&L:** ${m_is.get('net_pnl', 0):,.2f} | WR: {m_is.get('win_rate', 0):.1f}% | PF: {m_is.get('pf', 0):.2f}")
    with col_comp3:
        st.warning(f"**Out-of-Sample P&L:** ${m_oos.get('net_pnl', 0):,.2f} | WR: {m_oos.get('win_rate', 0):.1f}% | PF: {m_oos.get('pf', 0):.2f}")
        
    # Plotly Cumulative Equity Curve
    if len(trades_df) > 0:
        fig_eq = go.Figure()
        
        # Split trades
        is_trades = trades_df[trades_df['sample_type'] == 'In-Sample']
        oos_trades = trades_df[trades_df['sample_type'] == 'Out-of-Sample']
        
        fig_eq.add_trace(go.Scatter(
            x=is_trades['exit_time'],
            y=is_trades['cum_pnl'],
            mode='lines',
            name='In-Sample (Train)',
            line=dict(color='#00c087', width=2)
        ))
        
        if len(oos_trades) > 0:
            fig_eq.add_trace(go.Scatter(
                x=oos_trades['exit_time'],
                y=oos_trades['cum_pnl'],
                mode='lines',
                name='Out-of-Sample (Validation)',
                line=dict(color='#ff9f00', width=2, dash='solid')
            ))
            
        fig_eq.update_layout(
            title="Cumulative Net Equity Curve ($) with In-Sample vs. Out-of-Sample Holdout",
            xaxis_title="Date",
            yaxis_title="Net Profit ($)",
            template="plotly_dark",
            height=450,
            hovermode="x unified"
        )
        st.plotly_chart(fig_eq, use_container_width=True)

        # Drawdown Underwater Chart
        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(
            x=trades_df['exit_time'],
            y=trades_df['drawdown'],
            mode='lines',
            name='Drawdown ($)',
            fill='tozeroy',
            line=dict(color='#ef5350', width=1.5)
        ))
        fig_dd.update_layout(
            title="Underwater Drawdown Profile ($)",
            xaxis_title="Date",
            yaxis_title="Drawdown ($)",
            template="plotly_dark",
            height=250
        )
        st.plotly_chart(fig_dd, use_container_width=True)

# ================= TAB 2: PROP FIRM EVALUATION =================
with tab2:
    st.subheader(f"🎯 Prop Evaluation Simulation: {selected_preset}")
    st.write(f"Testing ability to hit **${base_rules['profit_target']:,} Profit Target** before hitting **${base_rules['max_drawdown']:,} {base_rules['dd_type']} Drawdown** or **${base_rules['daily_loss_limit']:,} Daily Loss Limit**.")
    
    if len(trades_df) > 10:
        sim_res = run_monte_carlo_prop_simulation(
            trades_df,
            base_rules,
            eval_risk_scaler=custom_eval_risk,
            funded_risk_scaler=custom_funded_risk,
            num_simulations=1000,
            consistency_threshold=0.50 if consistency_toggle else None
        )
        
        col_p1, col_p2, col_p3, col_p4 = st.columns(4)
        with col_p1:
            st.metric("Evaluation Pass Rate", f"{sim_res['pass_rate_pct']:.1f}%")
        with col_p2:
            if base_rules.get("dll_type") == "soft":
                st.metric("Daily Lockouts Hit (%)", f"{sim_res.get('soft_lockout_pct', 0.0):.1f}%", help="Soft daily lockout reached: positions flattened, account survives to next session")
            else:
                st.metric("Blown by Daily Loss Limit", f"{sim_res['blown_dll_pct']:.1f}%")
        with col_p3:
            st.metric("Blown by Max Trailing DD", f"{sim_res['blown_max_dd_pct']:.1f}%")
        with col_p4:
            st.metric("Median Days to Pass", f"{sim_res['median_days_to_pass']:.0f} trading days")
            
        if base_rules.get("dll_type") == "soft":
            st.caption("ℹ️ **Rule Notice:** Topstep Daily Loss Limit ($1,000) is enforced as a **Soft Breach / Daily Lockout**. Position is liquidated and trading resumes next session. Account is only terminated if cumulative trailing drawdown ($2,000) is breached.")
            
        # Distribution of Days to Pass
        eval_df = sim_res['eval_df']
        passed_trials = eval_df[eval_df['status'] == 'PASSED']
        
        if len(passed_trials) > 0:
            fig_hist = px.histogram(
                passed_trials,
                x="days",
                nbins=30,
                title="Distribution of Trading Days Required to Pass Evaluation",
                labels={"days": "Trading Days to Pass"},
                template="plotly_dark",
                color_discrete_sequence=['#00c087']
            )
            fig_hist.update_layout(height=350)
            st.plotly_chart(fig_hist, use_container_width=True)
            
        # Risk Sensitivity Matrix
        st.markdown("### 🔬 Sizing Sensitivity: How Risk Scaler Affects Pass Rate")
        sens_df = calculate_risk_sensitivity_matrix(trades_df, base_rules)
        st.dataframe(sens_df, use_container_width=True)

# ================= TAB 3: FUNDED STAGE & PAYOUTS =================
with tab3:
    st.subheader("💰 Funded Account Survival & Payout Modeling")
    st.markdown("""
    Once an evaluation is passed, you enter the **Funded Stage**. 
    Here, the goal shifts from hitting an aggressive profit target to **protecting the drawdown buffer and collecting recurring payouts**.
    """)
    
    if len(trades_df) > 10:
        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            st.metric("Funded Payout Rate", f"{sim_res['funded_stage_payout_rate_pct']:.1f}%", help="% of funded accounts that achieve at least 1 payout before blowing")
        with col_f2:
            st.metric("Avg Payout Collected", f"${sim_res['avg_payouts_usd']:,.2f}")
        with col_f3:
            ev_color = "normal" if sim_res['net_ev_per_challenge'] > 0 else "inverse"
            st.metric("Net EV per Evaluation Purchased", f"${sim_res['net_ev_per_challenge']:+,.2f}", help="Avg Payout minus Challenge Cost ($49)")
            
        st.info(f"💡 **Funded Risk Scaling Strategy:** You are currently risking **{custom_eval_risk:.2f}x** during Evaluation and scaling down to **{custom_funded_risk:.2f}x** during the Funded stage. Lowering risk in funded mode significantly prolongs account survival.")

# ================= TAB 4: TRADE LOG =================
with tab4:
    st.subheader("📜 Detailed Trade Log")
    if len(trades_df) > 0:
        display_cols = ['entry_time', 'exit_time', 'side', 'entry_price', 'exit_price', 'net_pts', 'net_pnl', 'sample_type', 'exit_reason']
        st.dataframe(trades_df[display_cols].sort_values('entry_time', ascending=False), use_container_width=True)
    else:
        st.warning("No trades generated with the selected parameters.")
