# 📈 Prop Backtest Engine

An institutional-grade algorithmic trading backtesting and prop firm challenge optimization platform built with Python and Streamlit.

Designed for quantitative futures (NQ/MNQ) and crypto traders to backtest intraday technical strategies (such as Volume Profile Value Area Reclaims and Opening Range Breakouts) while directly modeling prop firm risk parameters (**Topstep, Apex, Static Drawdown**).

---

## 🌟 Key Features

- **Interactive Streamlit Web Dashboard**: Full interactive visualization with Plotly equity curves, underwater drawdown profiles, Monte Carlo pass distributions, and risk sensitivity heatmaps.
- **Audited Market Realism**:
  - **Zero Lookahead Bias**: Volume profiles strictly built from prior CME sessions ($D-1$); causal rolling indicators.
  - **CME Futures Session Timing**: Handles Sunday 18:00 ET trade date boundaries accurately.
  - **Intrabar Path Precision**: Cross-validated against 977,000+ bars of 1-minute ground-truth tick data to eliminate bar-level exit ambiguity.
  - **Slippage & Commission Modeling**: Realistic frictional drag applied to every fill ($2/pt for MNQ, $20/pt for NQ).
- **Prop Firm Evaluation Simulator**:
  - **Presets**: Topstep ($50k, $100k, $150k), Apex ($50k), and Static Drawdown ($50k).
  - **Accurate Rules**: Models End-of-Day (EOD) trailing drawdowns, intraday peak trailing, and Topstep's real **Soft Daily Loss Lockout** (auto-flattens position and locks for the session without terminating the account).
  - **Monte Carlo Analysis**: 1,000-trial simulation providing empirical pass rates, median days to pass, payout probabilities, and challenge Net Expected Value (Net EV).
  - **Dual-Stage Risk Scaling**: Optimization matrix for aggressive sizing during evaluation vs. conservative capital preservation during the funded stage.

---

## 🚀 Quickstart & Local Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Wallaby-Alpha/prop-backtest-engine.git
cd prop-backtest-engine
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch the Web Application
```bash
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`.

---

## 🧪 Running CLI Verification & Audits

Run the end-to-end backtest and prop simulation directly from the terminal:
```bash
python test_engine_cli.py
```

Run the forensic intrabar ground-truth verification (15m vs 1m bars):
```bash
python intrabar_lookahead_audit.py
```

---

## 📂 Repository Structure

```
├── app.py                         # Main Streamlit web application
├── engine/                        # Core algorithmic engine package
│   ├── __init__.py
│   ├── data_manager.py            # CME sessions, volume profiles, IS/OOS splits
│   ├── strategies.py              # Strategy Registry (Value Area Reclaim, ORB)
│   ├── execution_engine.py        # P&L accounting, contract multipliers, slippage
│   └── prop_simulator.py          # Prop firm rules, Monte Carlo, payout model
├── data_cache/                    # Standardized Parquet OHLCV datasets
│   ├── mnq_15m.parquet            # 3 years of 15m MNQ futures (70k bars)
│   ├── mnq_5m.parquet             # 5m MNQ futures (205k bars)
│   ├── mnq_1m.parquet             # 1m MNQ futures (977k bars)
│   └── btc_15m.parquet            # 15m BTC crypto data
├── test_engine_cli.py             # CLI runner for end-to-end testing
├── intrabar_lookahead_audit.py    # Forensic 1m vs 15m audit script
├── audit_engine.py                # Prop rules and CME session verification
├── requirements.txt               # App dependencies
└── README.md                      # Documentation
```

---

## ☁️ Deploying to Streamlit Community Cloud

1. Fork or push this repository to GitHub (`Wallaby-Alpha/prop-backtest-engine`).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **New App**, select repository `Wallaby-Alpha/prop-backtest-engine`, branch `main`, and main file path `app.py`.
4. Click **Deploy!** The dataset is already included in `data_cache/`, so it works immediately.
