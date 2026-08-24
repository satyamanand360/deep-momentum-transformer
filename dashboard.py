import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import os
import calendar

st.set_page_config(page_title="Momentum Transformer", layout="wide", page_icon="📈")

# ── Custom CSS for a premium look ─────────────────────────────────────────────
st.markdown("""
    <style>
    .metric-container {
        background-color: #1E1E1E;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #333;
        margin-bottom: 20px;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #00C49A;
    }
    .metric-label {
        font-size: 14px;
        color: #AAAAAA;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("📈 Momentum Transformer Dashboard")
st.markdown("Advanced Deep Momentum Network with Attention (Decoder-Only TFT).")

# ── Sidebar Controls ─────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Settings")

mode = st.sidebar.radio("Data Source", ["Historical Backtest (2020-2024)", "Live Inference (2023-Present)"])

@st.cache_data
def load_data(mode_name):
    if "Live" in mode_name:
        port_file = "results/live_portfolio_returns.csv"
        bench_file = "results/live_benchmark_returns.csv"
        pos_file = "results/live_positions.csv"
    else:
        port_file = "results/portfolio_returns.csv"
        bench_file = "results/benchmark_returns.csv"
        pos_file = None

    if not os.path.exists(port_file) or not os.path.exists(bench_file):
        return None, None, None

    port = pd.read_csv(port_file, index_col=0)
    bench = pd.read_csv(bench_file, index_col=0)
    port.index = pd.to_datetime(port.index)
    bench.index = pd.to_datetime(bench.index)
    
    pos_df = None
    if pos_file and os.path.exists(pos_file):
        pos_df = pd.read_csv(pos_file)

    return port.iloc[:, 0], bench.iloc[:, 0], pos_df

port_ret_full, bench_ret_full, pos_df = load_data(mode)

if port_ret_full is None:
    st.error(f"Could not find results for {mode}. Please run the corresponding script first.")
    st.stop()

# Date Range Filter
min_date = port_ret_full.index.min().date()
max_date = port_ret_full.index.max().date()

st.sidebar.markdown("### Filter Dates")
date_range = st.sidebar.date_input("Select Range", value=(min_date, max_date), min_value=min_date, max_value=max_date)

if len(date_range) == 2:
    start_date, end_date = date_range
    port_ret = port_ret_full[(port_ret_full.index.date >= start_date) & (port_ret_full.index.date <= end_date)]
    bench_ret = bench_ret_full[(bench_ret_full.index.date >= start_date) & (bench_ret_full.index.date <= end_date)]
else:
    port_ret, bench_ret = port_ret_full, bench_ret_full

if len(port_ret) == 0:
    st.warning("No data in selected date range.")
    st.stop()

# ── Metrics Calculation ──────────────────────────────────────────────────────
def calc_all_metrics(strat, bench):
    ann = 252
    
    # Strategy
    s_ret = strat.mean() * ann
    s_vol = strat.std() * np.sqrt(ann)
    s_sharpe = s_ret / (s_vol + 1e-8)
    s_down = strat[strat < 0].std() * np.sqrt(ann)
    s_sortino = s_ret / (s_down + 1e-8)
    s_cum = (1 + strat).cumprod()
    s_dd = ((s_cum - s_cum.cummax()) / s_cum.cummax()).min()
    s_calmar = s_ret / abs(s_dd + 1e-8)
    
    # Benchmark
    b_ret = bench.mean() * ann
    b_vol = bench.std() * np.sqrt(ann)
    b_sharpe = b_ret / (b_vol + 1e-8)
    b_cum = (1 + bench).cumprod()
    b_dd = ((b_cum - b_cum.cummax()) / b_cum.cummax()).min()
    
    # Relative
    cov = np.cov(strat, bench)[0, 1]
    beta = cov / (bench.var() + 1e-8)
    alpha = s_ret - (beta * b_ret)
    
    active_ret = strat - bench
    te = active_ret.std() * np.sqrt(ann)
    ir = (s_ret - b_ret) / (te + 1e-8)
    
    return {
        "Return": (s_ret, b_ret),
        "Vol": (s_vol, b_vol),
        "Sharpe": (s_sharpe, b_sharpe),
        "Sortino": (s_sortino, None),
        "Max DD": (s_dd, b_dd),
        "Calmar": (s_calmar, None),
        "Win Rate": ((strat > 0).mean(), (bench > 0).mean()),
        "IR": (ir, None),
        "Beta": (beta, None),
        "Alpha": (alpha, None)
    }

m = calc_all_metrics(port_ret, bench_ret)

def metric_card(label, val, bench_val=None, is_pct=False, is_inverted=False):
    if is_pct:
        v_str = f"{val*100:.2f}%"
        b_str = f"{bench_val*100:.2f}%" if bench_val is not None else ""
    else:
        v_str = f"{val:.2f}"
        b_str = f"{bench_val:.2f}" if bench_val is not None else ""
        
    color = "#00C49A"
    if is_inverted and val < 0:
        color = "#FF6B6B"
    elif not is_inverted and val < 0:
        color = "#FF6B6B"
        
    vs_text = f"<div style='font-size: 12px; color: #888; margin-top: 5px;'>Bench: {b_str}</div>" if bench_val is not None else ""
    
    return f"""
    <div class="metric-container">
        <div class="metric-label">{label}</div>
        <div class="metric-value" style="color: {color};">{v_str}</div>
        {vs_text}
    </div>
    """

st.markdown("### 📊 Performance Overview")
c1, c2, c3, c4, c5 = st.columns(5)
c1.markdown(metric_card("Ann Return", m["Return"][0], m["Return"][1], True), unsafe_allow_html=True)
c2.markdown(metric_card("Sharpe Ratio", m["Sharpe"][0], m["Sharpe"][1]), unsafe_allow_html=True)
c3.markdown(metric_card("Max Drawdown", m["Max DD"][0], m["Max DD"][1], True, True), unsafe_allow_html=True)
c4.markdown(metric_card("Sortino", m["Sortino"][0]), unsafe_allow_html=True)
c5.markdown(metric_card("Calmar", m["Calmar"][0]), unsafe_allow_html=True)

c6, c7, c8, c9, c10 = st.columns(5)
c6.markdown(metric_card("Volatility", m["Vol"][0], m["Vol"][1], True), unsafe_allow_html=True)
c7.markdown(metric_card("Win Rate", m["Win Rate"][0], m["Win Rate"][1], True), unsafe_allow_html=True)
c8.markdown(metric_card("Info Ratio", m["IR"][0]), unsafe_allow_html=True)
c9.markdown(metric_card("Beta", m["Beta"][0]), unsafe_allow_html=True)
c10.markdown(metric_card("Alpha (Ann)", m["Alpha"][0], None, True), unsafe_allow_html=True)


# ── Interactive Charts ───────────────────────────────────────────────────────
st.markdown("---")

t1, t2, t3, t4 = st.tabs(["📈 Equity Curve", "📉 Drawdown", "🔄 Rolling Sharpe", "📅 Monthly Heatmap"])

with t1:
    s_cum = (1 + port_ret).cumprod()
    b_cum = (1 + bench_ret).cumprod()
    fig_eq = go.Figure()
    fig_eq.add_trace(go.Scatter(x=s_cum.index, y=s_cum, mode='lines', name='Momentum Transformer (30% Target)', line=dict(color='#00C49A', width=2)))
    fig_eq.add_trace(go.Scatter(x=b_cum.index, y=b_cum, mode='lines', name='Benchmark (Equal Weight)', line=dict(color='#FF6B6B', width=2, dash='dash')))
    fig_eq.update_layout(height=500, title="Cumulative Equity Curve", template="plotly_dark", hovermode="x unified", legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
    st.plotly_chart(fig_eq, use_container_width=True)

with t2:
    s_dd = (s_cum - s_cum.cummax()) / s_cum.cummax()
    b_dd = (b_cum - b_cum.cummax()) / b_cum.cummax()
    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(x=s_dd.index, y=s_dd, mode='lines', fill='tozeroy', name='Strategy Drawdown', line=dict(color='#00C49A', width=1)))
    fig_dd.add_trace(go.Scatter(x=b_dd.index, y=b_dd, mode='lines', fill='tozeroy', name='Benchmark Drawdown', line=dict(color='#FF6B6B', width=1)))
    fig_dd.update_layout(height=400, title="Historical Drawdowns", template="plotly_dark", hovermode="x unified")
    st.plotly_chart(fig_dd, use_container_width=True)

with t3:
    # 6-month rolling (126 days)
    roll_s = port_ret.rolling(126).mean() * 252
    roll_v = port_ret.rolling(126).std() * np.sqrt(252)
    roll_sharpe = roll_s / (roll_v + 1e-8)
    
    roll_b_s = bench_ret.rolling(126).mean() * 252
    roll_b_v = bench_ret.rolling(126).std() * np.sqrt(252)
    roll_b_sharpe = roll_b_s / (roll_b_v + 1e-8)
    
    fig_rs = go.Figure()
    fig_rs.add_trace(go.Scatter(x=roll_sharpe.index, y=roll_sharpe, mode='lines', name='Strategy 6M Sharpe', line=dict(color='#00C49A')))
    fig_rs.add_trace(go.Scatter(x=roll_b_sharpe.index, y=roll_b_sharpe, mode='lines', name='Benchmark 6M Sharpe', line=dict(color='#FF6B6B', dash='dot')))
    fig_rs.add_hline(y=0, line_dash="dash", line_color="gray")
    fig_rs.update_layout(height=400, title="Rolling 6-Month Sharpe Ratio", template="plotly_dark")
    st.plotly_chart(fig_rs, use_container_width=True)

with t4:
    # Monthly returns heatmap
    df_m = port_ret.resample('ME').apply(lambda x: (1+x).prod() - 1).to_frame(name='Return')
    df_m['Year'] = df_m.index.year
    df_m['Month'] = df_m.index.month
    pivot = df_m.pivot(index='Year', columns='Month', values='Return')
    pivot.columns = [calendar.month_abbr[i] for i in pivot.columns]
    
    # Add YTD
    pivot['YTD'] = df_m.groupby('Year')['Return'].apply(lambda x: (1+x).prod() - 1)
    
    fig_hm = px.imshow(pivot, text_auto=".2%", aspect="auto", 
                       color_continuous_scale="RdYlGn", color_continuous_midpoint=0,
                       title="Strategy Monthly Returns (%)")
    fig_hm.update_layout(height=400, template="plotly_dark")
    st.plotly_chart(fig_hm, use_container_width=True)

# ── Today's Positions (Live Mode Only) ───────────────────────────────────────
if "Live" in mode and pos_df is not None:
    st.markdown("---")
    st.markdown("### 🎯 Today's Target Positions")
    st.markdown("Most recent portfolio allocation based on live model inference.")
    
    def color_dir(val):
        if isinstance(val, str):
            if 'LONG' in val: return 'color: #00C49A; font-weight: bold'
            elif 'SHORT' in val: return 'color: #FF6B6B; font-weight: bold'
        return 'color: #AAAAAA'
        
    def label_dir(val):
        if val > 0.15: return 'LONG 🔼'
        elif val < -0.15: return 'SHORT 🔽'
        return 'FLAT ➖'
        
    pos_show = pos_df.copy()
    pos_show['Direction'] = pos_show['z'].apply(label_dir)
    pos_show = pos_show.sort_values('z', ascending=False)[['Ticker', 'Direction', 'z', 'scaled_z']]
    pos_show.columns = ['Ticker', 'Direction', 'Raw Signal (z)', 'Target Weight (%)']
    pos_show['Target Weight (%)'] = pos_show['Target Weight (%)'] * 100
    
    st.dataframe(
        pos_show.style.map(lambda x: color_dir(x), subset=['Direction'])\
                      .format({'Raw Signal (z)': '{:.4f}', 'Target Weight (%)': '{:.2f}%'}),
        height=400, use_container_width=True
    )
