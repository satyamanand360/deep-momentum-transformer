"""
live_test.py -- Run the trained Momentum Transformer on live 2024-2026 data.
Uses the existing MomentumDataset + DataLoader for fast batched GPU inference.
Does NOT modify any existing file.
"""
import torch
import pandas as pd
import numpy as np
import os
from datetime import datetime

from model import MomentumTransformer
from dataset import MomentumDataset
from torch.utils.data import DataLoader

SEQ_LENGTH = 252
VOL_TARGET = 0.30
COST_BPS   = 3


def calculate_metrics(returns):
    ann = 252
    ann_ret = returns.mean() * ann
    ann_vol = returns.std() * np.sqrt(ann)
    sharpe  = ann_ret / (ann_vol + 1e-8)
    down    = returns[returns < 0].std() * np.sqrt(ann)
    sortino = ann_ret / (down + 1e-8)
    cum = (1 + returns).cumprod()
    dd  = ((cum - cum.cummax()) / cum.cummax()).min()
    return {
        'Annualized Return':    ann_ret,
        'Annualized Volatility': ann_vol,
        'Sharpe Ratio':         sharpe,
        'Sortino Ratio':        sortino,
        'Max Drawdown':         dd,
        'Win Rate':             (returns > 0).mean()
    }


def run_live_test(features_path="data/live_features.csv",
                  model_path="models/best_model.pth"):

    today_str = datetime.today().strftime("%Y-%m-%d")
    print("=" * 60)
    print(f"  Momentum Transformer - Live Inference [{today_str}]")
    print("=" * 60)

    if not os.path.exists(features_path):
        print("ERROR: Run 'python live_data.py' first.")
        return

    # Force CPU to avoid OOM conflicts with the dashboard holding GPU memory
    device = torch.device('cpu')
    print(f"  Device: {device}")

    # ── Load model ────────────────────────────────────────────────────────────
    model = MomentumTransformer(num_features=8, d_model=64, n_heads=4)
    if not os.path.exists(model_path):
        print("ERROR: Model not found. Train first.")
        return

    state_dict = torch.load(model_path, map_location=device)
    missing, _ = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  [INFO] New layer weights randomly initialized: {missing}")

    model.to(device).eval()

    # ── Build Dataset using existing MomentumDataset (step=1, sliding window) ─
    df = pd.read_csv(features_path)
    df['Date'] = pd.to_datetime(df['Date'])

    # Drop rows where Target_Return is NaN (last row per ticker -- today)
    # We keep them separately for the "today's position" calculation
    df_hist   = df.dropna(subset=['Target_Return']).copy()
    df_latest = df.groupby('Ticker').last().reset_index()   # Last row per ticker

    # --- Historical inference via DataLoader (fast) ---
    dataset = MomentumDataset(df_hist, seq_length=SEQ_LENGTH, step=1)
    loader  = DataLoader(dataset, batch_size=128, shuffle=False)

    print(f"\n  Running inference on {len(dataset)} sequences...")

    results = []
    tgt_vol_daily = VOL_TARGET / np.sqrt(252)

    with torch.no_grad():
        for batch in loader:
            X       = batch['X'].to(device)
            y       = batch['y'].to(device)
            vol     = batch['vol'].to(device)
            dates   = batch['date']
            tickers = batch['ticker']

            z_seq, _, _ = model(X)           # (batch, seq)
            z   = z_seq[:, -1]               # last-day position
            y_t = y[:, -1]
            v_t = vol[:, -1]

            scaled_z = z * (tgt_vol_daily / (v_t + 1e-8))

            for i in range(len(z)):
                results.append({
                    'Date':       dates[i],
                    'Ticker':     tickers[i],
                    'z':          z[i].item(),
                    'scaled_z':   scaled_z[i].item(),
                    'raw_return': y_t[i].item(),
                    'vol':        v_t[i].item()
                })

    res = pd.DataFrame(results)
    res['Date'] = pd.to_datetime(res['Date'])
    res = res.sort_values(['Ticker', 'Date'])

    # Transaction costs per ticker
    res['prev_z']   = res.groupby('Ticker')['scaled_z'].shift(1).fillna(0)
    res['turnover'] = (res['scaled_z'] - res['prev_z']).abs()
    cost_dec        = COST_BPS / 10000.0
    res['gross']    = res['scaled_z'] * res['raw_return']
    res['net']      = res['gross'] - res['turnover'] * cost_dec

    # --- TODAY'S positions (last available sequence per ticker) ---
    print("\n" + "-" * 60)
    print(f"  TODAY'S POSITIONS  (model signal as of {today_str})")
    print("-" * 60)
    print(f"  {'Ticker':<10} {'Signal z':>10} {'Direction':>12}")
    print(f"  {'-'*35}")

    # For today's position, just take the last known z per ticker from results
    today_pos = res.groupby('Ticker').last().reset_index()[['Ticker', 'z', 'scaled_z']]
    today_pos = today_pos.sort_values('z', ascending=False)

    for _, row in today_pos.iterrows():
        if row['z'] > 0.15:
            direction = "LONG  [^]"
        elif row['z'] < -0.15:
            direction = "SHORT [v]"
        else:
            direction = "FLAT  [-]"
        print(f"  {row['Ticker']:<10} {row['z']:>10.4f} {direction:>12}")

    today_pos.to_csv("results/live_positions.csv", index=False)

    # --- Recent 90-day live performance ---
    latest_date = res['Date'].max()
    cutoff      = latest_date - pd.Timedelta(days=90)
    recent      = res[(res['Date'] > cutoff)]

    if len(recent) > 10:
        port  = recent.groupby('Date')['net'].mean()
        bench = recent.groupby('Date')['raw_return'].mean()

        # Apply vol targeting leverage
        rv  = port.std() * np.sqrt(252)
        lev = VOL_TARGET / (rv + 1e-8)
        port = port * lev

        sm = calculate_metrics(port)
        bm = calculate_metrics(bench)

        print("\n" + "-" * 60)
        print(f"  LIVE 90-DAY PERFORMANCE")
        print(f"  ({cutoff.strftime('%Y-%m-%d')} to {latest_date.strftime('%Y-%m-%d')})")
        print("-" * 60)
        print(f"  {'Metric':<26} {'Strategy':>10} {'Benchmark':>10}")
        print(f"  {'-'*48}")
        for k in sm:
            sv, bv = sm[k], bm[k]
            if k in ['Annualized Return', 'Annualized Volatility', 'Max Drawdown', 'Win Rate']:
                print(f"  {k:<26} {sv*100:>9.2f}%  {bv*100:>9.2f}%")
            else:
                print(f"  {k:<26} {sv:>10.4f}  {bv:>10.4f}")

        os.makedirs("results", exist_ok=True)
        port.to_csv("results/live_portfolio_returns.csv")
        bench.to_csv("results/live_benchmark_returns.csv")
        print("\n  Saved -> results/live_positions.csv")
        print("  Saved -> results/live_portfolio_returns.csv")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    run_live_test()
