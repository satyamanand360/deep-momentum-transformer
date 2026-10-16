"""
live_test.py -- Run the trained Momentum Transformer on live data (2024 onward).

* Live track record: every day whose next-day return is already known.
* Today's positions: the window ending on the LATEST date (whose return is not
  known yet), so the signal uses today's close, not yesterday's.
Run after the US market close so the latest bar is a final close.
Does NOT modify any training files.
"""
import os
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import MomentumDataset
from backtest import load_model
from train import predict_last_step
from portfolio import (asset_returns, equal_weight_portfolio, apply_leverage,
                       calculate_metrics, scale_positions, PORTFOLIO_VOL_TARGET, ANN)

SEQ_LENGTH = 252
COST_BPS = 3
RECENT_DAYS = 90
DIRECTION_BAND = 0.15   # display only: |z| below this is shown as FLAT


def direction(z):
    if z > DIRECTION_BAND:
        return "LONG"
    if z < -DIRECTION_BAND:
        return "SHORT"
    return "FLAT"


def run_live_test(features_path="data/live_features.csv", model_path="models/best_model.pth"):
    today_str = datetime.today().strftime("%Y-%m-%d")
    print("=" * 64)
    print(f"  Momentum Transformer - Live Inference [{today_str}]")
    print("=" * 64)

    if not os.path.exists(features_path):
        print("ERROR: Run 'python live_data.py' first.")
        return

    # CPU avoids GPU memory clashes with a running dashboard
    device = torch.device('cpu')
    model = load_model(model_path, device)

    df = pd.read_csv(features_path)
    df['Date'] = pd.to_datetime(df['Date'])
    labelled = df['Target_Return'].notna()
    # The latest row per ticker has no next-day return yet. Fill it with 0 only so
    # the dataset can hold it; that value is never used for the position.
    df_all = df.assign(Target_Return=df['Target_Return'].fillna(0.0))

    dataset = MomentumDataset(df_all, seq_length=SEQ_LENGTH, step=1)
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    print(f"  Running inference on {len(dataset)} windows...")
    preds = predict_last_step(model, loader, device)
    preds['Date'] = pd.to_datetime(preds['Date'])

    unlabelled = df.loc[~labelled, ['Date', 'Ticker']].assign(_latest=True)
    preds = preds.merge(unlabelled, on=['Date', 'Ticker'], how='left')
    is_latest = preds.pop('_latest').fillna(False).astype(bool)
    hist, latest = preds[~is_latest], preds[is_latest]

    # ---- Live track record (decisions whose outcome is known) ----------------
    assets = asset_returns(hist, cost_bps=COST_BPS)
    port_raw = equal_weight_portfolio(assets)
    port, lev = apply_leverage(port_raw, mode="ex_ante", target_vol=PORTFOLIO_VOL_TARGET)
    bench = hist.groupby('Date')['raw_return'].mean().loc[port.index]

    # ---- Today's positions ---------------------------------------------------
    # Leverage for the next session uses every portfolio return realised so far
    vol_now = port_raw.ewm(span=60, min_periods=20).std().iloc[-1] * np.sqrt(ANN)
    lev_today = PORTFOLIO_VOL_TARGET / (vol_now + 1e-8)
    today_pos = latest[['Date', 'Ticker', 'z']].copy()
    today_pos['scaled_z'] = scale_positions(latest['z'].values, latest['vol'].values)
    # Actual portfolio weight: vol-targeted position x portfolio leverage / N
    today_pos['portfolio_weight'] = today_pos['scaled_z'] * lev_today / max(len(today_pos), 1)
    today_pos['Direction'] = today_pos['z'].apply(direction)
    today_pos = today_pos.sort_values('z', ascending=False)

    signal_date = today_pos['Date'].max().date() if len(today_pos) else None
    print("\n" + "-" * 64)
    print(f"  POSITIONS FOR THE NEXT SESSION (signal from the {signal_date} close)")
    print(f"  Portfolio leverage {lev_today:.2f}x to target {PORTFOLIO_VOL_TARGET:.0%} vol")
    print("-" * 64)
    print(f"  {'Ticker':<8} {'Signal z':>9} {'Direction':>10} {'Weight':>9}")
    for _, row in today_pos.iterrows():
        print(f"  {row['Ticker']:<8} {row['z']:>9.4f} {row['Direction']:>10} {row['portfolio_weight']:>8.2%}")

    os.makedirs("results", exist_ok=True)
    today_pos.to_csv("results/live_positions.csv", index=False)
    port.to_csv("results/live_portfolio_returns.csv")
    bench.to_csv("results/live_benchmark_returns.csv")

    # ---- Recent performance (short windows are anecdotal) --------------------
    cutoff = port.index.max() - pd.Timedelta(days=RECENT_DAYS)
    recent_s, recent_b = port[port.index > cutoff], bench[bench.index > cutoff]
    if len(recent_s) > 10:
        sm, bm = calculate_metrics(recent_s), calculate_metrics(recent_b)
        print("\n" + "-" * 64)
        print(f"  LIVE {RECENT_DAYS}-DAY PERFORMANCE ({cutoff.date()} to {port.index.max().date()})")
        print(f"  Strategy levered ex-ante to {PORTFOLIO_VOL_TARGET:.0%} vol; benchmark unlevered")
        print("-" * 64)
        print(f"  {'Metric':<24} {'Strategy':>10} {'Benchmark':>10}")
        for k in ['Annualized Return', 'Annualized Volatility', 'Sharpe Ratio',
                  'Sortino Ratio', 'Max Drawdown', 'Win Rate']:
            sv, bv = sm[k], bm[k]
            if k in ['Annualized Return', 'Annualized Volatility', 'Max Drawdown', 'Win Rate']:
                print(f"  {k:<24} {sv*100:>9.2f}% {bv*100:>9.2f}%")
            else:
                print(f"  {k:<24} {sv:>10.4f} {bv:>10.4f}")

    full = calculate_metrics(port)
    print(f"\n  Full live record since {port.index.min().date()}: Sharpe {full['Sharpe Ratio']:.2f} "
          f"over {full['Days']} days")
    print("\n  Saved -> results/live_positions.csv, live_portfolio_returns.csv, live_benchmark_returns.csv")
    print("=" * 64)


if __name__ == "__main__":
    run_live_test()
