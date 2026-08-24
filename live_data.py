"""
live_data.py — Download and feature-engineer fresh data up to today.
Does NOT modify any existing data files. Writes to data/live_features.csv.
"""
import yfinance as yf
import pandas as pd
import numpy as np
import os
from datetime import datetime

# ── Same 70 tickers as in data.py ──────────────────────────────────────────
TICKERS = [
    'SPY', 'QQQ', 'IWM', 'EFA', 'EEM',
    'TLT', 'IEF', 'SHY', 'LQD', 'HYG',
    'GLD', 'SLV', 'USO', 'UNG', 'DBA',
    'VNQ', 'XLF', 'XLK', 'XLV', 'XLE',
    'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'NVDA', 'TSLA', 'BRK-B', 'UNH', 'JNJ',
    'JPM', 'V', 'PG', 'MA', 'HD', 'CVX', 'MRK', 'ABBV', 'LLY', 'PEP',
    'KO', 'BAC', 'AVGO', 'TMO', 'COST', 'MCD', 'CSCO', 'WMT', 'CRM', 'DIS',
    'PFE', 'DHR', 'NKE', 'XOM', 'TXN', 'UNP', 'LIN', 'ADBE', 'VZ', 'PM',
    'NEE', 'RTX', 'QCOM', 'INTC', 'AMD', 'HON', 'UPS', 'AMGN', 'IBM', 'BA'
]

# We need at least 252 days of history BEFORE the live window for the model's
# sequence. So we pull from 2023-01-01 to give plenty of warm-up history.
LIVE_START = "2023-01-01"
LIVE_END   = datetime.today().strftime("%Y-%m-%d")


def compute_macd(close, short_window, long_window):
    ewm_short = close.ewm(span=short_window, adjust=False).mean()
    ewm_long  = close.ewm(span=long_window,  adjust=False).mean()
    return ewm_short - ewm_long


def compute_live_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identical feature engineering to features.py — kept in sync manually.
    Output columns exactly match what the model was trained on.
    """
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(['Ticker', 'Date']).reset_index(drop=True)

    parts = []
    for ticker, grp in df.groupby('Ticker'):
        grp = grp.copy().reset_index(drop=True)

        grp['Return_1d'] = grp['Close'].pct_change()

        # EWMA vol (same as features.py)
        lam = 1.0 - 1.0 / 60.0
        sq_rets = grp['Return_1d'] ** 2
        ewma_var = sq_rets.ewm(alpha=1 - lam, adjust=False).mean()
        grp['Vol_60d'] = np.sqrt(ewma_var).ffill()

        # Multi-horizon returns
        for h in [1, 21, 63, 126, 252]:
            grp[f'Return_{h}d'] = grp['Close'].pct_change(periods=h)
            grp[f'Return_{h}d_scaled'] = grp[f'Return_{h}d'] / (grp['Vol_60d'] * np.sqrt(h) + 1e-8)

        # MACD signals
        for s, l in [(8, 24), (16, 28), (32, 96)]:
            grp[f'MACD_{s}_{l}'] = compute_macd(grp['Close'], s, l)
            grp[f'MACD_{s}_{l}_scaled'] = grp[f'MACD_{s}_{l}'] / (grp['Close'] * grp['Vol_60d'] * np.sqrt(l) + 1e-8)

        # Target: next day return (NaN on the final row — that's today's prediction day)
        grp['Target_Return'] = grp['Return_1d'].shift(-1)

        parts.append(grp)

    out = pd.concat(parts, ignore_index=True)
    # Drop rows missing *feature* columns only; keep last row even if Target_Return is NaN
    feature_cols = [
        'Return_1d_scaled', 'Return_21d_scaled', 'Return_63d_scaled',
        'Return_126d_scaled', 'Return_252d_scaled',
        'MACD_8_24_scaled', 'MACD_16_28_scaled', 'MACD_32_96_scaled',
        'Vol_60d'
    ]
    out = out.dropna(subset=feature_cols).reset_index(drop=True)
    return out


def download_live_data():
    today = datetime.today().strftime("%Y-%m-%d")
    print(f"Downloading live data: {LIVE_START} to {today}")
    raw = yf.download(TICKERS, start=LIVE_START, end=today)
    raw = raw.stack(level=1).rename_axis(['Date', 'Ticker']).reset_index()

    os.makedirs("data", exist_ok=True)
    raw.to_csv("data/live_raw.csv", index=False)
    print(f"Raw live data saved -> data/live_raw.csv  ({len(raw)} rows)")

    features = compute_live_features(raw)
    features.to_csv("data/live_features.csv", index=False)
    print(f"Live features saved  → data/live_features.csv  ({len(features)} rows)")
    return features


if __name__ == "__main__":
    download_live_data()
