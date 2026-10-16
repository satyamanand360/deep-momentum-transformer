import pandas as pd
import numpy as np
import os

def compute_macd(close, short_window, long_window):
    """Compute MACD signal for a given short and long window."""
    ewm_short = close.ewm(span=short_window, adjust=False).mean()
    ewm_long = close.ewm(span=long_window, adjust=False).mean()
    return ewm_short - ewm_long

FEATURE_COLS = [
    'Return_1d_scaled', 'Return_21d_scaled', 'Return_63d_scaled',
    'Return_126d_scaled', 'Return_252d_scaled',
    'MACD_8_24_scaled', 'MACD_16_28_scaled', 'MACD_32_96_scaled'
]

def compute_features(df, keep_unlabeled=False):
    """
    Computes momentum features for each ticker.
    df is expected to have ['Date', 'Ticker', 'Close', 'Open', 'High', 'Low', 'Volume']

    keep_unlabeled=False (training/backtest): drop every row with a NaN, including
        each ticker's last row, whose next-day return is not known yet.
    keep_unlabeled=True (live trading): keep rows whose features are complete even
        if Target_Return is NaN, so the most recent day can still be traded.
    """
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by=['Ticker', 'Date']).reset_index(drop=True)
    
    processed_dfs = []
    
    for ticker, group in df.groupby('Ticker'):
        group = group.copy()
        
        # Daily return
        group['Return_1d'] = group['Close'].pct_change()
        
        # Ex-ante volatility: EWMA of squared returns, then sqrt (Paper Section VI.A)
        # sigma^2_t = lambda * sigma^2_{t-1} + (1-lambda) * r^2_{t-1}, lambda = 1 - 1/60
        lam = 1.0 - 1.0 / 60.0
        sq_rets = group['Return_1d'] ** 2
        ewma_var = sq_rets.ewm(alpha=1 - lam, adjust=False).mean()
        group['Vol_60d'] = np.sqrt(ewma_var)
        
        # Forward fill and drop leading NaNs for Vol
        group['Vol_60d'] = group['Vol_60d'].ffill()
        
        # Multi-horizon returns (1, 21, 63, 126, 252 days)
        horizons = [1, 21, 63, 126, 252]
        for h in horizons:
            # We use log returns over horizon to avoid compounding issues, or simple returns
            group[f'Return_{h}d'] = group['Close'].pct_change(periods=h)
            
            # Volatility scaling
            # Paper uses returns normalized by ex-ante volatility
            group[f'Return_{h}d_scaled'] = group[f'Return_{h}d'] / (group['Vol_60d'] * np.sqrt(h) + 1e-8)
            
        # MACD indicators (8, 24), (16, 28), (32, 96)
        # Paper Eq.4: m̄_{t,s,l} = (v_{t,s} - v_{t,l}) / (sigma_t * sqrt(l))
        # where v_{t,s/l} is the EMA of Close, sigma_t is daily vol
        macd_pairs = [(8, 24), (16, 28), (32, 96)]
        for s, l in macd_pairs:
            group[f'MACD_{s}_{l}'] = compute_macd(group['Close'], s, l)
            # Correct normalisation: EMA difference is in price units, divide by (sigma * Close * sqrt(l))
            # to get a dimensionless signal (volatility-normalised price difference)
            group[f'MACD_{s}_{l}_scaled'] = group[f'MACD_{s}_{l}'] / (group['Close'] * group['Vol_60d'] * np.sqrt(l) + 1e-8)
            
        # Target: next day return
        # Paper uses continuous sizing z_t in (-1, 1). We train to maximize Sharpe.
        # But we need the next day's raw return to calculate the Sharpe loss.
        group['Target_Return'] = group['Return_1d'].shift(-1)
        
        processed_dfs.append(group)
        
    final_df = pd.concat(processed_dfs, ignore_index=True)
    # Drop rows where we don't have enough history for the 252d return or vol
    if keep_unlabeled:
        final_df = final_df.dropna(subset=FEATURE_COLS + ['Vol_60d']).reset_index(drop=True)
    else:
        final_df = final_df.dropna().reset_index(drop=True)
    return final_df

if __name__ == "__main__":
    if os.path.exists("data/raw_data.csv"):
        df = pd.read_csv("data/raw_data.csv")
        features_df = compute_features(df)
        os.makedirs("data", exist_ok=True)
        features_df.to_csv("data/features.csv", index=False)
        print(f"Features computed and saved to data/features.csv. Shape: {features_df.shape}")
        print(features_df.head())
    else:
        print("data/raw_data.csv not found. Run data.py first.")
