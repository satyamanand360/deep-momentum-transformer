"""
live_data.py — Download and feature-engineer fresh data up to today.
Does NOT modify the historical data files. Writes to data/live_raw.csv and
data/live_features.csv. Run after the US market close so the latest bar is final.
"""
import os
from datetime import datetime, timedelta

from data import TICKERS, download_data
from features import compute_features

# A prediction needs 252 rows for the 252-day return feature plus a 252-day
# model window, i.e. about two trading years of history. Starting in 2022
# gives live predictions from early 2024 onward.
LIVE_START = "2022-01-01"


def download_live_data(start=LIVE_START):
    # yfinance's `end` is exclusive, so ask for tomorrow to include today's close
    end = (datetime.today() + timedelta(days=1)).strftime("%Y-%m-%d")
    raw = download_data(TICKERS, start, end, "data/live_raw.csv")
    print(f"Raw live data saved -> data/live_raw.csv  ({len(raw)} rows)")

    # Same feature code as training. keep_unlabeled=True keeps each ticker's
    # latest row (its next-day return is unknown) so today can be traded.
    features = compute_features(raw, keep_unlabeled=True)
    os.makedirs("data", exist_ok=True)
    features.to_csv("data/live_features.csv", index=False)
    print(f"Live features saved -> data/live_features.csv  ({len(features)} rows, "
          f"last date {features['Date'].max().date()})")
    return features


if __name__ == "__main__":
    download_live_data()
