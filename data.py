import yfinance as yf
import pandas as pd
import os

# 20 Macro/Sector ETFs + 50 Mega-Cap Stocks = 70 highly liquid assets.
# Note: the stock list is today's mega caps, so historical backtests on it carry
# survivorship bias (see README).
TICKERS = [
    # Macro / sector ETFs
    'SPY', 'QQQ', 'IWM', 'EFA', 'EEM',
    'TLT', 'IEF', 'SHY', 'LQD', 'HYG',
    'GLD', 'SLV', 'USO', 'UNG', 'DBA',
    'VNQ', 'XLF', 'XLK', 'XLV', 'XLE',
    # Mega-cap stocks
    'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'NVDA', 'TSLA', 'BRK-B', 'UNH', 'JNJ',
    'JPM', 'V', 'PG', 'MA', 'HD', 'CVX', 'MRK', 'ABBV', 'LLY', 'PEP',
    'KO', 'BAC', 'AVGO', 'TMO', 'COST', 'MCD', 'CSCO', 'WMT', 'CRM', 'DIS',
    'PFE', 'DHR', 'NKE', 'XOM', 'TXN', 'UNP', 'LIN', 'ADBE', 'VZ', 'PM',
    'NEE', 'RTX', 'QCOM', 'INTC', 'AMD', 'HON', 'UPS', 'AMGN', 'IBM', 'BA'
]


def download_data(tickers, start_date, end_date, output_path=None):
    """
    Downloads OHLCV data for a list of tickers and returns it in long format
    (one row per Date x Ticker).
    """
    print(f"Downloading data for {len(tickers)} tickers from {start_date} to {end_date}...")
    # auto_adjust=True: Close is adjusted for splits and dividends. Older yfinance
    # versions defaulted to False (raw Close + separate 'Adj Close'), which would
    # turn a 4-for-1 split into a fake -75% one-day return. Pass it explicitly.
    data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True, progress=False)

    # yfinance returns MultiIndex columns (field, ticker) for several tickers;
    # stack the ticker level into rows to get long format.
    if len(tickers) > 1:
        data = data.stack(level=1).rename_axis(['Date', 'Ticker']).reset_index()
    else:
        data = data.reset_index()
        data['Ticker'] = tickers[0]

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        data.to_csv(output_path, index=False)
        print(f"Data saved to {output_path}")

    return data


if __name__ == "__main__":
    df = download_data(TICKERS, "2005-01-01", "2024-01-01", "data/raw_data.csv")
    print(df.head())
