import yfinance as yf
import pandas as pd
import os

def download_data(tickers, start_date, end_date, output_path=None):
    """
    Downloads OHLCV data for a list of tickers.
    """
    print(f"Downloading data for {tickers} from {start_date} to {end_date}...")
    data = yf.download(tickers, start=start_date, end=end_date)
    
    # yfinance returns a MultiIndex column dataframe if multiple tickers are passed
    # We want to stack it to have 'Ticker' as a level or melt it
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
    # 20 diverse liquid ETFs across sectors, bonds, and commodities
    tickers = None
    if tickers is None:
        # 20 Macro ETFs + Top 50 Mega-Cap Stocks = 70 Highly Liquid Assets
        tickers = [
            # Original 20 Macro/Sector ETFs
            'SPY', 'QQQ', 'IWM', 'EFA', 'EEM', 
            'TLT', 'IEF', 'SHY', 'LQD', 'HYG', 
            'GLD', 'SLV', 'USO', 'UNG', 'DBA', 
            'VNQ', 'XLF', 'XLK', 'XLV', 'XLE',
            # Top 50 Mega-Cap Stocks
            'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'NVDA', 'TSLA', 'BRK-B', 'UNH', 'JNJ',
            'JPM', 'V', 'PG', 'MA', 'HD', 'CVX', 'MRK', 'ABBV', 'LLY', 'PEP',
            'KO', 'BAC', 'AVGO', 'TMO', 'COST', 'MCD', 'CSCO', 'WMT', 'CRM', 'DIS',
            'PFE', 'DHR', 'NKE', 'XOM', 'TXN', 'UNP', 'LIN', 'ADBE', 'VZ', 'PM',
            'NEE', 'RTX', 'QCOM', 'INTC', 'AMD', 'HON', 'UPS', 'AMGN', 'IBM', 'BA'
        ]
    df = download_data(tickers, "2005-01-01", "2024-01-01", "data/raw_data.csv")
    print(df.head())
