import nbformat as nbf

nb = nbf.v4.new_notebook()

cells = [
    nbf.v4.new_markdown_cell("# Momentum Transformer Pipeline\nThis notebook executes the end-to-end pipeline described in the paper *Trading with the Momentum Transformer*."),
    
    nbf.v4.new_markdown_cell("## 1. Setup & Imports"),
    nbf.v4.new_code_cell("import pandas as pd\nimport torch\nimport numpy as np\nfrom data import download_data\nfrom features import compute_features\nfrom dataset import get_dataloaders\nfrom model import MomentumTransformer\nfrom train import train_model\nfrom backtest import run_backtest"),
    
    nbf.v4.new_markdown_cell("## 2. Data Download\nDownloading OHLCV data for a basket of liquid ETFs representing different asset classes."),
    nbf.v4.new_code_cell("tickers = ['SPY', 'QQQ', 'IWM', 'EFA', 'EEM', 'TLT', 'IEF', 'SHY', 'LQD', 'HYG', 'GLD', 'SLV', 'USO', 'UNG', 'DBA', 'VNQ', 'XLF', 'XLK', 'XLV', 'XLE', 'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'NVDA', 'TSLA', 'BRK-B', 'UNH', 'JNJ', 'JPM', 'V', 'PG', 'MA', 'HD', 'CVX', 'MRK', 'ABBV', 'LLY', 'PEP', 'KO', 'BAC', 'AVGO', 'TMO', 'COST', 'MCD', 'CSCO', 'WMT', 'CRM', 'DIS', 'PFE', 'DHR', 'NKE', 'XOM', 'TXN', 'UNP', 'LIN', 'ADBE', 'VZ', 'PM', 'NEE', 'RTX', 'QCOM', 'INTC', 'AMD', 'HON', 'UPS', 'AMGN', 'IBM', 'BA']\nstart_date = '2005-01-01'\nend_date = '2024-01-01'\ndf_raw = download_data(tickers, start_date, end_date, 'data/raw_data.csv')\ndf_raw.head()"),
    
    nbf.v4.new_markdown_cell("## 3. Feature Engineering\nComputing multi-horizon returns, volatility scaling, and MACD indicators."),
    nbf.v4.new_code_cell("df_features = compute_features(df_raw)\ndf_features.to_csv('data/features.csv', index=False)\ndf_features.tail()"),
    
    nbf.v4.new_markdown_cell("## 4. Leakage-safe Train/Val/Test Split & Dataloaders"),
    nbf.v4.new_code_cell("train_loader, val_loader, test_loader = get_dataloaders(df_features, batch_size=256)\nprint(f'Train batches: {len(train_loader)}')\nprint(f'Val batches: {len(val_loader)}')\nprint(f'Test batches: {len(test_loader)}')"),
    
    nbf.v4.new_markdown_cell("## 5 & 6. Model Architecture & Training Loop\nTraining the Decoder-Only TFT to maximize the Sharpe ratio."),
    nbf.v4.new_code_cell("# This will train the model and save the best weights to models/best_model.pth\n# Note: This might take some time depending on your hardware.\ntrain_model()"),
    
    nbf.v4.new_markdown_cell("## 7 & 8. Backtesting & Metrics Computation\nRunning the backtest with transaction costs and calculating all requested metrics."),
    nbf.v4.new_code_cell("metrics, bench_metrics, port_returns, bench_returns = run_backtest('data/features.csv', 'models/best_model.pth', cost_bps=3)"),
    
    nbf.v4.new_markdown_cell("## 9. Dashboard\nTo view the interactive dashboard with equity curves, drawdowns, and interpretability insights, run the following command in your terminal:\n`streamlit run dashboard.py`")
]

nb['cells'] = cells
with open('pipeline.ipynb', 'w') as f:
    nbf.write(nb, f)
print("pipeline.ipynb created successfully.")
