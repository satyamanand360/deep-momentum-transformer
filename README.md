# Deep Momentum Transformer 📈

An industry-grade, mathematically rigorous implementation of a deep learning quantitative trading strategy based on the paper *Trading with the Momentum Transformer* (Wood et al., 2021). 

This repository implements a **Decoder-Only Temporal Fusion Transformer (TFT)** tailored specifically for momentum trading across a universe of 70 mega-cap equities and macro ETFs.

## 🌟 Key Features

* **Zero Look-Ahead Bias:** Fully causal mathematical pipeline (strictly lower-triangular causal masks, proper `shift(-1)` target alignment, and rolling EWMA normalization).
* **Interpretable Multi-Head Attention (IMHA):** Custom PyTorch attention mechanism sharing Value weights across heads to extract feature importance.
* **Variable Selection Networks (VSN) & Gated Residual Networks (GRN):** Dynamic feature weighting and nonlinear processing.
* **Custom Sharpe Loss:** Directly optimizes the empirical Sharpe Ratio (`ddof=0`) with a built-in transaction cost penalty (3bps) to suppress high-turnover noise.
* **Volatility Targeting:** Employs post-hoc leverage targeting a 30% annualized volatility, a standard industry risk-budgeting technique.
* **Live Inference Engine:** Seamlessly transitions from historical backtesting to live daily trading signals.
* **Streamlit Dashboard:** A beautiful, interactive dashboard for monitoring strategy performance, drawdowns, rolling metrics, and today's target positions.

## 📂 Architecture

- `data.py`: Downloads historical OHLCV data from Yahoo Finance.
- `features.py`: Computes ex-ante volatility, multi-horizon returns, and MACD indicators.
- `dataset.py`: Slices data into non-overlapping sliding windows for PyTorch DataLoaders.
- `model.py`: The core PyTorch implementation of the Momentum Transformer architecture.
- `train.py`: The training loop utilizing the custom negative Sharpe loss function over the $\Omega$ domain.
- `backtest.py`: Out-of-sample historical evaluation and metric computation.
- `live_data.py` & `live_test.py`: Operational scripts for downloading today's data and generating live portfolio positions.
- `dashboard.py`: Interactive Streamlit UI for monitoring live and historical performance.

## 🚀 Getting Started

### 1. Install Dependencies
```bash
pip install torch pandas numpy yfinance tqdm scipy streamlit plotly
```

### 2. Historical Training & Backtesting
```bash
python data.py
python features.py
python train.py      # Trains the model and saves to models/best_model.pth
python backtest.py   # Runs out-of-sample evaluation on 2020-2024 data
```

### 3. Live Inference
To generate today's portfolio weights:
```bash
python live_data.py
python live_test.py
```

### 4. Launch the Dashboard
Monitor the strategy's performance locally:
```bash
python -m streamlit run dashboard.py
```

## ⚠️ Disclaimer
This codebase is an exact implementation of an academic paper meant for educational and research purposes. The historical backtest utilizes a hardcoded universe of current mega-cap stocks, which introduces survivorship bias into the historical metrics. This bias does not affect live forward trading. Trade at your own risk.
