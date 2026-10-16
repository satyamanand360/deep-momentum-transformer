# Deep Momentum Transformer 📈

A PyTorch re-implementation of the deep-learning trend-following strategy from *Trading with the Momentum Transformer* (Wood, Giegerich, Roberts & Zohren, 2021, [arXiv:2112.08534](https://arxiv.org/abs/2112.08534)), applied to a universe of 70 liquid US ETFs and large-cap stocks.

The model is a **decoder-only Temporal Fusion Transformer (TFT)**: it reads one year of volatility-normalised momentum features per asset and outputs a daily position in (−1, 1), trained end-to-end to maximise the Sharpe ratio net of transaction costs.

## 🌟 Key Features

* **Causal by construction:** features use only past prices, the target is the next-day return (`shift(-1)`), the LSTM is unidirectional, attention uses a lower-triangular mask, and data are split by date.
* **Interpretable Multi-Head Attention (IMHA):** per-head queries/keys with a value projection shared across heads, so averaged attention weights are interpretable.
* **Variable Selection Networks (VSN) & Gated Residual Networks (GRN):** per-day feature weights and gated non-linear processing.
* **Sharpe-ratio loss with costs:** directly maximises the Sharpe ratio over all asset-days in a batch (paper Eq. 11), with a 3 bps turnover penalty (Eq. 12) in the loss.
* **Tradeable portfolio construction:** per-asset volatility targeting (15%), equal-weight portfolio, and portfolio leverage to 30% volatility estimated **ex-ante** from past returns only.
* **Baselines:** classic TSMOM (sign of 12-month return, vol-scaled, same costs and leverage) and an equal-weight long-only benchmark.
* **Live inference & Streamlit dashboard:** positions for the next session from the latest close, a live track record since 2024, equity curve, drawdowns, rolling Sharpe and monthly returns.

## 📂 Architecture

- `data.py`: downloads split/dividend-adjusted daily prices from Yahoo Finance (`auto_adjust=True`).
- `features.py`: ex-ante EWMA volatility, vol-scaled returns over 1/21/63/126/252 days, three vol-normalised MACD signals, next-day target.
- `dataset.py`: 252-day windows per ticker; time split (train < 2018, validation 2018–2019, test ≥ 2020). Validation/test windows use earlier, already-known prices as look-back context, so the test trades every day from January 2020.
- `model.py`: VSN → LSTM → masked IMHA → feed-forward, each with a gated residual, and a tanh output head.
- `portfolio.py`: shared P&L logic: vol targeting, per-ticker costs, 1/N portfolio, ex-ante leverage, metrics, TSMOM.
- `train.py`: negative Sharpe loss, Adam, gradient clipping, LR halving on plateau, early stopping on the **validation Sharpe of the tradeable strategy** (last-step positions, exact per-ticker costs). Seeded.
- `backtest.py`: out-of-sample evaluation, baselines, cost sensitivity, Sharpe standard error; writes `results/metrics.md`.
- `live_data.py` & `live_test.py`: live features (same code as training) and positions/weights for the next session.
- `dashboard.py`: Streamlit UI for historical and live results.
- `create_*.py` + `notebook_tools.py`: build the single-file Colab / Kaggle notebooks from the modules.

## 🚀 Getting Started

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Historical Training & Backtesting
```bash
python data.py
python features.py
python train.py --seed 42     # saves models/best_model.pth
python backtest.py            # out-of-sample 2020–2023, writes results/metrics.md
```
`backtest.py --leverage constant` reproduces a presentation-only rescaling with the whole test period's volatility (same Sharpe, uses future information); `--leverage none` shows the raw 1/N portfolio.

### 3. Live Inference
Run after the US market close:
```bash
python live_data.py
python live_test.py
```

### 4. Launch the Dashboard
```bash
python -m streamlit run dashboard.py
```

## 📊 Results

After running `backtest.py`, the results table is in [`results/metrics.md`](results/metrics.md) (this file is tracked by git; the rest of `results/` is not). Compare **Sharpe ratios**: the strategy and TSMOM are levered to 30% volatility, the benchmark is not.

## 🔬 Differences from the paper

This is a faithful re-implementation of the architecture and training objective, not an exact replication of the paper's experiment:

| | Paper | This repo |
|---|---|---|
| Data | 50 continuous futures, 1990–2020 | 20 ETFs + 50 US stocks (Yahoo), 2005–2023 |
| Evaluation | Expanding window, retrain every 5 years, 5 repeats | One split (train < 2018, val 2018–19, test 2020–23), seeded runs |
| Hyperparameters | Random grid search | Fixed values from the paper's grid |
| Changepoint detection (CPD) features | Optional | Not implemented |
| Static covariates (asset class) | Yes | No |
| MACD normalisation | Rolling price std, then rolling signal std (Baz et al.) | Divided by price × σ × √(long span) |
| Transaction costs | Studied after training | 3 bps turnover penalty inside the loss |

## ⚠️ Limitations & Disclaimer

* **Survivorship bias:** the stock list is today's mega caps, chosen with hindsight; historical results are flattered, especially long positions. A rigorous study would use point-in-time constituents or futures.
* **Limited evidence:** about four years of out-of-sample data from one split; the standard error of an annualised Sharpe over 4 years is roughly 0.5–0.6 (printed by `backtest.py`). Repeat training with several seeds and report the spread.
* **Simplified costs:** a flat 3 bps per unit of turnover; no short-borrow or financing costs. Per-asset volatility targeting can lever low-volatility assets (e.g. SHY) heavily.
* **Correlated universe:** many stocks and equity ETFs share one risk factor, so 1/N weighting is less diversified than it looks.

Educational and research code. Not investment advice.
