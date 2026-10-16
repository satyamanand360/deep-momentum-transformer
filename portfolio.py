"""
portfolio.py — Turning per-asset positions into portfolio returns and metrics.

Pure pandas/numpy (no torch), shared by train.py (validation), backtest.py and
live_test.py so all three compute P&L in exactly the same way.

Conventions
-----------
* A "record" is one (Date, Ticker) decision made at the close of Date, with
  z in (-1, 1) from the model and raw_return = the return from that close to
  the next close (Target_Return).
* Per-asset volatility targeting (paper Eq. 9): scaled_z = z * sigma_tgt / sigma_t.
* Transaction cost (paper Eq. 12): cost_bps * |scaled_z_t - scaled_z_{t-1}|,
  computed per ticker in date order.
* Portfolio: equal weight (1/N) across tickers available on each date.
* Portfolio leverage: by default *ex-ante* — uses only portfolio returns up to
  the previous day, so it is tradeable and has no look-ahead.
"""
import numpy as np
import pandas as pd

ANN = 252
ASSET_VOL_TARGET = 0.15      # per-asset annual vol target used in training (paper)
PORTFOLIO_VOL_TARGET = 0.30  # portfolio-level annual vol target for reporting


def scale_positions(z, vol, target_vol=ASSET_VOL_TARGET):
    """Per-asset volatility targeting: z * (daily target vol / daily ex-ante vol)."""
    return z * ((target_vol / np.sqrt(ANN)) / (vol + 1e-8))


def asset_returns(records, cost_bps=3.0, target_vol=ASSET_VOL_TARGET, position_col='z'):
    """
    records: DataFrame with Date, Ticker, raw_return, vol and a position column.
    Returns a copy sorted by (Ticker, Date) with scaled_z, turnover, gross, net.
    The first observed day of each ticker pays the cost of opening its position.
    """
    df = records.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(['Ticker', 'Date']).reset_index(drop=True)
    df['scaled_z'] = scale_positions(df[position_col].values, df['vol'].values, target_vol)
    prev = df.groupby('Ticker')['scaled_z'].shift(1).fillna(0.0)
    df['turnover'] = (df['scaled_z'] - prev).abs()
    df['gross'] = df['scaled_z'] * df['raw_return']
    df['net'] = df['gross'] - df['turnover'] * (cost_bps / 1e4)
    return df


def equal_weight_portfolio(asset_df, col='net'):
    """1/N portfolio: average across the tickers present on each date."""
    return asset_df.groupby('Date')[col].mean().sort_index()


def ex_ante_leverage(portfolio_returns, target_vol=PORTFOLIO_VOL_TARGET,
                     span=60, min_periods=20, max_leverage=None):
    """
    Leverage for day t from an EWMA estimate of portfolio volatility using
    returns up to day t-1 only (shift(1)), so it is known before trading.
    Days without enough history get NaN and are dropped by apply_leverage.
    """
    vol = portfolio_returns.ewm(span=span, min_periods=min_periods).std().shift(1) * np.sqrt(ANN)
    lev = target_vol / (vol + 1e-8)
    if max_leverage is not None:
        lev = lev.clip(upper=max_leverage)
    return lev


def apply_leverage(portfolio_returns, mode='ex_ante', target_vol=PORTFOLIO_VOL_TARGET, **kw):
    """
    mode='ex_ante'  : tradeable, uses only past returns (default).
    mode='constant' : one scale factor from the realised vol of the whole sample.
                      Uses future information; only for presentation (rescaling
                      a finished track record, as the paper does for its plots).
                      Sharpe and Sortino are identical to the unlevered series.
    mode='none'     : no portfolio leverage.
    Returns (levered_returns, leverage) with warm-up days dropped.
    """
    if mode == 'none':
        return portfolio_returns, pd.Series(1.0, index=portfolio_returns.index)
    if mode == 'constant':
        lev_value = target_vol / (portfolio_returns.std() * np.sqrt(ANN) + 1e-8)
        lev = pd.Series(lev_value, index=portfolio_returns.index)
        return portfolio_returns * lev, lev
    if mode == 'ex_ante':
        lev = ex_ante_leverage(portfolio_returns, target_vol, **kw)
        out = (portfolio_returns * lev).dropna()
        return out, lev.loc[out.index]
    raise ValueError(f"unknown leverage mode: {mode}")


def sharpe(returns):
    """Annualised Sharpe ratio of daily returns (rf = 0)."""
    returns = pd.Series(returns).dropna()
    return float(returns.mean() / (returns.std() + 1e-12) * np.sqrt(ANN))


def calculate_metrics(returns, risk_free_rate=0.0):
    """Standard metrics on daily returns (arithmetic annualisation, rf = 0 by default)."""
    returns = pd.Series(returns).dropna()
    ann_return = returns.mean() * ANN
    ann_volatility = returns.std() * np.sqrt(ANN)
    excess = returns - risk_free_rate / ANN
    sharpe_ratio = excess.mean() / (returns.std() + 1e-8) * np.sqrt(ANN)
    # Downside deviation over all days (textbook Sortino denominator)
    downside_dev = np.sqrt(np.mean(np.minimum(returns.values, 0.0) ** 2)) * np.sqrt(ANN)
    sortino = excess.mean() * ANN / (downside_dev + 1e-8)
    cum = (1 + returns).cumprod()
    max_drawdown = ((cum - cum.cummax()) / cum.cummax()).min()
    n_years = len(returns) / ANN
    cagr = cum.iloc[-1] ** (1 / n_years) - 1 if n_years > 0 and cum.iloc[-1] > 0 else np.nan
    return {
        'Annualized Return': ann_return,
        'CAGR': cagr,
        'Annualized Volatility': ann_volatility,
        'Sharpe Ratio': sharpe_ratio,
        'Sortino Ratio': sortino,
        'Max Drawdown': max_drawdown,
        'Calmar Ratio': ann_return / (abs(max_drawdown) + 1e-8),
        'Win Rate': (returns > 0).mean(),
        'Days': len(returns),
    }


def sharpe_standard_error(sharpe_ratio, n_days):
    """Approximate standard error of an annualised Sharpe ratio (Lo, 2002, iid returns)."""
    years = n_days / ANN
    return float(np.sqrt((1 + 0.5 * sharpe_ratio ** 2) / years)) if years > 0 else np.nan


def tsmom_positions(trailing_return):
    """Classic time-series momentum (Moskowitz et al. 2012): long if the 12-month return is positive."""
    return np.sign(trailing_return)
