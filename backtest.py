import json
import os

import numpy as np
import pandas as pd
import torch

from dataset import get_dataloaders
from model import MomentumTransformer
from train import predict_last_step
from portfolio import (asset_returns, equal_weight_portfolio, apply_leverage,
                       calculate_metrics, sharpe, sharpe_standard_error,
                       tsmom_positions, PORTFOLIO_VOL_TARGET)


def calculate_drawdown(returns):
    cum_returns = (1 + returns).cumprod()
    running_max = cum_returns.cummax()
    return (cum_returns - running_max) / running_max


def load_model(model_path, device):
    """Load trained weights. Any missing or unexpected key is an error: a silently
    re-initialised layer would make the backtest meaningless."""
    model = MomentumTransformer(num_features=8, d_model=64, n_heads=4)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"{model_path} not found. Run train.py first.")
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    return model.to(device).eval()


def run_backtest(df_path="data/features.csv", model_path="models/best_model.pth",
                 cost_bps=3, leverage_mode="ex_ante", target_vol=PORTFOLIO_VOL_TARGET,
                 results_dir="results"):
    """
    Out-of-sample backtest on the test split (2020-01-01 onward).

    leverage_mode:
      'ex_ante'  (default) portfolio scaled to `target_vol` using only past returns
                 (60-day EWMA vol, lagged one day). Tradeable, no look-ahead.
      'constant' one scale factor from the whole test period's realised vol.
                 Presentation only: it uses future information. Sharpe/Sortino are
                 unaffected; return, vol and drawdown are "at target vol".
      'none'     raw 1/N portfolio of vol-targeted assets.
    The same leverage method is applied to the TSMOM baseline. The long-only
    benchmark is left unlevered and is compared on Sharpe.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    df = pd.read_csv(df_path)
    df['Date'] = pd.to_datetime(df['Date'])

    _, _, test_loader = get_dataloaders(df)
    model = load_model(model_path, device)

    # 1) Model positions: last step of every test window -> one (Date, Ticker) decision
    preds = predict_last_step(model, test_loader, device)
    preds['Date'] = pd.to_datetime(preds['Date'])

    # 2) Per-asset returns with exact per-ticker transaction costs
    strat_assets = asset_returns(preds, cost_bps=cost_bps)
    strat_raw = equal_weight_portfolio(strat_assets)

    # 3) TSMOM baseline on the same dates/tickers: sign of trailing 12-month return
    base = preds[['Date', 'Ticker', 'raw_return', 'vol']].merge(
        df[['Date', 'Ticker', 'Return_252d']], on=['Date', 'Ticker'], how='left')
    base['z'] = tsmom_positions(base['Return_252d'].values)
    tsmom_assets = asset_returns(base, cost_bps=cost_bps)
    tsmom_raw = equal_weight_portfolio(tsmom_assets)

    # 4) Equal-weight long-only benchmark (no costs, unlevered)
    benchmark_returns = preds.groupby('Date')['raw_return'].mean().sort_index()

    # 5) Portfolio leverage
    portfolio_returns, leverage = apply_leverage(strat_raw, mode=leverage_mode, target_vol=target_vol)
    tsmom_returns, _ = apply_leverage(tsmom_raw, mode=leverage_mode, target_vol=target_vol)
    common = portfolio_returns.index.intersection(tsmom_returns.index)
    portfolio_returns, tsmom_returns = portfolio_returns.loc[common], tsmom_returns.loc[common]
    benchmark_returns = benchmark_returns.loc[common]

    metrics = calculate_metrics(portfolio_returns)
    tsmom_metrics = calculate_metrics(tsmom_returns)
    bench_metrics = calculate_metrics(benchmark_returns)

    active = portfolio_returns - benchmark_returns
    ir = active.mean() * 252 / (active.std() * np.sqrt(252) + 1e-8)
    sharpe_se = sharpe_standard_error(metrics['Sharpe Ratio'], metrics['Days'])

    # Cost sensitivity: same positions, different cost assumptions (Sharpe only)
    cost_sens = {c: sharpe(equal_weight_portfolio(asset_returns(preds, cost_bps=c)))
                 for c in [0, 1, 3, 5, 10]}

    start, end = common.min().date(), common.max().date()
    print(f"Out-of-sample period: {start} to {end} ({len(common)} trading days)")
    lev_note = "none" if leverage_mode == "none" else \
        f"{leverage_mode} to {target_vol:.0%} vol (mean leverage {leverage.mean():.2f}x)"
    print(f"Costs: {cost_bps} bps per unit turnover | portfolio leverage: {lev_note}\n")
    table = pd.DataFrame({'Momentum Transformer': metrics, 'TSMOM baseline': tsmom_metrics,
                          'Long-only benchmark': bench_metrics})
    print(table.round(4).to_string())
    print(f"\nInformation ratio vs benchmark: {ir:.4f}")
    print(f"Sharpe standard error (Lo 2002): {sharpe_se:.2f}")
    print("Sharpe by cost assumption: " + ", ".join(f"{c} bps: {s:.2f}" for c, s in cost_sens.items()))

    os.makedirs(results_dir, exist_ok=True)
    portfolio_returns.to_csv(f"{results_dir}/portfolio_returns.csv")
    benchmark_returns.to_csv(f"{results_dir}/benchmark_returns.csv")
    tsmom_returns.to_csv(f"{results_dir}/tsmom_returns.csv")
    summary = {
        'period': [str(start), str(end)], 'trading_days': int(len(common)),
        'cost_bps': cost_bps, 'leverage_mode': leverage_mode, 'target_vol': target_vol,
        'strategy': metrics, 'tsmom': tsmom_metrics, 'benchmark': bench_metrics,
        'information_ratio': ir, 'sharpe_standard_error': sharpe_se,
        'sharpe_by_cost_bps': cost_sens,
    }
    with open(f"{results_dir}/metrics.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    write_markdown_summary(table, summary, f"{results_dir}/metrics.md")
    print(f"\nSaved {results_dir}/metrics.md (commit this file to show results on GitHub)")

    return metrics, bench_metrics, portfolio_returns, benchmark_returns


def write_markdown_summary(table, summary, path):
    """A small results table that can be committed to the repo (see .gitignore)."""
    pct_rows = {'Annualized Return', 'CAGR', 'Annualized Volatility', 'Max Drawdown', 'Win Rate'}
    lines = [
        "# Backtest results",
        "",
        f"Out-of-sample: {summary['period'][0]} to {summary['period'][1]} "
        f"({summary['trading_days']} trading days), {summary['cost_bps']} bps costs, "
        f"portfolio leverage `{summary['leverage_mode']}` to {summary['target_vol']:.0%} vol "
        "(benchmark unlevered).",
        "",
        "| Metric | " + " | ".join(table.columns) + " |",
        "|---|" + "---:|" * len(table.columns),
    ]
    for metric, row in table.iterrows():
        cells = []
        for v in row.values:
            if metric == 'Days':
                cells.append(f"{int(v)}")
            elif metric in pct_rows:
                cells.append(f"{v:.2%}")
            else:
                cells.append(f"{v:.2f}")
        lines.append(f"| {metric} | " + " | ".join(cells) + " |")
    lines += [
        "",
        f"Information ratio vs benchmark: {summary['information_ratio']:.2f}. "
        f"Approximate Sharpe standard error: {summary['sharpe_standard_error']:.2f}.",
        "",
        "Sharpe of the unlevered strategy by cost assumption: " +
        ", ".join(f"{c} bps = {s:.2f}" for c, s in summary['sharpe_by_cost_bps'].items()) + ".",
        "",
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Out-of-sample backtest")
    parser.add_argument("--leverage", choices=["ex_ante", "constant", "none"], default="ex_ante")
    parser.add_argument("--cost-bps", type=float, default=3)
    parser.add_argument("--model-path", default="models/best_model.pth")
    args = parser.parse_args()
    run_backtest(model_path=args.model_path, cost_bps=args.cost_bps, leverage_mode=args.leverage)
