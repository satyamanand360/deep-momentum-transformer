import torch
import torch.optim as optim
import pandas as pd
import numpy as np
import random
import os
from dataset import get_dataloaders
from model import MomentumTransformer
from portfolio import asset_returns, equal_weight_portfolio, sharpe, ASSET_VOL_TARGET
from tqdm import tqdm

COST = 0.0003   # 3 bps per unit of turnover
SEED = 42


def set_seed(seed=SEED):
    """Make runs repeatable (up to nondeterministic GPU kernels)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sharpe_loss(returns, ann_factor=252):
    """
    Negative annualised Sharpe ratio, the paper's Eq. 11:
    -sqrt(252) * E[R] / std(R) over all asset-time pairs in the batch (Omega).
    Population std (ddof=0) matches the paper's Var[] definition.
    """
    mean_ret = torch.mean(returns)
    std_ret = torch.std(returns, unbiased=False)
    sharpe_ratio = (mean_ret * np.sqrt(ann_factor)) / (std_ret + 1e-8)
    return -sharpe_ratio


def strategy_returns(z, y, vol, cost=COST):
    """
    Per-asset, per-day strategy returns inside each sequence (paper Eq. 9 + 12):
    scaled position = z * sigma_tgt / sigma_t, return = position * next-day return
    minus cost * |change in position|. The first day of a window has no
    previous position inside the window, so no turnover is charged there.
    """
    tgt_vol_daily = ASSET_VOL_TARGET / np.sqrt(252)
    scaled_z = z * (tgt_vol_daily / (vol + 1e-8))
    prev_scaled_z = torch.cat([scaled_z[:, :1], scaled_z[:, :-1]], dim=1)
    turnover = torch.abs(scaled_z - prev_scaled_z)
    return scaled_z * y - cost * turnover


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    epoch_returns = []

    for batch in tqdm(loader, desc="Training"):
        X = batch['X'].to(device)      # (batch, seq, features)
        y = batch['y'].to(device)      # (batch, seq)
        vol = batch['vol'].to(device)  # (batch, seq)

        optimizer.zero_grad()
        z, _, _ = model(X)             # (batch, seq)

        # Flatten into Omega: all asset-time pairs in the batch (Eq. 11)
        omega_returns = strategy_returns(z, y, vol).reshape(-1)
        loss = sharpe_loss(omega_returns)
        loss.backward()

        # Clip gradients (max grad norm 100 is in the paper's TFT search grid)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=100.0)
        optimizer.step()

        total_loss += loss.item()
        epoch_returns.append(omega_returns.detach())

    epoch_returns = torch.cat(epoch_returns)
    epoch_sharpe = -sharpe_loss(epoch_returns).item()
    return total_loss / len(loader), epoch_sharpe


@torch.no_grad()
def predict_last_step(model, loader, device):
    """
    Run every window and keep only its last day: one decision per (Date, Ticker),
    made with the full 252-day look-back, exactly as in the backtest.
    """
    model.eval()
    rows = []
    for batch in loader:
        z, _, _ = model(batch['X'].to(device))
        rows.append(pd.DataFrame({
            'Date': list(batch['date']),
            'Ticker': list(batch['ticker']),
            'z': z[:, -1].cpu().numpy(),
            'raw_return': batch['y'][:, -1].numpy(),
            'vol': batch['vol'][:, -1].numpy(),
        }))
    return pd.concat(rows, ignore_index=True)


def eval_epoch(model, loader, device, cost_bps=COST * 1e4):
    """
    Validation score = Sharpe of the tradeable strategy on the validation period:
    last-step positions, exact per-ticker costs, equal-weight portfolio.
    (Sharpe does not depend on portfolio leverage, so none is applied.)
    """
    preds = predict_last_step(model, loader, device)
    assets = asset_returns(preds, cost_bps=cost_bps)
    return sharpe(equal_weight_portfolio(assets))


def train_model(seed=SEED, features_path="data/features.csv", model_path="models/best_model.pth"):
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device} | seed: {seed}")

    df = pd.read_csv(features_path)
    # Paper uses batch size 64 for the TFT (from its grid 32/64/128)
    train_loader, val_loader, _ = get_dataloaders(df, batch_size=64)
    print(f"Train sequences: {len(train_loader.dataset)} | validation decisions: {len(val_loader.dataset)}")

    model = MomentumTransformer(num_features=8, d_model=64, n_heads=4).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    best_val_sharpe = -float('inf')
    patience = 25
    patience_counter = 0
    epochs = 300

    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)

    for epoch in range(epochs):
        train_loss, train_sharpe = train_epoch(model, train_loader, optimizer, device)
        val_sharpe = eval_epoch(model, val_loader, device)

        scheduler.step(val_sharpe)
        print(f"Epoch {epoch+1}/{epochs} | Loss: {train_loss:.4f} | Train Sharpe: {train_sharpe:.4f} | Val Sharpe: {val_sharpe:.4f}")

        if val_sharpe > best_val_sharpe:
            best_val_sharpe = val_sharpe
            patience_counter = 0
            torch.save(model.state_dict(), model_path)
            print("  --> Saved new best model!")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

    print(f"Training complete. Best validation Sharpe: {best_val_sharpe:.4f}")
    return best_val_sharpe


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train the Momentum Transformer")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--model-path", default="models/best_model.pth")
    args = parser.parse_args()
    train_model(seed=args.seed, model_path=args.model_path)
