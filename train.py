import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import os
from dataset import get_dataloaders
from model import MomentumTransformer
from tqdm import tqdm

def sharpe_loss(returns, ann_factor=252):
    """
    Computes negative annualized Sharpe ratio as the loss.
    Matches the paper's definition: sqrt(252) * E[R] / std(R).
    Uses population std (ddof=0) to match the expectation formula exactly.
    returns: tensor of daily portfolio returns over Omega.
    """
    mean_ret = torch.mean(returns)
    # Population std (ddof=0) matches paper's Var[] definition
    std_ret = torch.std(returns, unbiased=False)
    
    # Add a small epsilon to avoid division by zero
    sharpe = (mean_ret * np.sqrt(ann_factor)) / (std_ret + 1e-8)
    return -sharpe

def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    epoch_returns = []
    
    # 3bps average transaction cost
    cost_bps = 0.0003
    tgt_vol_daily = 0.15 / np.sqrt(252)
    
    for batch in tqdm(loader, desc="Training"):
        X = batch['X'].to(device) # (batch, seq, features)
        y = batch['y'].to(device) # (batch, seq)
        vol = batch['vol'].to(device) # (batch, seq)
        
        optimizer.zero_grad()
        z, _, _ = model(X) # (batch, seq)
        
        scaled_z = z * (tgt_vol_daily / (vol + 1e-8))
        
        # Calculate turnover across the sequence
        prev_scaled_z = torch.cat([torch.zeros_like(scaled_z[:, :1]), scaled_z[:, :-1]], dim=1)
        turnover = torch.abs(scaled_z - prev_scaled_z)
        
        # Raw strategy returns minus transaction cost penalty (Eq 12)
        strat_returns = scaled_z * y - cost_bps * turnover
        
        # Flatten into Omega tensor containing all asset-time pairs (batch * seq)
        omega_returns = strat_returns.reshape(-1)
        
        # Compute Sharpe over entire Omega tensor (Eq 11)
        loss = sharpe_loss(omega_returns)
        loss.backward()
        
        # Clip gradients (Paper uses 100 for TFT)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=100.0)
        optimizer.step()
        
        total_loss += loss.item()
        epoch_returns.append(omega_returns.detach())
        
    epoch_returns = torch.cat(epoch_returns)
    epoch_sharpe = -sharpe_loss(epoch_returns).item()
    
    return total_loss / len(loader), epoch_sharpe

def eval_epoch(model, loader, device):
    model.eval()
    all_returns = []
    
    cost_bps = 0.0003
    tgt_vol_daily = 0.15 / np.sqrt(252)
    
    with torch.no_grad():
        for batch in loader:
            X = batch['X'].to(device)
            y = batch['y'].to(device)
            vol = batch['vol'].to(device)
            
            z, _, _ = model(X)
            
            scaled_z = z * (tgt_vol_daily / (vol + 1e-8))
            prev_scaled_z = torch.cat([torch.zeros_like(scaled_z[:, :1]), scaled_z[:, :-1]], dim=1)
            turnover = torch.abs(scaled_z - prev_scaled_z)
            
            strat_returns = scaled_z * y - cost_bps * turnover
            all_returns.append(strat_returns.reshape(-1))
            
    all_returns = torch.cat(all_returns)
    sharpe = -sharpe_loss(all_returns).item()
    return sharpe

def train_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    df = pd.read_csv("data/features.csv")
    # Paper uses batch size 64 for TFT
    train_loader, val_loader, test_loader = get_dataloaders(df, batch_size=64)
    
    model = MomentumTransformer(num_features=8, d_model=64, n_heads=4).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    best_val_sharpe = -float('inf')
    patience = 25
    patience_counter = 0
    epochs = 300
    
    os.makedirs("models", exist_ok=True)
    
    for epoch in range(epochs):
        train_loss, train_sharpe = train_epoch(model, train_loader, optimizer, device)
        val_sharpe = eval_epoch(model, val_loader, device)
        
        scheduler.step(val_sharpe)
        print(f"Epoch {epoch+1}/{epochs} | Loss: {train_loss:.4f} | Train Sharpe: {train_sharpe:.4f} | Val Sharpe: {val_sharpe:.4f}")
        
        if val_sharpe > best_val_sharpe:
            best_val_sharpe = val_sharpe
            patience_counter = 0
            torch.save(model.state_dict(), "models/best_model.pth")
            print("  --> Saved new best model!")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break
                
    print("Training complete.")
    
if __name__ == "__main__":
    train_model()
