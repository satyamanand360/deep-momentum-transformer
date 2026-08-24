import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

class MomentumDataset(Dataset):
    def __init__(self, df, seq_length=252, step=1):
        self.seq_length = seq_length
        self.step = step
        self.features = [
            'Return_1d_scaled', 'Return_21d_scaled', 'Return_63d_scaled', 
            'Return_126d_scaled', 'Return_252d_scaled',
            'MACD_8_24_scaled', 'MACD_16_28_scaled', 'MACD_32_96_scaled'
        ]
        
        self.indices = []
        self.data_X = []
        self.data_y = []
        self.data_vol = []
        self.data_date = []
        self.data_ticker = []
        
        # Group by ticker to prevent sequence overlap between different assets
        for ticker, group in df.groupby('Ticker'):
            group = group.sort_values('Date').reset_index(drop=True)
            self.data_X.append(group[self.features].values.astype(np.float32))
            self.data_y.append(group['Target_Return'].values.astype(np.float32))
            self.data_vol.append(group['Vol_60d'].values.astype(np.float32))
            self.data_date.append(group['Date'].astype(str).values)
            self.data_ticker.append(ticker)
            
            ticker_idx = len(self.data_ticker) - 1
            for i in range(0, len(group) - seq_length + 1, self.step):
                self.indices.append((ticker_idx, i))
                
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        t_idx, start_idx = self.indices[idx]
        return {
            'X': torch.tensor(self.data_X[t_idx][start_idx : start_idx + self.seq_length]),
            'y': torch.tensor(self.data_y[t_idx][start_idx : start_idx + self.seq_length]),
            'vol': torch.tensor(self.data_vol[t_idx][start_idx : start_idx + self.seq_length]),
            'date': self.data_date[t_idx][start_idx + self.seq_length - 1],
            'ticker': self.data_ticker[t_idx]
        }

def get_dataloaders(df, seq_length=252, batch_size=64, train_end='2018-01-01', val_end='2020-01-01'):
    df['Date'] = pd.to_datetime(df['Date'])
    
    train_df = df[df['Date'] < train_end]
    val_df = df[(df['Date'] >= train_end) & (df['Date'] < val_end)]
    test_df = df[df['Date'] >= val_end]
    
    train_dataset = MomentumDataset(train_df, seq_length, step=seq_length)
    val_dataset = MomentumDataset(val_df, seq_length, step=1)
    test_dataset = MomentumDataset(test_df, seq_length, step=1)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # Val and Test shouldn't be shuffled for evaluation ordering if needed, but for Sharpe it doesn't matter much.
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    import os
    if os.path.exists("data/features.csv"):
        df = pd.read_csv("data/features.csv")
        train_loader, val_loader, test_loader = get_dataloaders(df)
        batch = next(iter(train_loader))
        print(f"Batch X shape: {batch['X'].shape}")
        print(f"Batch y shape: {batch['y'].shape}")
