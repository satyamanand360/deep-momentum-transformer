import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

from features import FEATURE_COLS


class MomentumDataset(Dataset):
    """
    252-day windows of one ticker's features.

    min_end_date: only keep windows whose LAST day is on or after this date.
        This lets validation/test windows use earlier rows (already-known
        prices) as look-back context, so predictions start on the first day of
        the split instead of 252 trading days later.
    """
    def __init__(self, df, seq_length=252, step=1, min_end_date=None):
        self.seq_length = seq_length
        self.step = step
        self.features = list(FEATURE_COLS)

        self.indices = []
        self.data_X = []
        self.data_y = []
        self.data_vol = []
        self.data_date = []
        self.data_ticker = []

        min_end = pd.Timestamp(min_end_date) if min_end_date is not None else None

        # Group by ticker to prevent sequence overlap between different assets
        for ticker, group in df.groupby('Ticker'):
            group = group.sort_values('Date').reset_index(drop=True)
            n = len(group)
            if n < seq_length:
                continue
            dates = pd.to_datetime(group['Date'])
            first_start = 0
            if min_end is not None:
                # first row on/after min_end, minus the look-back
                first_end = int((dates < min_end).sum())
                if first_end >= n:
                    continue
                first_start = max(0, first_end - (seq_length - 1))

            self.data_X.append(group[self.features].values.astype(np.float32))
            self.data_y.append(group['Target_Return'].values.astype(np.float32))
            self.data_vol.append(group['Vol_60d'].values.astype(np.float32))
            self.data_date.append(dates.dt.strftime('%Y-%m-%d').values)
            self.data_ticker.append(ticker)

            ticker_idx = len(self.data_ticker) - 1
            for i in range(first_start, n - seq_length + 1, self.step):
                self.indices.append((ticker_idx, i))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        t_idx, start_idx = self.indices[idx]
        end = start_idx + self.seq_length
        return {
            'X': torch.tensor(self.data_X[t_idx][start_idx:end]),
            'y': torch.tensor(self.data_y[t_idx][start_idx:end]),
            'vol': torch.tensor(self.data_vol[t_idx][start_idx:end]),
            'date': self.data_date[t_idx][end - 1],
            'ticker': self.data_ticker[t_idx]
        }


def split_frames(df, train_end='2018-01-01', val_end='2020-01-01'):
    """
    Time-based split. Validation and test frames include the earlier rows as
    look-back context only; which dates are actually scored is controlled by
    MomentumDataset(min_end_date=...). Features are backward-looking, so using
    already-known earlier prices as context is not leakage.
    """
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    train_df = df[df['Date'] < train_end]
    val_ctx_df = df[df['Date'] < val_end]   # train-period rows used only as context
    test_ctx_df = df                        # val-period rows used only as context
    return train_df, val_ctx_df, test_ctx_df


def get_dataloaders(df, seq_length=252, batch_size=64, train_end='2018-01-01',
                    val_end='2020-01-01', eval_batch_size=256):
    train_df, val_ctx_df, test_ctx_df = split_frames(df, train_end, val_end)

    # Training: non-overlapping windows (paper: start-time step 252 for the TFT)
    train_dataset = MomentumDataset(train_df, seq_length, step=seq_length)
    # Validation / test: one window ending on every trading day of the split
    val_dataset = MomentumDataset(val_ctx_df, seq_length, step=1, min_end_date=train_end)
    test_dataset = MomentumDataset(test_ctx_df, seq_length, step=1, min_end_date=val_end)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=eval_batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=eval_batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    import os
    if os.path.exists("data/features.csv"):
        df = pd.read_csv("data/features.csv")
        train_loader, val_loader, test_loader = get_dataloaders(df)
        batch = next(iter(train_loader))
        print(f"Batch X shape: {batch['X'].shape}")
        print(f"Batch y shape: {batch['y'].shape}")
        print(f"Train/val/test sequences: {len(train_loader.dataset)}, "
              f"{len(val_loader.dataset)}, {len(test_loader.dataset)}")
