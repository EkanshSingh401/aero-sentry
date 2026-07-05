"""
LSTM baseline for C-MAPSS RUL prediction.

Pipeline:
  load FD001 -> piecewise-linear RUL -> normalize sensors -> window (size 30)
  -> split by UNIT (not row) into train/val -> train 2-layer LSTM -> evaluate
  on official test set using NASA score + RMSE.

Run from the model/ directory:
    python lstm_baseline.py
"""

import os
import sys
import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import wandb

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "data"))
from loader import load_subset
from labeling import add_piecewise_rul, add_test_rul, normalize_sensors, make_windows, make_test_windows_last_only

from scoring import evaluate as nasa_evaluate

WINDOW_SIZE = 30
BATCH_SIZE = 64
HIDDEN_SIZE = 64
NUM_LAYERS = 2
LEARNING_RATE = 1e-3
NUM_EPOCHS = 40
VAL_FRACTION = 0.2
SEED = int(os.environ.get("AEROSENTRY_SEED", 42))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RULDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class LSTMRegressor(nn.Module):
    def __init__(self, input_size, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        # x: (batch, window, features)
        out, (h_n, c_n) = self.lstm(x)
        last_hidden = out[:, -1, :]  # (batch, hidden_size)
        return self.head(last_hidden).squeeze(-1)


def split_by_unit(units_array, val_fraction=VAL_FRACTION, seed=SEED):
    """Return a boolean mask selecting which rows belong to validation,
    splitting by unique unit id so no engine appears in both train and val."""
    rng = np.random.default_rng(seed)
    unique_units = np.unique(units_array)
    rng.shuffle(unique_units)
    n_val = max(1, int(len(unique_units) * val_fraction))
    val_units = set(unique_units[:n_val].tolist())
    mask = np.array([u in val_units for u in units_array])
    return mask


def set_all_seeds(seed=SEED):
    """
    Seeds every source of randomness that affects training results.

    Found via real variance between runs: only the train/val SPLIT was
    seeded (via np.random.default_rng in split_by_unit), but model weight
    initialization, dropout masks, and DataLoader shuffling all draw from
    PyTorch's global RNG, which was never seeded. Result: identical code,
    identical data, meaningfully different metrics every run (e.g. RMSE
    12.90 vs 14.76 on the same LSTM baseline). Fixing this is what makes
    reported numbers actually reproducible by someone else running this
    code, not just reproducible by luck.
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    set_all_seeds(SEED)
    wandb.init(
        project="aerosentry",
        name=f"lstm_baseline_seed{SEED}",
        config={
            "model": "LSTM",
            "seed": SEED,
            "window_size": WINDOW_SIZE,
            "batch_size": BATCH_SIZE,
            "hidden_size": HIDDEN_SIZE,
            "num_layers": NUM_LAYERS,
            "learning_rate": LEARNING_RATE,
            "num_epochs": NUM_EPOCHS,
            "loss_fn": "MSE",
        },
    )
    print(f"Device: {DEVICE}")

    train_df, test_df, true_rul = load_subset("FD001")
    train_df = add_piecewise_rul(train_df)
    test_df = add_test_rul(test_df, true_rul)

    feature_cols = [c for c in train_df.columns if c.startswith("sensor_")]
    # Drop near-constant sensors (near-zero variance in FD001 under one
    # operating condition) -- they add noise/dimensionality with no signal.
    stds = train_df[feature_cols].std()
    keep_cols = [c for c in feature_cols if stds[c] > 1e-4]
    dropped = set(feature_cols) - set(keep_cols)
    print(f"Dropping near-constant sensors: {sorted(dropped)}")
    feature_cols = keep_cols

    train_df, test_df, norm_stats = normalize_sensors(train_df, test_df, sensor_cols=feature_cols)

    # Save normalization stats -- the streaming pipeline and edge deployment
    # MUST use these exact same stats, or predictions will be garbage.
    stats_path = os.path.join(os.path.dirname(__file__), "norm_stats.json")
    with open(stats_path, "w") as f:
        json.dump({k: list(v) for k, v in norm_stats.items()}, f, indent=2)
    print(f"Saved normalization stats to {stats_path}")

    X, y, meta = make_windows(train_df, feature_cols, window_size=WINDOW_SIZE)
    units_array = np.array([m[0] for m in meta])

    val_mask = split_by_unit(units_array)
    X_train, y_train = X[~val_mask], y[~val_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    print(f"Train windows: {len(X_train)}, Val windows: {len(X_val)}")

    train_ds = RULDataset(X_train, y_train)
    val_ds = RULDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = LSTMRegressor(input_size=len(feature_cols)).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()

    best_val_loss = float("inf")
    checkpoint_path = os.path.join(os.path.dirname(__file__), "lstm_baseline.pt")

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                pred = model(xb)
                val_losses.append(loss_fn(pred, yb).item())

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        print(f"Epoch {epoch:3d}/{NUM_EPOCHS} | train_loss={train_loss:.3f} | val_loss={val_loss:.3f}")
        wandb.log({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "model_state": model.state_dict(),
                "feature_cols": feature_cols,
                "window_size": WINDOW_SIZE,
                "hidden_size": HIDDEN_SIZE,
                "num_layers": NUM_LAYERS,
            }, checkpoint_path)

    print(f"\nBest val loss: {best_val_loss:.3f}. Checkpoint saved to {checkpoint_path}")

    # ---- Final evaluation on the OFFICIAL test set (last window per unit,
    # matching RUL_FD001.txt ground truth) ----
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    X_test, y_test, test_units = make_test_windows_last_only(test_df, feature_cols, window_size=WINDOW_SIZE)
    X_test_t = torch.from_numpy(X_test).to(DEVICE)

    with torch.no_grad():
        preds = model(X_test_t).cpu().numpy()

    metrics = nasa_evaluate(y_test, preds)
    print("\n=== Official FD001 test set evaluation ===")
    print(f"RMSE:       {metrics['rmse']:.2f}")
    print(f"NASA score: {metrics['nasa_score']:.2f}")
    print("Published FD001 LSTM baselines typically report RMSE ~12-15, "
          "NASA score ~200-400 -- use this as a sanity range, not a target to game.")

    wandb.log({
        "test_rmse": metrics["rmse"],
        "test_nasa_score": metrics["nasa_score"],
        "test_nasa_score_avg": metrics["nasa_score"] / len(y_test),
        "best_val_loss": best_val_loss,
    })
    wandb.finish()


if __name__ == "__main__":
    main()