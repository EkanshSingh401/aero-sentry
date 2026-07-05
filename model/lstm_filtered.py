"""
Trains the LSTM on EMA-FILTERED sensor data (same alpha=0.3, same formula
as pipeline/cpp_signal/signal_filter.cpp -- cross-validated to produce
byte-identical output). This directly fixes the train/serve distribution
mismatch found when the live consumer was feeding a raw-trained model
filtered inputs: normalizing and filtering are both linear and provably
commute, so there was never an "order of operations" bug to fix by
rearranging code. The only correct fix is training (or at least
evaluating) on the actual distribution the model will see in production,
which is what this script does.

Directly comparable to lstm_baseline.py's results: same architecture, same
hyperparameters, same seed, only the input representation differs.

Run from the model/ directory:
    python lstm_filtered.py
"""

import os
import sys
import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "data"))
from loader import load_subset
from labeling import (
    add_piecewise_rul, add_test_rul, normalize_sensors,
    make_windows, make_test_windows_last_only, apply_ema_filter,
)

from scoring import evaluate as nasa_evaluate
from lstm_baseline import (
    RULDataset, LSTMRegressor, split_by_unit, set_all_seeds,
    WINDOW_SIZE, BATCH_SIZE, HIDDEN_SIZE, NUM_LAYERS, LEARNING_RATE,
    NUM_EPOCHS, VAL_FRACTION, SEED,
)

EMA_ALPHA = 0.3  # must match pipeline/cpp_signal/signal_filter.cpp exactly
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    set_all_seeds(SEED)
    print(f"Device: {DEVICE}")
    print(f"EMA alpha: {EMA_ALPHA} (must match signal_filter.cpp)")

    train_df, test_df, true_rul = load_subset("FD001")
    train_df = add_piecewise_rul(train_df)
    test_df = add_test_rul(test_df, true_rul)

    all_sensor_cols = [c for c in train_df.columns if c.startswith("sensor_")]

    # Apply EMA filtering BEFORE computing near-constant-sensor exclusion
    # and BEFORE normalization -- the model must see the exact same
    # transformation pipeline in training as it will in production.
    print("Applying EMA filter to train and test sensor data...")
    train_df = apply_ema_filter(train_df, all_sensor_cols, alpha=EMA_ALPHA)
    test_df = apply_ema_filter(test_df, all_sensor_cols, alpha=EMA_ALPHA)

    stds = train_df[all_sensor_cols].std()
    feature_cols = [c for c in all_sensor_cols if stds[c] > 1e-4]
    dropped = set(all_sensor_cols) - set(feature_cols)
    print(f"Dropping near-constant sensors (post-filtering): {sorted(dropped)}")

    train_df, test_df, norm_stats = normalize_sensors(train_df, test_df, sensor_cols=feature_cols)

    # Save separately from norm_stats.json -- these stats are only valid
    # for filtered inputs, using them on raw values would be wrong.
    stats_path = os.path.join(os.path.dirname(__file__), "norm_stats_filtered.json")
    with open(stats_path, "w") as f:
        json.dump({k: list(v) for k, v in norm_stats.items()}, f, indent=2)
    print(f"Saved filtered-data normalization stats to {stats_path}")

    X, y, meta = make_windows(train_df, feature_cols, window_size=WINDOW_SIZE)
    units_array = np.array([m[0] for m in meta])

    val_mask = split_by_unit(units_array, val_fraction=VAL_FRACTION, seed=SEED)
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
    checkpoint_path = os.path.join(os.path.dirname(__file__), "lstm_filtered.pt")

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

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "model_state": model.state_dict(),
                "feature_cols": feature_cols,
                "window_size": WINDOW_SIZE,
                "hidden_size": HIDDEN_SIZE,
                "num_layers": NUM_LAYERS,
                "ema_alpha": EMA_ALPHA,
            }, checkpoint_path)

    print(f"\nBest val loss: {best_val_loss:.3f}. Checkpoint saved to {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    X_test, y_test, test_units = make_test_windows_last_only(test_df, feature_cols, window_size=WINDOW_SIZE)
    X_test_t = torch.from_numpy(X_test).to(DEVICE)

    with torch.no_grad():
        preds = model(X_test_t).cpu().numpy()

    metrics = nasa_evaluate(y_test, preds)
    print("\n=== Official FD001 test set evaluation (EMA-FILTERED pipeline) ===")
    print(f"RMSE:       {metrics['rmse']:.2f}")
    print(f"NASA score: {metrics['nasa_score']:.2f}")
    print("Compare against RAW-trained baseline (single run): RMSE 12.90-14.27 range")
    print("Compare against RAW-trained baseline (multi-seed mean): RMSE 14.03 +/- 0.27")
    print("NOTE: this is a SINGLE run for the filtered model -- given the earlier")
    print("multi-seed lesson (single-run comparisons are unreliable), treat this as")
    print("a first data point, not a final verdict, unless run across multiple seeds.")


if __name__ == "__main__":
    main()