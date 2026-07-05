"""
Overfitting diagnostic.

Loads already-trained checkpoints and evaluates each on TRAIN, VAL, and TEST
splits using the same metrics (RMSE, NASA score). The gap between train and
val/test performance is the actual overfitting signal -- not the shape of
the loss curve alone.

Two mitigations are already in place for every model in this project:
  1. Dropout (0.2) in every LSTM/Transformer
  2. Checkpointing by BEST VALIDATION loss, not final-epoch weights

This script checks whether those mitigations actually worked, rather than
assuming they did.

Run from the model/ directory:
    python check_overfitting.py
"""

import os
import sys

import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "data"))
from loader import load_subset
from labeling import add_piecewise_rul, add_test_rul, normalize_sensors, make_windows, make_test_windows_last_only

from scoring import evaluate as nasa_evaluate
from lstm_baseline import LSTMRegressor, split_by_unit, WINDOW_SIZE, VAL_FRACTION, SEED

DEVICE = torch.device("cpu")


def prepare_data():
    train_df, test_df, true_rul = load_subset("FD001")
    train_df = add_piecewise_rul(train_df)
    test_df = add_test_rul(test_df, true_rul)

    feature_cols = [c for c in train_df.columns if c.startswith("sensor_")]
    stds = train_df[feature_cols].std()
    feature_cols = [c for c in feature_cols if stds[c] > 1e-4]

    train_df, test_df, _ = normalize_sensors(train_df, test_df, sensor_cols=feature_cols)

    X, y, meta = make_windows(train_df, feature_cols, window_size=WINDOW_SIZE)
    units_array = np.array([m[0] for m in meta])
    val_mask = split_by_unit(units_array, val_fraction=VAL_FRACTION, seed=SEED)

    X_train, y_train = X[~val_mask], y[~val_mask]
    X_val, y_val = X[val_mask], y[val_mask]

    X_test, y_test, _ = make_test_windows_last_only(test_df, feature_cols, window_size=WINDOW_SIZE)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test), feature_cols


def check_lstm_baseline(feature_cols, splits):
    checkpoint_path = os.path.join(os.path.dirname(__file__), "lstm_baseline.pt")
    if not os.path.exists(checkpoint_path):
        print("lstm_baseline.pt not found, skipping.")
        return

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model = LSTMRegressor(input_size=len(feature_cols))
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    print("\n" + "=" * 60)
    print("LSTM BASELINE (MSE-trained)")
    print("=" * 60)
    for split_name, (X, y) in splits.items():
        with torch.no_grad():
            preds = model(torch.from_numpy(X)).numpy()
        m = nasa_evaluate(y, preds)
        avg_score = m['nasa_score'] / len(y)
        print(f"  {split_name:6s} | RMSE: {m['rmse']:6.2f} | NASA score (avg/sample): {avg_score:6.2f} | n={len(y)}")


def check_asymmetric(feature_cols, splits):
    checkpoint_path = os.path.join(os.path.dirname(__file__), "lstm_asymmetric.pt")
    if not os.path.exists(checkpoint_path):
        print("lstm_asymmetric.pt not found, skipping (train it first).")
        return

    from lstm_asymmetric import AsymmetricLSTM

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model = AsymmetricLSTM(input_size=len(feature_cols))
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    print("\n" + "=" * 60)
    print("ASYMMETRIC LSTM (NASA-loss center head)")
    print("=" * 60)
    for split_name, (X, y) in splits.items():
        with torch.no_grad():
            _, center, _ = model(torch.from_numpy(X))
            preds = center.numpy()
        m = nasa_evaluate(y, preds)
        avg_score = m['nasa_score'] / len(y)
        print(f"  {split_name:6s} | RMSE: {m['rmse']:6.2f} | NASA score (avg/sample): {avg_score:6.2f} | n={len(y)}")


def main():
    print("Preparing data (same split as training)...")
    train_split, val_split, test_split, feature_cols = prepare_data()

    splits = {
        "TRAIN": train_split,
        "VAL": val_split,
        "TEST": test_split,
    }

    check_lstm_baseline(feature_cols, splits)
    check_asymmetric(feature_cols, splits)

    print("\n" + "=" * 60)
    print("HOW TO READ THIS")
    print("=" * 60)
    print("Compare TRAIN vs VAL/TEST for each model.")
    print("Healthy: train RMSE somewhat better than val/test, but not wildly so")
    print("  (e.g. train RMSE ~10-11 vs val/test RMSE ~12-14 -- normal generalization gap)")
    print("Overfitting red flag: train RMSE dramatically lower than val/test")
    print("  (e.g. train RMSE ~4 vs val/test RMSE ~13 -- model memorized training engines)")
    print("Note: VAL uses full sliding windows (many per engine), TEST uses only the")
    print("LAST window per engine -- not perfectly comparable, but TRAIN vs VAL is the")
    print("cleaner overfitting signal.")


if __name__ == "__main__":
    main()
