"""
Quantile-regression LSTM for C-MAPSS RUL prediction.

Instead of predicting a single RUL number, this model predicts three
quantiles (p10, p50, p90) simultaneously, giving an uncertainty band around
the RUL estimate: "RUL is 47, with 80% confidence it's between 39 and 55."
This is a materially different and more useful output for a real
maintenance decision than a bare point estimate -- a single number implies
false precision that no sensor-based prediction actually has.

Trained with pinball loss (quantile loss) instead of MSE. Same data
pipeline, same architecture backbone, and same train/val split as
lstm_baseline.py, so the point-estimate (p50) quality is directly
comparable to the original LSTM baseline.

Run from the model/ directory:
    python lstm_quantile.py
"""

import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "data"))
from loader import load_subset
from labeling import add_piecewise_rul, add_test_rul, normalize_sensors, make_windows, make_test_windows_last_only

from scoring import evaluate as nasa_evaluate
from lstm_baseline import RULDataset, split_by_unit, set_all_seeds, WINDOW_SIZE, BATCH_SIZE, HIDDEN_SIZE, NUM_LAYERS, LEARNING_RATE, NUM_EPOCHS, VAL_FRACTION, SEED

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

QUANTILES = [0.1, 0.5, 0.9]


class QuantileLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, quantiles=QUANTILES):
        super().__init__()
        self.quantiles = quantiles
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0.0,
        )
        # One output head per quantile -- simplest correct way to guarantee
        # each quantile gets its own learned mapping from the shared LSTM
        # representation.
        self.shared = nn.Sequential(nn.Linear(hidden_size, 32), nn.ReLU())
        self.heads = nn.ModuleList([nn.Linear(32, 1) for _ in quantiles])

    def forward(self, x):
        out, _ = self.lstm(x)
        last_hidden = out[:, -1, :]
        shared_repr = self.shared(last_hidden)
        preds = [head(shared_repr).squeeze(-1) for head in self.heads]
        return torch.stack(preds, dim=1)  # (batch, num_quantiles)


def pinball_loss(preds: torch.Tensor, target: torch.Tensor, quantiles: list) -> torch.Tensor:
    """
    Pinball (quantile) loss. For quantile q, under-prediction and
    over-prediction are penalized asymmetrically:
        loss = max(q * (target - pred), (q - 1) * (target - pred))
    This is what forces the p10 head to actually learn a LOW estimate (only
    10% of true values should fall below it) and the p90 head to learn a
    HIGH estimate (90% of true values should fall below it).

    preds: (batch, num_quantiles)
    target: (batch,)
    """
    target = target.unsqueeze(1)  # (batch, 1) to broadcast against preds
    errors = target - preds  # (batch, num_quantiles)
    losses = []
    for i, q in enumerate(quantiles):
        e = errors[:, i]
        losses.append(torch.max((q - 1) * e, q * e))
    return torch.stack(losses, dim=1).mean()


def calibration_check(y_true: np.ndarray, preds: np.ndarray, quantiles: list) -> dict:
    """
    Checks whether the predicted quantile intervals actually contain the
    true value at the expected rate. E.g. the p10-p90 interval SHOULD
    contain the true RUL about 80% of the time. If it doesn't, the
    uncertainty bounds are miscalibrated -- overconfident (too narrow) or
    underconfident (too wide).
    """
    p10_idx, p50_idx, p90_idx = 0, 1, 2
    p10, p50, p90 = preds[:, p10_idx], preds[:, p50_idx], preds[:, p90_idx]

    within_80 = np.mean((y_true >= p10) & (y_true <= p90))
    expected = quantiles[2] - quantiles[0]  # 0.9 - 0.1 = 0.8

    return {
        "expected_coverage": expected,
        "actual_coverage": float(within_80),
        "mean_interval_width": float(np.mean(p90 - p10)),
    }


def main():
    set_all_seeds(SEED)
    wandb.init(
        project="aerosentry",
        name=f"lstm_quantile_seed{SEED}",
        config={
            "model": "LSTM-Quantile",
            "seed": SEED,
            "window_size": WINDOW_SIZE,
            "batch_size": BATCH_SIZE,
            "hidden_size": HIDDEN_SIZE,
            "num_layers": NUM_LAYERS,
            "learning_rate": LEARNING_RATE,
            "num_epochs": NUM_EPOCHS,
            "loss_fn": "pinball (symmetric quantile loss)",
            "quantiles": QUANTILES,
        },
    )
    print(f"Device: {DEVICE}")

    train_df, test_df, true_rul = load_subset("FD001")
    train_df = add_piecewise_rul(train_df)
    test_df = add_test_rul(test_df, true_rul)

    feature_cols = [c for c in train_df.columns if c.startswith("sensor_")]
    stds = train_df[feature_cols].std()
    feature_cols = [c for c in feature_cols if stds[c] > 1e-4]

    train_df, test_df, norm_stats = normalize_sensors(train_df, test_df, sensor_cols=feature_cols)

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

    model = QuantileLSTM(input_size=len(feature_cols)).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_loss = float("inf")
    checkpoint_path = os.path.join(os.path.dirname(__file__), "lstm_quantile.pt")

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            preds = model(xb)
            loss = pinball_loss(preds, yb, QUANTILES)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                preds = model(xb)
                val_losses.append(pinball_loss(preds, yb, QUANTILES).item())

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        print(f"Epoch {epoch:3d}/{NUM_EPOCHS} | train_pinball={train_loss:.3f} | val_pinball={val_loss:.3f}")
        wandb.log({"epoch": epoch, "train_pinball_loss": train_loss, "val_pinball_loss": val_loss})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "model_state": model.state_dict(),
                "feature_cols": feature_cols,
                "window_size": WINDOW_SIZE,
                "quantiles": QUANTILES,
            }, checkpoint_path)

    print(f"\nBest val pinball loss: {best_val_loss:.3f}. Checkpoint saved to {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    # ---- Evaluate p50 (median) against NASA score/RMSE, same as point models ----
    X_test, y_test, test_units = make_test_windows_last_only(test_df, feature_cols, window_size=WINDOW_SIZE)
    X_test_t = torch.from_numpy(X_test).to(DEVICE)

    with torch.no_grad():
        preds_test = model(X_test_t).cpu().numpy()  # (n, 3) -> p10, p50, p90

    p50_preds = preds_test[:, 1]
    metrics = nasa_evaluate(y_test, p50_preds)
    print("\n=== Official FD001 test set evaluation (p50 / median) ===")
    print(f"RMSE:       {metrics['rmse']:.2f}")
    print(f"NASA score: {metrics['nasa_score']:.2f}")
    print("Compare against LSTM point-estimate baseline: RMSE 12.90, NASA score 283.63")

    # ---- Calibration check on VALIDATION set (test set is small, val gives
    # a more statistically meaningful calibration estimate) ----
    X_val_t = torch.from_numpy(X_val).to(DEVICE)
    with torch.no_grad():
        preds_val = model(X_val_t).cpu().numpy()

    calib = calibration_check(y_val, preds_val, QUANTILES)
    print("\n=== Calibration check (validation set) ===")
    print(f"Expected coverage (p10-p90 interval): {calib['expected_coverage']*100:.1f}%")
    print(f"Actual coverage:                      {calib['actual_coverage']*100:.1f}%")
    print(f"Mean interval width (p90 - p10):       {calib['mean_interval_width']:.2f} cycles")
    print("If actual coverage is far from 80%, the uncertainty bounds are "
          "miscalibrated -- report this honestly rather than hiding it.")

    wandb.log({
        "test_rmse_p50": metrics["rmse"],
        "test_nasa_score_p50": metrics["nasa_score"],
        "test_nasa_score_p50_avg": metrics["nasa_score"] / len(y_test),
        "calibration_expected_coverage": calib["expected_coverage"],
        "calibration_actual_coverage": calib["actual_coverage"],
        "calibration_interval_width": calib["mean_interval_width"],
    })
    wandb.finish()


if __name__ == "__main__":
    main()