"""
Asymmetric-loss LSTM for C-MAPSS RUL prediction.

Closes a specific gap found in lstm_quantile.py: that model's p50 head was
trained on symmetric pinball loss, which has no knowledge of the NASA
scoring function's late/early asymmetry, so its point estimate underperformed
a plain MSE-trained model on the metric that actually matters for
deployment.

This model uses THREE heads with two different, deliberately-chosen losses:
  - p10, p90: trained on standard pinball loss. These exist purely to give
    a statistically calibrated uncertainty interval -- calibration is a
    genuinely different objective than point-estimate accuracy, and pinball
    loss is the right tool for it.
  - center: trained DIRECTLY on a differentiable version of the NASA scoring
    function itself (nasa_loss_torch). This head's job is the point
    estimate, so it is optimized against the actual deployment metric, not
    a generic proxy.

This means the point estimate is no longer decoupled from the uncertainty
band by using two separate models -- one model, three heads, each trained
against the objective that's actually appropriate for its job.

Run from the model/ directory:
    python lstm_asymmetric.py
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

from scoring import evaluate as nasa_evaluate, nasa_loss_torch
from lstm_baseline import RULDataset, split_by_unit, set_all_seeds, WINDOW_SIZE, BATCH_SIZE, HIDDEN_SIZE, NUM_LAYERS, LEARNING_RATE, NUM_EPOCHS, VAL_FRACTION, SEED
from lstm_quantile import pinball_loss, calibration_check

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BOUND_QUANTILES = [0.1, 0.9]
CENTER_LOSS_WEIGHT = 1.0
BOUND_LOSS_WEIGHT = 1.0


class AsymmetricLSTM(nn.Module):
    """Three heads: p10, center (point estimate), p90."""

    def __init__(self, input_size, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0.0,
        )
        self.shared = nn.Sequential(nn.Linear(hidden_size, 32), nn.ReLU())
        self.p10_head = nn.Linear(32, 1)
        self.center_head = nn.Linear(32, 1)
        self.p90_head = nn.Linear(32, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        last_hidden = out[:, -1, :]
        shared_repr = self.shared(last_hidden)
        p10 = self.p10_head(shared_repr).squeeze(-1)
        center = self.center_head(shared_repr).squeeze(-1)
        p90 = self.p90_head(shared_repr).squeeze(-1)
        return p10, center, p90


def combined_loss(p10, center, p90, target):
    bound_loss = pinball_loss(torch.stack([p10, p90], dim=1), target, BOUND_QUANTILES)
    center_loss = nasa_loss_torch(center, target)
    total = BOUND_LOSS_WEIGHT * bound_loss + CENTER_LOSS_WEIGHT * center_loss
    return total, bound_loss, center_loss


def main():
    set_all_seeds(SEED)
    wandb.init(
        project="aerosentry",
        name=f"lstm_asymmetric_seed{SEED}",
        config={
            "model": "LSTM-Asymmetric",
            "seed": SEED,
            "window_size": WINDOW_SIZE,
            "batch_size": BATCH_SIZE,
            "hidden_size": HIDDEN_SIZE,
            "num_layers": NUM_LAYERS,
            "learning_rate": LEARNING_RATE,
            "num_epochs": NUM_EPOCHS,
            "center_loss_fn": "differentiable NASA score (clamp_d=200)",
            "bound_loss_fn": "pinball (p10/p90)",
            "grad_clip_max_norm": 5.0,
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

    model = AsymmetricLSTM(input_size=len(feature_cols)).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_loss = float("inf")
    checkpoint_path = os.path.join(os.path.dirname(__file__), "lstm_asymmetric.pt")

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            p10, center, p90 = model(xb)
            loss, bound_l, center_l = combined_loss(p10, center, p90, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                p10, center, p90 = model(xb)
                loss, _, _ = combined_loss(p10, center, p90, yb)
                val_losses.append(loss.item())

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
            }, checkpoint_path)

    print(f"\nBest val loss: {best_val_loss:.3f}. Checkpoint saved to {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    X_test, y_test, test_units = make_test_windows_last_only(test_df, feature_cols, window_size=WINDOW_SIZE)
    X_test_t = torch.from_numpy(X_test).to(DEVICE)

    with torch.no_grad():
        p10_test, center_test, p90_test = model(X_test_t)
        center_test = center_test.cpu().numpy()

    metrics = nasa_evaluate(y_test, center_test)
    print("\n=== Official FD001 test set evaluation (NASA-loss center head) ===")
    print(f"RMSE:       {metrics['rmse']:.2f}")
    print(f"NASA score: {metrics['nasa_score']:.2f}")
    print("Compare against:")
    print("  MSE-trained LSTM baseline:        RMSE 12.90, NASA score 283.63")
    print("  Pinball-loss quantile p50:        RMSE 16.34, NASA score 627.55")

    X_val_t = torch.from_numpy(X_val).to(DEVICE)
    with torch.no_grad():
        p10_val, center_val, p90_val = model(X_val_t)
        p10_val_np = p10_val.cpu().numpy()
        p90_val_np = p90_val.cpu().numpy()

    within_80 = np.mean((y_val >= p10_val_np) & (y_val <= p90_val_np))
    interval_width = float(np.mean(p90_val_np - p10_val_np))
    print("\n=== Calibration check (validation set, p10-p90 bounds) ===")
    print(f"Expected coverage: 80.0%")
    print(f"Actual coverage:   {within_80*100:.1f}%")
    print(f"Mean interval width: {interval_width:.2f} cycles")

    wandb.log({
        "test_rmse": metrics["rmse"],
        "test_nasa_score": metrics["nasa_score"],
        "test_nasa_score_avg": metrics["nasa_score"] / len(y_test),
        "calibration_expected_coverage": 0.80,
        "calibration_actual_coverage": float(within_80),
        "calibration_interval_width": interval_width,
        "best_val_loss": best_val_loss,
    })
    wandb.finish()


if __name__ == "__main__":
    main()