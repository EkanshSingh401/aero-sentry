"""
Transformer encoder baseline for C-MAPSS RUL prediction.

Same data pipeline as lstm_baseline.py (identical windowing, normalization,
unit-level split, scoring) so results are directly comparable. The only
difference is the model architecture: an encoder-only Transformer with
learned positional encoding over the 30-cycle window, instead of an LSTM.

Why this matters beyond "it might score better": the attention weights this
model produces are extractable and interpretable -- you can see which
timesteps in the 30-cycle window the model weighted most heavily when making
a prediction. That's the foundation for the explainability layer (Phase 1.5
in the project plan) and is a substantively different capability than the
LSTM baseline, not just a architecture swap for its own sake.

Run from the model/ directory:
    python transformer_model.py
"""

import os
import sys
import json
import math

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "data"))
from loader import load_subset
from labeling import add_piecewise_rul, add_test_rul, normalize_sensors, make_windows, make_test_windows_last_only

from scoring import evaluate as nasa_evaluate
from lstm_baseline import RULDataset, split_by_unit, set_all_seeds, WINDOW_SIZE, BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS, VAL_FRACTION, SEED

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

D_MODEL = 64
NUM_HEADS = 4
NUM_ENCODER_LAYERS = 3
DIM_FEEDFORWARD = 128
DROPOUT = 0.2


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding (Vaswani et al.), added to the
    input projection so the model knows WHERE in the 30-cycle window each
    timestep falls, since attention itself has no notion of order."""

    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class TransformerEncoderLayerWithAttn(nn.Module):
    """Wraps nn.MultiheadAttention directly (instead of nn.TransformerEncoderLayer)
    so we can retrieve attention weights for explainability -- the built-in
    nn.TransformerEncoderLayer doesn't expose them cleanly."""

    def __init__(self, d_model, num_heads, dim_feedforward, dropout):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x):
        attn_out, attn_weights = self.self_attn(x, x, x, need_weights=True, average_attn_weights=True)
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = self.norm2(x + self.dropout(ff_out))
        return x, attn_weights  # attn_weights: (batch, seq_len, seq_len)


class TransformerRegressor(nn.Module):
    def __init__(self, input_size, d_model=D_MODEL, num_heads=NUM_HEADS,
                 num_layers=NUM_ENCODER_LAYERS, dim_feedforward=DIM_FEEDFORWARD, dropout=DROPOUT):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([
            TransformerEncoderLayerWithAttn(d_model, num_heads, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        self.head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        self.last_attn_weights = None  # populated on each forward pass

    def forward(self, x, return_attn=False):
        x = self.input_proj(x)
        x = self.pos_encoding(x)

        attn_weights_per_layer = []
        for layer in self.layers:
            x, attn_w = layer(x)
            attn_weights_per_layer.append(attn_w)

        # Mean-pool over the time dimension (alternative to using only the
        # last timestep -- lets every cycle in the window contribute).
        pooled = x.mean(dim=1)
        out = self.head(pooled).squeeze(-1)

        if return_attn:
            self.last_attn_weights = attn_weights_per_layer
        return out


def main():
    set_all_seeds(SEED)
    wandb.init(
        project="aerosentry",
        name=f"transformer_seed{SEED}",
        config={
            "model": "Transformer",
            "seed": SEED,
            "window_size": WINDOW_SIZE,
            "batch_size": BATCH_SIZE,
            "d_model": D_MODEL,
            "num_heads": NUM_HEADS,
            "num_encoder_layers": NUM_ENCODER_LAYERS,
            "dim_feedforward": DIM_FEEDFORWARD,
            "dropout": DROPOUT,
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
    stds = train_df[feature_cols].std()
    keep_cols = [c for c in feature_cols if stds[c] > 1e-4]
    feature_cols = keep_cols

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

    model = TransformerRegressor(input_size=len(feature_cols)).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()

    best_val_loss = float("inf")
    checkpoint_path = os.path.join(os.path.dirname(__file__), "transformer_model.pt")

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
                "d_model": D_MODEL,
                "num_heads": NUM_HEADS,
                "num_layers": NUM_ENCODER_LAYERS,
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
    print("\n=== Official FD001 test set evaluation (Transformer) ===")
    print(f"RMSE:       {metrics['rmse']:.2f}")
    print(f"NASA score: {metrics['nasa_score']:.2f}")
    print("Compare directly against the LSTM baseline: RMSE 12.90, NASA score 283.63")

    wandb.log({
        "test_rmse": metrics["rmse"],
        "test_nasa_score": metrics["nasa_score"],
        "test_nasa_score_avg": metrics["nasa_score"] / len(y_test),
        "best_val_loss": best_val_loss,
    })
    wandb.finish()


if __name__ == "__main__":
    main()