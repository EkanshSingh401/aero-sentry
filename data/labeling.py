"""
RUL labeling and windowing for C-MAPSS.

Piecewise-linear RUL: assume the engine is fully healthy (flat, capped RUL)
up to some "knee point" before end-of-life, after which RUL decays linearly
to zero at the last cycle. Default cap: 125 cycles (standard convention so
scores are comparable to published FD001/FD003 baselines).
"""

import numpy as np
import pandas as pd

DEFAULT_RUL_CAP = 125


def add_piecewise_rul(train_df: pd.DataFrame, cap: int = DEFAULT_RUL_CAP) -> pd.DataFrame:
    df = train_df.copy()
    max_cycle_per_unit = df.groupby("unit")["cycle"].transform("max")
    raw_rul = max_cycle_per_unit - df["cycle"]
    df["rul"] = np.minimum(raw_rul, cap)
    return df


def add_test_rul(test_df: pd.DataFrame, true_rul: pd.Series, cap: int = DEFAULT_RUL_CAP) -> pd.DataFrame:
    df = test_df.copy()
    last_cycle_per_unit = df.groupby("unit")["cycle"].transform("max")
    unit_true_rul = df["unit"].map(true_rul)
    raw_rul = unit_true_rul + (last_cycle_per_unit - df["cycle"])
    df["rul"] = np.minimum(raw_rul, cap)
    return df


def normalize_sensors(train_df: pd.DataFrame, test_df: pd.DataFrame, sensor_cols=None):
    if sensor_cols is None:
        sensor_cols = [c for c in train_df.columns if c.startswith("sensor_")]

    train_df = train_df.copy()
    test_df = test_df.copy()
    stats = {}
    for col in sensor_cols:
        mean = train_df[col].mean()
        std = train_df[col].std()
        std = std if std > 1e-8 else 1.0
        stats[col] = (mean, std)
        train_df[col] = (train_df[col] - mean) / std
        test_df[col] = (test_df[col] - mean) / std

    return train_df, test_df, stats


def make_windows(df: pd.DataFrame, feature_cols, window_size: int = 30, label_col: str = "rul"):
    X_list, y_list, meta = [], [], []

    for unit, group in df.groupby("unit"):
        group = group.sort_values("cycle")
        feats = group[feature_cols].values
        labels = group[label_col].values
        cycles = group["cycle"].values

        n = feats.shape[0]
        if n < window_size:
            pad_amount = window_size - n
            pad = np.repeat(feats[0:1], pad_amount, axis=0)
            feats = np.concatenate([pad, feats], axis=0)
            labels = np.concatenate([np.repeat(labels[0], pad_amount), labels])
            cycles = np.concatenate([np.repeat(cycles[0], pad_amount), cycles])
            n = feats.shape[0]

        for end in range(window_size, n + 1):
            X_list.append(feats[end - window_size:end])
            y_list.append(labels[end - 1])
            meta.append((unit, cycles[end - 1]))

    X = np.stack(X_list).astype(np.float32)
    y = np.array(y_list, dtype=np.float32)
    return X, y, meta


def make_test_windows_last_only(df: pd.DataFrame, feature_cols, window_size: int = 30):
    X_list, y_list, units = [], [], []

    for unit, group in df.groupby("unit"):
        group = group.sort_values("cycle")
        feats = group[feature_cols].values
        labels = group["rul"].values

        n = feats.shape[0]
        if n < window_size:
            pad_amount = window_size - n
            pad = np.repeat(feats[0:1], pad_amount, axis=0)
            feats = np.concatenate([pad, feats], axis=0)
            n = feats.shape[0]

        X_list.append(feats[n - window_size:n])
        y_list.append(labels[-1])
        units.append(unit)

    X = np.stack(X_list).astype(np.float32)
    y = np.array(y_list, dtype=np.float32)
    return X, y, units
