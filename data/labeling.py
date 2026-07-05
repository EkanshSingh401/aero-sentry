"""
RUL labeling and windowing for C-MAPSS.

Piecewise-linear RUL (standard convention from the RUL literature, e.g.
Heimes 2008, Zheng et al. 2017): assume the engine is fully healthy (flat,
capped RUL) up to some "knee point" before end-of-life, after which RUL
decays linearly to zero at the last cycle. This reflects the fact that
degradation is not linear from cycle 1 -- early cycles carry little
information about time-to-failure.

Default cap: 125 cycles (matches the widely-used convention in FD001/FD003
papers so your reported scores are comparable to published baselines).
"""

import numpy as np
import pandas as pd

DEFAULT_RUL_CAP = 125


def add_piecewise_rul(train_df: pd.DataFrame, cap: int = DEFAULT_RUL_CAP) -> pd.DataFrame:
    """
    Add a 'rul' column to a training dataframe with full run-to-failure
    trajectories. For each unit, RUL at the last cycle is 0, and RUL
    increases going backward, capped at `cap`.
    """
    df = train_df.copy()
    max_cycle_per_unit = df.groupby("unit")["cycle"].transform("max")
    raw_rul = max_cycle_per_unit - df["cycle"]
    df["rul"] = np.minimum(raw_rul, cap)
    return df


def add_test_rul(test_df: pd.DataFrame, true_rul: pd.Series, cap: int = DEFAULT_RUL_CAP) -> pd.DataFrame:
    """
    Add a 'rul' column to a truncated test dataframe. Since the test
    trajectories are cut off before failure, true_rul gives the RUL at the
    LAST cycle of each unit's trajectory. We back-compute RUL for every row
    the same way as training: rul_at_row = true_rul[unit] + (last_cycle - cycle),
    then cap.
    """
    df = test_df.copy()
    last_cycle_per_unit = df.groupby("unit")["cycle"].transform("max")
    unit_true_rul = df["unit"].map(true_rul)
    raw_rul = unit_true_rul + (last_cycle_per_unit - df["cycle"])
    df["rul"] = np.minimum(raw_rul, cap)
    return df


def normalize_sensors(train_df: pd.DataFrame, test_df: pd.DataFrame, sensor_cols=None):
    """
    Global z-score normalization fit on train, applied to both train and test.
    Returns (train_df, test_df, stats) where stats = {col: (mean, std)}.
    """
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
    """
    Slide a fixed-size window over each unit's trajectory.

    For units shorter than window_size, left-pads with the first row repeated
    (standard trick so short trajectories aren't dropped entirely).

    Returns:
        X: np.ndarray, shape (num_windows, window_size, num_features)
        y: np.ndarray, shape (num_windows,)  -- label at the LAST cycle of each window
        meta: list of (unit, end_cycle) for traceability
    """
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


def apply_ema_filter(df: pd.DataFrame, sensor_cols, alpha: float = 0.3) -> pd.DataFrame:
    """
    Applies an exponential moving average filter to sensor columns,
    independently per engine unit -- mirrors pipeline/cpp_signal/
    signal_filter.cpp EXACTLY (same formula, same per-unit reset behavior),
    so a model trained on this filtered representation matches what the
    live C++ stage actually produces in production.

    filtered[t] = alpha * raw[t] + (1 - alpha) * filtered[t-1]
    First value per unit is initialized to the raw value itself (no
    artificial zero-bias), then filtering proceeds forward in cycle order.

    This exists because of a real, disclosed finding: normalizing and
    filtering are both linear operations and provably commute, so there's
    no "order of operations" bug to fix by rearranging code. The actual
    issue is that a model trained on RAW data has a different learned
    input distribution than filtered data -- the only correct fix is
    training (or at least evaluating) directly on the filtered
    representation, which is what this function enables.
    """
    df = df.copy()
    for col in sensor_cols:
        filtered_values = []
        for unit, group in df.groupby("unit"):
            group = group.sort_values("cycle")
            raw_vals = group[col].values
            filtered = np.zeros_like(raw_vals, dtype=np.float64)
            filtered[0] = raw_vals[0]
            for t in range(1, len(raw_vals)):
                filtered[t] = alpha * raw_vals[t] + (1 - alpha) * filtered[t - 1]
            filtered_values.append(pd.Series(filtered, index=group.index))
        df[col] = pd.concat(filtered_values).sort_index()
    return df


def make_test_windows_last_only(df: pd.DataFrame, feature_cols, window_size: int = 30):
    """
    For test evaluation against the official RUL_FDxxx.txt file: take only the
    LAST window of each unit's trajectory (that's what the true_rul file
    corresponds to).
    """
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