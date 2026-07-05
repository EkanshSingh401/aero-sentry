"""
C-MAPSS data loader.

File format (space-delimited, no header):
  col 1       -> unit number (engine id)
  col 2       -> cycle (time step)
  col 3-5     -> operational settings (3)
  col 6-26    -> sensor readings (21)

Total: 26 columns.

Expected files in data/raw/:
  train_FD00{1..4}.txt
  test_FD00{1..4}.txt
  RUL_FD00{1..4}.txt   (true RUL for each unit in the test set, one value per line)
"""

import os
import pandas as pd

RAW_DIR = os.path.join(os.path.dirname(__file__), "raw")

COLUMN_NAMES = (
    ["unit", "cycle"]
    + [f"op_setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)


def _load_raw_file(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", header=None)
    df = df.dropna(axis=1, how="all")
    if df.shape[1] != 26:
        raise ValueError(
            f"Expected 26 columns after cleanup, got {df.shape[1]} in {path}. "
            "Check that the raw file matches the standard C-MAPSS format."
        )
    df.columns = COLUMN_NAMES
    return df


def load_subset(subset: str = "FD001"):
    """
    Load train, test, and true RUL data for a given C-MAPSS subset.
    Returns: train_df, test_df, rul_series
    """
    train_path = os.path.join(RAW_DIR, f"train_{subset}.txt")
    test_path = os.path.join(RAW_DIR, f"test_{subset}.txt")
    rul_path = os.path.join(RAW_DIR, f"RUL_{subset}.txt")

    for p in (train_path, test_path, rul_path):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Missing {p}. Download the C-MAPSS dataset and place the raw "
                f"train_{subset}.txt, test_{subset}.txt, RUL_{subset}.txt files "
                f"in {RAW_DIR}/"
            )

    train_df = _load_raw_file(train_path)
    test_df = _load_raw_file(test_path)

    rul_values = pd.read_csv(rul_path, sep=r"\s+", header=None).dropna(axis=1, how="all")
    rul_values = rul_values.iloc[:, 0].reset_index(drop=True)
    rul_series = pd.Series(
        rul_values.values, index=range(1, len(rul_values) + 1), name="true_rul"
    )
    rul_series.index.name = "unit"

    return train_df, test_df, rul_series


if __name__ == "__main__":
    train_df, test_df, rul_series = load_subset("FD001")
    print(f"train: {train_df.shape}, units: {train_df['unit'].nunique()}")
    print(f"test:  {test_df.shape}, units: {test_df['unit'].nunique()}")
    print(f"rul:   {rul_series.shape}")
    print(train_df.head())
