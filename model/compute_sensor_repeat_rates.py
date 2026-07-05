"""
Computes, per sensor, the empirical rate of EXACT consecutive-cycle repeats
within the same engine, using real training data.

Why this exists: a variance-based threshold (std) doesn't correctly
identify which sensors will produce false "stuck" positives. sensor_6 has
tiny std AND repeats naturally. sensor_17 has ordinary std (~1.5) but
STILL repeats naturally at a real rate -- almost certainly because it's
stored at coarse/integer precision in the raw C-MAPSS files. Measuring the
actual repeat rate directly, from real data, is the correct way to find
this -- guessing a magnitude threshold is not.

Run from model/ directory:
    python compute_sensor_repeat_rates.py
"""

import os
import sys
import json

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "data"))
from loader import load_subset


def main():
    train_df, _, _ = load_subset("FD001")
    sensor_cols = [c for c in train_df.columns if c.startswith("sensor_")]

    repeat_rates = {}
    for col in sensor_cols:
        total_pairs = 0
        exact_repeats = 0
        for unit, group in train_df.groupby("unit"):
            values = group.sort_values("cycle")[col].values
            for i in range(1, len(values)):
                total_pairs += 1
                if values[i] == values[i - 1]:
                    exact_repeats += 1
        rate = exact_repeats / total_pairs if total_pairs > 0 else 0.0
        repeat_rates[col] = rate

    print("Empirical exact-repeat rate per sensor:")
    for col, rate in sorted(repeat_rates.items(), key=lambda x: -x[1]):
        flag = "  <-- likely quantized, exempt from stuck check" if rate > 0.02 else ""
        print(f"  {col}: {rate:.4f}{flag}")

    out_path = os.path.join(os.path.dirname(__file__), "sensor_repeat_rates.json")
    with open(out_path, "w") as f:
        json.dump(repeat_rates, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
