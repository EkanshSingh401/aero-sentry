"""
Quick sanity check on loaded C-MAPSS data: plots sensors known to show clear
monotonic degradation trends against cycle, for a few sample units.
"""

import matplotlib.pyplot as plt
from loader import load_subset
from labeling import add_piecewise_rul

SENSORS_TO_CHECK = ["sensor_2", "sensor_3", "sensor_4", "sensor_7", "sensor_11", "sensor_15"]
SAMPLE_UNITS = [1, 2, 3]


def main():
    train_df, test_df, rul_series = load_subset("FD001")
    train_df = add_piecewise_rul(train_df)

    print(f"Loaded {train_df['unit'].nunique()} training units, "
          f"{train_df.shape[0]} total rows")
    print(f"Cycle range: {train_df['cycle'].min()} - {train_df['cycle'].max()}")
    print(f"RUL range after capping: {train_df['rul'].min()} - {train_df['rul'].max()}")

    fig, axes = plt.subplots(len(SENSORS_TO_CHECK), 1, figsize=(8, 2.2 * len(SENSORS_TO_CHECK)), sharex=False)

    for ax, sensor in zip(axes, SENSORS_TO_CHECK):
        for unit in SAMPLE_UNITS:
            sub = train_df[train_df["unit"] == unit]
            ax.plot(sub["cycle"], sub[sensor], label=f"unit {unit}", alpha=0.8)
        ax.set_ylabel(sensor)
        ax.legend(fontsize=7, loc="upper right")

    axes[-1].set_xlabel("cycle")
    fig.suptitle("Sensor trends vs. cycle (look for monotonic drift = degradation signal)")
    fig.tight_layout()
    out_path = "sensor_trend_check.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()
