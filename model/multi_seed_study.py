"""
Multi-seed robustness study.

A single training run's result is a sample from a distribution, not a fixed
truth -- we saw this directly when the LSTM-vs-Transformer "winner" flipped
between two different (unseeded, then seeded-but-different-seed) runs. This
script runs each of the four models across multiple seeds and reports
mean +/- std for RMSE and NASA score, so any comparison between models is
backed by an actual measure of variance, not a single arbitrary draw.

This does NOT retrain with different data splits -- the train/val split
(by engine unit) is controlled by SEED too via split_by_unit(), so different
seeds here vary BOTH the split and the model initialization together. This
is a deliberate choice: it tests overall pipeline robustness, not just
weight-init sensitivity. If you wanted to isolate weight-init variance only,
you'd fix the split and vary just torch's seed -- worth noting as a
limitation, not hidden.

Run from the model/ directory:
    python multi_seed_study.py

Takes several minutes (4 models x 3 seeds x ~1-2 min each on CPU).
"""

import os
import re
import statistics
import subprocess
import sys

SEEDS = [42, 123, 777]
SCRIPTS = {
    "LSTM baseline": "lstm_baseline.py",
    "Transformer": "transformer_model.py",
    "Quantile (p50)": "lstm_quantile.py",
    "Asymmetric (center)": "lstm_asymmetric.py",
}

RMSE_PATTERN = re.compile(r"RMSE:\s+([\d.]+)")
NASA_PATTERN = re.compile(r"NASA score:\s+([\d.]+)")


def run_one(script_name: str, seed: int) -> dict:
    env = os.environ.copy()
    env["AEROSENTRY_SEED"] = str(seed)
    env["CUDA_VISIBLE_DEVICES"] = ""

    result = subprocess.run(
        [sys.executable, script_name],
        env=env,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"  !! {script_name} (seed={seed}) FAILED, stderr tail:")
        print("  " + "\n  ".join(result.stderr.strip().splitlines()[-10:]))
        return {"rmse": None, "nasa_score": None}

    # The "final evaluation" block always comes last in stdout, but scripts
    # with multiple RMSE/NASA score mentions (quantile, asymmetric compare
    # against other baselines in their print statements) mean we take the
    # FIRST match, which corresponds to that script's OWN test evaluation
    # printed immediately after "=== Official FD001 test set evaluation".
    rmse_matches = RMSE_PATTERN.findall(result.stdout)
    nasa_matches = NASA_PATTERN.findall(result.stdout)

    if not rmse_matches or not nasa_matches:
        print(f"  !! Could not parse metrics from {script_name} (seed={seed}) output")
        return {"rmse": None, "nasa_score": None}

    return {"rmse": float(rmse_matches[0]), "nasa_score": float(nasa_matches[0])}


def main():
    all_results = {}

    for model_name, script_name in SCRIPTS.items():
        print(f"\n{'=' * 60}")
        print(f"Running {model_name} across seeds {SEEDS}")
        print(f"{'=' * 60}")

        rmses, nasa_scores = [], []
        for seed in SEEDS:
            print(f"  seed={seed}...", end=" ", flush=True)
            metrics = run_one(script_name, seed)
            if metrics["rmse"] is not None:
                rmses.append(metrics["rmse"])
                nasa_scores.append(metrics["nasa_score"])
                print(f"RMSE={metrics['rmse']:.2f}, NASA={metrics['nasa_score']:.2f}")
            else:
                print("FAILED")

        all_results[model_name] = {"rmses": rmses, "nasa_scores": nasa_scores}

    print(f"\n\n{'=' * 70}")
    print("MULTI-SEED SUMMARY (mean +/- std across seeds)")
    print(f"{'=' * 70}")
    print(f"{'Model':<22} {'RMSE (mean+-std)':<20} {'NASA score (mean+-std)':<25}")
    print("-" * 70)

    for model_name, results in all_results.items():
        rmses = results["rmses"]
        nasa_scores = results["nasa_scores"]
        if len(rmses) < 2:
            print(f"{model_name:<22} insufficient successful runs to compute std")
            continue
        rmse_mean, rmse_std = statistics.mean(rmses), statistics.stdev(rmses)
        nasa_mean, nasa_std = statistics.mean(nasa_scores), statistics.stdev(nasa_scores)
        print(f"{model_name:<22} {rmse_mean:5.2f} +/- {rmse_std:4.2f}      "
              f"{nasa_mean:7.2f} +/- {nasa_std:6.2f}")

    print("\nHOW TO READ THIS:")
    print("If two models' mean +/- std ranges overlap substantially, the")
    print("apparent 'winner' from any single run is not a reliable finding --")
    print("it's within the noise band. A real difference should show up as")
    print("means that are separated by more than roughly one std deviation.")


if __name__ == "__main__":
    main()