"""
Fault injection for testing whether the pipeline's data-quality monitoring
actually catches degraded telemetry, independent of what the RUL model
itself predicts.

Three fault types:
  - DROPPED PACKET: a cycle's message never gets sent.
  - STUCK SENSOR: freezes and STAYS frozen for a sustained run of cycles.
  - NOISE SPIKE: a large, physically implausible jump for one cycle.

BUG FIX: originally froze a sensor for exactly ONE cycle, but the detector
requires 5 CONSECUTIVE identical readings -- a one-cycle freeze could
never be detected by construction. Now sustains 8-15 cycles, matching how
a real stuck sensor (ADC hang) actually behaves.
"""

import random


class FaultInjector:
    def __init__(self, norm_stats: dict, drop_prob=0.03, stuck_prob=0.01,
                 noise_prob=0.03, noise_magnitude_std=8.0,
                 stuck_duration_min=8, stuck_duration_max=15, seed=None):
        self.norm_stats = norm_stats
        self.sensor_cols = list(norm_stats.keys())
        self.drop_prob = drop_prob
        self.stuck_prob = stuck_prob
        self.noise_prob = noise_prob
        self.noise_magnitude_std = noise_magnitude_std
        self.stuck_duration_min = stuck_duration_min
        self.stuck_duration_max = stuck_duration_max
        self.rng = random.Random(seed)
        self._last_values = {}
        self._active_stuck = {}

    def maybe_drop(self) -> bool:
        return self.rng.random() < self.drop_prob

    def maybe_corrupt(self, unit: int, row: dict) -> dict:
        row = dict(row)
        prev = self._last_values.get(unit)

        active = self._active_stuck.get(unit)
        if active and active["remaining"] > 0:
            row[active["col"]] = active["value"]
            row["_injected_fault"] = f"stuck:{active['col']}"
            active["remaining"] -= 1
            if active["remaining"] == 0:
                del self._active_stuck[unit]

        elif prev is not None and self.rng.random() < self.stuck_prob:
            stuck_col = self.rng.choice(self.sensor_cols)
            frozen_value = prev[stuck_col]
            duration = self.rng.randint(self.stuck_duration_min, self.stuck_duration_max)
            row[stuck_col] = frozen_value
            row["_injected_fault"] = f"stuck:{stuck_col}"
            self._active_stuck[unit] = {
                "col": stuck_col, "value": frozen_value, "remaining": duration - 1
            }

        elif self.rng.random() < self.noise_prob:
            noisy_col = self.rng.choice(self.sensor_cols)
            mean, std = self.norm_stats[noisy_col]
            spike = self.rng.choice([-1, 1]) * self.noise_magnitude_std * std
            row[noisy_col] = row[noisy_col] + spike
            row["_injected_fault"] = f"noise:{noisy_col}"

        self._last_values[unit] = dict(row)
        return row
