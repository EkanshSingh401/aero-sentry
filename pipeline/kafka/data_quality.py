"""
Data-quality monitoring, deliberately independent of the RUL model.

Three checks: sequence gaps, stuck sensors (exact repetition), outliers
(z-score vs training distribution).

TWO ROUNDS OF BUG FIXES from live testing:
Round 1: sensor_6 (tiny std) false-positived on BOTH stuck and outlier
checks. Fixed with a std-based exemption (MIN_STD_FOR_CHECKS).
Round 2: sensor_17 (ordinary std ~1.5, but 30% natural repeat rate from
coarse storage precision) still false-positived on the STUCK check only.
Fixed by measuring each sensor's real repeat rate from training data
(compute_sensor_repeat_rates.py) rather than guessing a magnitude proxy.
"""

STUCK_WINDOW = 5
OUTLIER_Z_THRESHOLD = 5.0
MIN_STD_FOR_CHECKS = 0.01
MAX_NATURAL_REPEAT_RATE = 0.02


class DataQualityMonitor:
    def __init__(self, norm_stats: dict, repeat_rates: dict = None,
                 stuck_window=STUCK_WINDOW,
                 outlier_z_threshold=OUTLIER_Z_THRESHOLD,
                 min_std_for_checks=MIN_STD_FOR_CHECKS,
                 max_natural_repeat_rate=MAX_NATURAL_REPEAT_RATE):
        self.norm_stats = norm_stats
        self.repeat_rates = repeat_rates or {}
        self.stuck_window = stuck_window
        self.outlier_z_threshold = outlier_z_threshold
        self.min_std_for_checks = min_std_for_checks
        self.max_natural_repeat_rate = max_natural_repeat_rate
        self._raw_history = {}
        self._last_cycle = {}

        low_std = {col for col, (mean, std) in norm_stats.items() if std < min_std_for_checks}
        quantized = {
            col for col, rate in self.repeat_rates.items()
            if rate > max_natural_repeat_rate
        }

        self.exempt_from_outlier = low_std
        self.exempt_from_stuck = low_std | quantized

        if self.exempt_from_outlier:
            print(f"[DataQualityMonitor] Exempting from OUTLIER check (std < "
                  f"{min_std_for_checks}): {sorted(self.exempt_from_outlier)}")
        if self.exempt_from_stuck:
            print(f"[DataQualityMonitor] Exempting from STUCK check (low std or "
                  f"natural repeat rate > {max_natural_repeat_rate}): "
                  f"{sorted(self.exempt_from_stuck)}")

    def check(self, unit: int, cycle: int, raw_row: dict) -> list:
        flags = []

        last_cycle = self._last_cycle.get(unit)
        if last_cycle is not None and cycle != last_cycle + 1:
            gap = cycle - last_cycle - 1
            if gap > 0:
                flags.append(f"missing_cycles:{gap}")
        self._last_cycle[unit] = cycle

        history = self._raw_history.setdefault(unit, {})
        for col, (mean, std) in self.norm_stats.items():
            val = raw_row.get(col)
            if val is None:
                continue

            if col not in self.exempt_from_stuck:
                hist = history.setdefault(col, [])
                hist.append(val)
                if len(hist) > self.stuck_window:
                    hist.pop(0)
                if len(hist) == self.stuck_window and len(set(hist)) == 1:
                    flags.append(f"stuck:{col}")

            if col not in self.exempt_from_outlier and std > 1e-8:
                z = abs((val - mean) / std)
                if z > self.outlier_z_threshold:
                    flags.append(f"outlier:{col}:z={z:.1f}")

        return flags
