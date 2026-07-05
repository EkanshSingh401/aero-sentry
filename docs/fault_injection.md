# Fault Injection & Data-Quality Monitoring

## Objective

Build a data-quality layer that is genuinely INDEPENDENT of the RUL model
-- capable of flagging degraded telemetry even when the model's own
prediction looks unremarkable, since an LSTM has no built-in way to express
"I don't trust this input." Validate it against known-injected faults to
confirm it actually works, rather than assuming it does.

## Architecture

- `fault_injector.py` (producer side): injects dropped packets, sustained
  stuck-sensor faults, and noise spikes into real C-MAPSS telemetry before
  it reaches Kafka. The `_injected_fault` ground-truth tag is stripped
  before sending -- detection must be blind, from the data alone.
- `data_quality.py` (consumer side): three independent checks --
  sequence-gap detection, stuck-sensor detection (exact repetition), and
  outlier detection (z-score vs. training distribution).

## Two rounds of real bugs found via live testing, not assumed

### Bug 1: sensor_6 broke both the stuck and outlier checks

sensor_6 has an extremely small natural standard deviation (~0.0014).
This caused two related failures:
- Stuck check: coarse real-world sensor precision meant genuinely
  different readings frequently rounded to the exact same stored value
  across consecutive cycles -- indistinguishable from a truly frozen
  sensor under a pure exact-match rule.
- Outlier check: z = deviation / std. When std is nearly zero, even a
  tiny, completely normal fluctuation produces an enormous z-score.

Fix: exempt sensors with std below a threshold (0.01) from both checks.

### Bug 2: sensor_17 broke ONLY the stuck check, and the Bug 1 fix didn't touch it

sensor_17 has an ordinary standard deviation (~1.5) -- the std-based fix
above correctly left its outlier check active. But it still produced
constant stuck-check false positives. Root cause: sensor_17 is stored at
coarse/integer precision in the raw C-MAPSS files, a property of storage
resolution that isn't reflected in overall variance. Confirmed empirically
(not guessed) with compute_sensor_repeat_rates.py, which measures the
real rate of exact consecutive-cycle repeats per sensor across the actual
training data:

| Sensor | Natural repeat rate |
|---|---|
| sensor_1, 5, 10, 16, 18, 19 | 1.0000 (already excluded from the model as near-constant) |
| sensor_6 | 0.9644 |
| sensor_17 | 0.2990 |
| sensor_8, sensor_13 | ~0.09 |
| sensor_20, sensor_11 | ~0.027 |
| all other sensors | < 0.01 |

There's a clean gap between the highest "genuine" sensor (0.0086) and the
lowest "quantized" one (0.0269) -- a real bimodal split in the data, not
an arbitrary line drawn through a continuum.

Fix: exempt any sensor whose measured natural repeat rate exceeds 2%
from the STUCK check specifically (outlier check unaffected, since
variance -- not quantization -- is what breaks that check).

### A related fix: the fault injector itself was unwinnable

The original injector froze a sensor for exactly one cycle. The detector
requires 5 consecutive identical readings. A one-cycle freeze can never
satisfy that rule -- every injected stuck fault was undetectable by
construction, regardless of detector quality. Fixed by making stuck
faults persist for a realistic 8-15 cycles, matching how a real stuck
sensor (frozen ADC/register) actually behaves.

## Validation against real traffic

After both fixes, a full run was cross-checked line-by-line against the
producer's ground-truth injection log:

- 9/9 noise spikes on non-exempted sensors: caught, with sensible z-scores
- 8/8 dropped packets: caught via missing_cycles
- 1/1 sustained stuck fault: caught, with an exactly explainable trigger
  cycle (the frozen value coincides with the last real reading before the
  fault, so the 5-in-a-row window fills one cycle earlier than a naive
  calculation suggests)
- Zero false positives in the entire run

## Known, disclosed limitation

sensor_6's noise injections are never caught, because it's fully exempt
from the outlier check (the std-based fix from Bug 1). This is a real,
understood blind spot, not a hidden one: a genuine fault landing
specifically on a near-zero-variance sensor won't be flagged by this
z-score mechanism. A more complete fix would use an absolute deviation
threshold for such sensors instead of disabling the check outright --
noted as a possible future improvement rather than solved here.
