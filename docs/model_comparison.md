# Model Comparison: LSTM vs. Transformer for RUL Prediction

## Objective

Benchmark two model architectures on identical data, under identical
conditions, to make an evidence-based choice for the production RUL
(Remaining Useful Life) estimator in AeroSentry -- rather than defaulting to
whichever architecture is currently fashionable.

## Methodology

Both models were trained and evaluated under strictly identical conditions
to ensure the comparison isolates architecture, not data handling:

- **Dataset:** NASA C-MAPSS FD001 (100 train engines, 100 test engines,
  single operating condition, single fault mode -- HPC degradation)
- **Labeling:** Piecewise-linear RUL, capped at 125 cycles (standard
  convention in the literature, ensures comparability with published
  baselines)
- **Feature set:** 21 sensor channels, near-constant sensors dropped
  (std < 1e-4 under FD001's single operating condition)
- **Normalization:** Z-score, fit on training data only, applied identically
  to both models
- **Windowing:** Fixed 30-cycle sliding windows
- **Train/validation split:** By engine unit (not by row) -- validation
  engines are never seen during training, preventing leakage across a
  single engine's trajectory
- **Optimizer:** Adam, learning rate 1e-3, 40 epochs, batch size 64
- **Evaluation:** Official FD001 test set, using the last window of each
  test engine's truncated trajectory, scored against the ground-truth
  RUL_FD001.txt values

### LSTM architecture
2-layer LSTM (hidden size 64, dropout 0.2 between layers) -> feedforward
head (64 -> 32 -> 1). Prediction taken from the final timestep's hidden
state.

### Transformer architecture
Encoder-only Transformer: linear input projection to d_model=64, sinusoidal
positional encoding, 3 encoder layers (4 attention heads, feedforward
dimension 128, dropout 0.2), mean-pooled over the time dimension ->
feedforward head (64 -> 32 -> 1). Attention weights are retained per layer
for explainability (see below).

## Results (Original Single-Run Comparison -- see correction below)

| Metric      | LSTM   | Transformer | Winner |
|-------------|--------|-------------|--------|
| RMSE        | 12.90  | 13.66       | LSTM   |
| NASA Score  | 283.63 | 315.01      | LSTM   |

Both metrics favor the LSTM in this single run. For reference, published
FD001 baselines using LSTM and attention-augmented architectures typically
report RMSE in the 12-15 range and NASA score in the 200-400 range -- both
models land within that published range.

**IMPORTANT: see the "Multi-Seed Robustness Study" section below -- this
single-run comparison was later found to be within normal run-to-run noise,
not a reliable architectural finding.**

## Interpretation (Original, Since Partially Corrected)

The Transformer underperforming the LSTM here is a real, well-documented
phenomenon in general, though the specific numeric comparison here didn't
hold up under multi-seed testing (see below):

1. **Dataset size.** After the unit-level train/validation split, the
   training set contains roughly 14,500 windows. Transformers have no
   built-in assumption that recent timesteps matter more than distant ones
   -- that has to be learned from data via attention weights. LSTMs have
   this bias built into their recurrence structure.

2. **Pooling strategy.** The Transformer's prediction is derived from a
   mean pool across all 30 timesteps in the window, which can dilute
   signal versus the LSTM's final hidden state.

3. **Capacity vs. regularization.** The Transformer has more parameters
   and needs more aggressive regularization to generalize well on a small
   dataset.

This tracks with the broader literature: attention-based architectures
reliably need more data than recurrent architectures to realize their
advantages. However, as shown below, this general tendency did not produce
a statistically distinguishable result in our specific multi-seed test.

## Quantile Regression (Uncertainty Bounds)

A third model -- an LSTM with three parallel output heads trained on
pinball (quantile) loss for p10/p50/p90 -- was built to provide an
uncertainty band around the RUL estimate, rather than a bare point number.

### Results

| Metric                          | Value  |
|----------------------------------|--------|
| p50 RMSE                         | 16.34  |
| p50 NASA score                   | 627.55 |
| p10-p90 expected coverage        | 80.0%  |
| p10-p90 actual coverage (val)    | 75.0%  |
| Mean interval width (p90 - p10)  | 22.39 cycles |

### Finding: the quantile model's own point estimate is worse, and that's informative

The quantile model's p50 head performs substantially worse than the
MSE-trained LSTM baseline on both RMSE and NASA score. Root cause: pinball
loss at the median (q=0.5) is mathematically equivalent to MAE, and more
importantly, **pinball loss has no awareness of the NASA scoring
function's asymmetric late/early penalty** -- a model can be
well-calibrated in the statistical sense while performing poorly on the
domain-specific safety metric, because nothing in its training objective
encodes that late errors are more dangerous than early ones.

### Decision

The uncertainty band (p10/p90 from this model) is retained and reported
alongside the point estimate, but the point estimate itself comes from
the original MSE-trained LSTM baseline, not this model's p50 head.

## Multi-Seed Robustness Study -- Correcting an Overconfident Earlier Conclusion

The comparisons above (LSTM vs. Transformer, and the quantile/asymmetric
results) were each based on a SINGLE training run per model. Before writing
final conclusions, we discovered a reproducibility bug: model weight
initialization, dropout masks, and DataLoader shuffling were never seeded
(only the train/val split was), so identical code produced meaningfully
different metrics on every run -- e.g. the LSTM baseline's RMSE varied from
12.90 to 14.76 across otherwise-identical runs.

After fixing this (seeding Python's random, NumPy, and PyTorch together),
each of the four models was retrained across three different seeds (42,
123, 777) to measure actual run-to-run variance, rather than trusting any
single draw.

### Results (mean +/- std across 3 seeds)

| Model | RMSE | NASA score |
|---|---|---|
| LSTM baseline | 14.03 +/- 0.27 | 344.09 +/- 54.14 |
| Transformer | 13.65 +/- 0.94 | 348.17 +/- 91.38 |
| Asymmetric (NASA-loss center) | 13.73 +/- 0.78 | 328.60 +/- 98.26 |
| Quantile (pinball p50) | 15.64 +/- 0.46 | 527.18 +/- 59.01 |

### Corrected conclusion

**The earlier claim that "the LSTM beats the Transformer" is NOT supported
once run-to-run variance is measured.** The RMSE and NASA score ranges for
the LSTM baseline, Transformer, and asymmetric-loss model all overlap
substantially -- the apparent winner in any single run falls within
ordinary noise for this dataset size, not a real, repeatable architectural
advantage.

**What DOES hold up: the quantile model's point-estimate degradation is
real, not noise.** Its RMSE (15.18-16.10 range) and NASA score (468-586
range) do not overlap with any of the other three models even accounting
for their variance. This confirms the original diagnosis is a genuine,
repeatable finding, not an artifact of one unlucky run.

### Honest limitation of this correction itself

Three seeds is enough to reveal that the original single-run comparison
was unreliable, but it is a small sample for computing a standard
deviation (2 degrees of freedom). A more rigorous version of this study
would use 5-10 seeds per model.

### Why this matters more than the original comparison

Catching and correcting an overconfident conclusion -- rather than keeping
whichever single-run result looked most favorable -- is the more
significant engineering finding here. It demonstrates that architecture
selection decisions on small datasets require variance-aware comparison,
not single-run benchmarking, which is a real and common failure mode in
applied ML work.

## Decision (Final, Post-Correction)

**No architecture (LSTM vs. Transformer vs. asymmetric-loss LSTM) showed a
statistically distinguishable advantage** in this multi-seed study. Given
that, the LSTM baseline is retained as the production model for
simplicity and lower computational cost, not because it's been shown to
be more accurate. The Transformer is retained for its extractable
attention weights (explainability layer). The quantile model's p10/p90
bounds are retained for uncertainty estimation; its point estimate is not
used.

## Next steps

- Extract and visualize Transformer attention weights against known
  degradation-sensitive sensors (sensor_2, sensor_3, sensor_4, sensor_7,
  sensor_11, sensor_15)
- All experiments now tracked in Weights & Biases, tagged by seed, for
  reproducible comparison as the model set grows
- Possible future improvement: retrain quantile heads with a loss function
  that incorporates the NASA scoring function's asymmetry directly
- Possible future improvement: extend the multi-seed study to 5-10 seeds
  for a more statistically solid variance estimate
