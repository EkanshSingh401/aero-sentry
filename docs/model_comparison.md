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
  `RUL_FD001.txt` values

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

## Results

| Metric      | LSTM   | Transformer | Winner |
|-------------|--------|-------------|--------|
| RMSE        | 12.90  | 13.66       | LSTM   |
| NASA Score  | 283.63 | 315.01      | LSTM   |

Both metrics favor the LSTM. For reference, published FD001 baselines using
LSTM and attention-augmented architectures typically report RMSE in the
12-15 range and NASA score in the 200-400 range -- both models land within
that published range, so neither result reflects a data or pipeline bug;
this is a genuine architecture-level outcome.

## Interpretation

The Transformer underperforming the LSTM here is a real, well-documented
phenomenon, not a sign of a broken implementation:

1. **Dataset size.** After the unit-level train/validation split, the
   training set contains roughly 14,500 windows. Transformers have no
   built-in assumption that recent timesteps matter more than distant ones
   -- that has to be learned from data via attention weights. LSTMs have
   this bias built into their recurrence structure. On a dataset this size,
   the LSTM's inductive bias is doing useful work that the Transformer has
   to learn from scratch and doesn't have enough examples to fully acquire.

2. **Pooling strategy.** The Transformer's prediction is derived from a
   mean pool across all 30 timesteps in the window. This can dilute
   signal if the most informative cycles are concentrated near the end of
   the window, whereas the LSTM's final hidden state naturally emphasizes
   recent context by construction.

3. **Capacity vs. regularization.** The Transformer has more parameters
   and needs more aggressive regularization to generalize well on a small
   dataset. The dropout used here (0.2) may not fully compensate for the
   architecture's larger capacity relative to the available training data.

This tracks with the broader literature: attention-based architectures
reliably need more data than recurrent architectures to realize their
advantages, and small, structured benchmark datasets like C-MAPSS FD001
tend to favor LSTMs and GRUs unless the Transformer is specifically
augmented (larger multi-condition subsets like FD002/FD004, data
augmentation, or hybrid attention+recurrence architectures).

## Decision

**The LSTM is used as the production RUL estimator** going into the
streaming pipeline (Phase 2) and edge deployment (Phase 3), based on its
superior RMSE and NASA score on the official test set.

**The Transformer is retained**, not discarded, for a different purpose:
its per-layer attention weights are directly extractable and interpretable,
letting us identify which cycles within a 30-cycle window the model
weighted most heavily for a given prediction. The LSTM's hidden state does
not offer this directly. This becomes the foundation for the
explainability layer in later phases -- a capability need, not an accuracy
need.

## What this demonstrates

Choosing the architecture that actually performs better on held-out data,
over the architecture that sounds more sophisticated, is a deliberate
engineering decision under real evidence -- not an assumption. The two
models are also not mutually exclusive: using the LSTM for the deployed
prediction and the Transformer for interpretability is a legitimate systems
design choice, matching how production ML systems often combine multiple
models for different purposes rather than picking one "best" model
end-to-end.

## Next steps

- Add quantile regression heads (p10/p50/p90) to the LSTM for uncertainty
  bounds on RUL predictions
- Extract and visualize Transformer attention weights against known
  degradation-sensitive sensors (sensor_2, sensor_3, sensor_4, sensor_7,
  sensor_11, sensor_15 -- confirmed visually drifting in the Phase 0 sanity
  check)
- Track all future experiments in Weights & Biases for reproducible
  comparison as the model set grows

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
MSE-trained LSTM baseline on both RMSE and NASA score -- the NASA score
gap (627.55 vs 283.63) is disproportionately larger than the RMSE gap,
which is itself a signal worth explaining rather than ignoring.

Root cause: pinball loss at the median (q=0.5) is mathematically equivalent
to MAE, which optimizes toward a different central tendency than MSE
whenever the error distribution is not symmetric (RUL values are bounded
at 0 and capped at 125, so this asymmetry is expected). More importantly,
**pinball loss has no awareness of the NASA scoring function's asymmetric
late/early penalty** -- a model can be well-calibrated in the statistical
sense (correctly capturing the 10th/90th percentiles) while performing
poorly on the domain-specific safety metric, because nothing in its
training objective encodes that late errors are more dangerous than early
ones. This is a genuine, general ML engineering lesson: the training loss
and the deployment metric are not automatically aligned, and treating them
as interchangeable is a common, subtle mistake.

The calibration result itself (75% actual vs. 80% expected coverage) is
reasonably close given the validation set size, and the resulting 22.39
cycle interval width is a usable, honest uncertainty band -- the model's
statistical calibration is not the problem; using its own point estimate
as the production prediction is.

### Decision

The uncertainty band (p10/p90 from this model) is retained and reported
alongside the point estimate, but the **point estimate itself comes from
the original MSE-trained LSTM baseline** (RMSE 12.90), not from this
model's p50 head. This decouples "what's the best single-number
prediction" from "how confident should we be in it" -- two related but
distinct questions that don't have to be answered by the same training
objective.

## Updated Next Steps

- Extract and visualize Transformer attention weights against known
  degradation-sensitive sensors (sensor_2, sensor_3, sensor_4, sensor_7,
  sensor_11, sensor_15)
- Track all future experiments in Weights & Biases for reproducible
  comparison as the model set grows
- Possible future improvement: retrain quantile heads with a loss function
  that incorporates the NASA scoring function's asymmetry directly, so the
  calibrated bounds and the deployment-relevant point estimate are
  optimized toward the same real-world cost structure
