# C++ Signal Filtering: Full Pipeline Integration

## Architecture

producer.py (Python, + fault injection) -> Kafka "sensor-raw" -> signal_filter (C++, EMA filtering per unit/sensor) -> Kafka "sensor-filtered" (raw AND filtered values, same message) -> consumer.py (Python): RAW values go to data-quality checks, FILTERED values go to RUL inference -> InfluxDB

## Why C++ for this stage

Signal filtering close to the data source is a standard real-world pattern in embedded/edge telemetry systems -- lightweight, performance-critical transformation before more expensive downstream processing. Implemented with librdkafka (C++ wrapper) and nlohmann::json, both standard Ubuntu apt packages.

## A real train/serve distribution mismatch, found and properly fixed

Feeding a model trained on RAW data with FILTERED inputs at inference time is a genuine mismatch, not fixable by rearranging code. Verified mathematically: z-score normalization and EMA filtering are both linear operations, and provably commute (the algebra telescopes exactly). There was never an "order of operations" bug -- normalizing-then-filtering and filtering-then-normalizing give byte-identical results.

The actual issue: the model was trained on the RAW distribution's statistical character (variance, autocorrelation). Filtered data has different variance regardless of normalization. The only correct fix is training directly on the distribution the model will actually see in production.

### The fix

1. data/labeling.py apply_ema_filter() -- Python EMA filter matching signal_filter.cpp exactly. Cross-validated against real live C++ output, reproduced identical values to the same floating-point representation.
2. model/lstm_filtered.py -- retrains the LSTM directly on EMA-filtered training data, with normalization stats computed on the filtered distribution.
3. pipeline/kafka/consumer.py -- loads lstm_filtered.pt and norm_stats_filtered.json, so inference is now fully consistent with training.

## Results

Raw-trained baseline (multi-seed mean): RMSE 14.03 +/- 0.27, NASA score 344.09 +/- 54.14
Filtered-trained model (single run): RMSE 14.03, NASA score 403.38

RMSE is statistically indistinguishable from the raw baseline. NASA score is slightly above the raw baseline's one-std range, but well within the run-to-run variance already observed for the raw model across seeds (which spanned 283-454). Honest conclusion: filtering shows no clear, statistically supported effect based on this one run. A multi-seed study would be needed to say more -- noted as a next step, not completed here.

## End-to-end validation against live traffic

Ran the full four-component pipeline and cross-checked every data-quality flag against the producer's ground-truth injection log. Every noise spike, dropped packet, and sustained stuck fault checked matched exactly, zero false positives -- confirming raw-value quality checks remain fully decoupled from the filtered-value inference path.
