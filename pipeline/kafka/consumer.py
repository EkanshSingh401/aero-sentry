"""
Consumes live sensor telemetry from Kafka topic "sensor-filtered" -- the
output of the C++ signal_filter stage, which sits between the Python
producer and this consumer. Full pipeline:

    producer.py (Python, + fault injection)
      -> Kafka "sensor-raw"
      -> signal_filter (C++, EMA filtering per unit/sensor)
      -> Kafka "sensor-filtered"  (raw AND filtered values, same message)
      -> consumer.py (this file)

Maintains a rolling 30-cycle window PER ENGINE using FILTERED values for
inference, runs an LSTM trained DIRECTLY ON FILTERED DATA (lstm_filtered.py)
for RUL the moment a window fills, and writes predictions to InfluxDB.
Data-quality checks use the RAW values from the same message -- a filter
would smooth out exactly the anomalies those checks are trying to detect.

Key design decisions:

1. State is keyed by engine unit (a dict of deques), so concurrent/
   interleaved engines are handled correctly.

2. Early-life padding matches the OFFLINE evaluation convention exactly
   (make_test_windows_last_only() pads short trajectories by repeating
   the first row).

3. FIXED train/serve distribution mismatch, not just disclosed: the model
   loaded here (lstm_filtered.pt) was trained directly on EMA-filtered
   data with matching normalization stats (norm_stats_filtered.json),
   not on raw data with raw-data stats. Filtering and normalization are
   both linear and provably commute, so there was never an "order of
   operations" bug to fix by rearranging code -- the only correct fix was
   training on the actual distribution the model sees in production,
   which is what lstm_filtered.py does.

Run from pipeline/kafka/ (with producer.py AND signal_filter running):
    python consumer.py
"""

import os
import sys
import json
from collections import defaultdict, deque

import torch
from kafka import KafkaConsumer
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "model"))
from lstm_baseline import LSTMRegressor  # same architecture works for the filtered-trained model too
from data_quality import DataQualityMonitor

BOOTSTRAP_SERVERS = "localhost:9092"
TOPIC = "sensor-filtered"

INFLUX_URL = "http://localhost:8086"
INFLUX_TOKEN = "aerosentry-dev-token"
INFLUX_ORG = "aerosentry"
INFLUX_BUCKET = "engine-health"

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "model")
CHECKPOINT_PATH = os.path.join(MODEL_DIR, "lstm_filtered.pt")
NORM_STATS_PATH = os.path.join(MODEL_DIR, "norm_stats_filtered.json")
REPEAT_RATES_PATH = os.path.join(MODEL_DIR, "sensor_repeat_rates.json")


def load_model_and_stats():
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
    feature_cols = checkpoint["feature_cols"]
    window_size = checkpoint["window_size"]

    model = LSTMRegressor(input_size=len(feature_cols))
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    with open(NORM_STATS_PATH) as f:
        norm_stats = json.load(f)

    repeat_rates = {}
    if os.path.exists(REPEAT_RATES_PATH):
        with open(REPEAT_RATES_PATH) as f:
            repeat_rates = json.load(f)
    else:
        print(f"WARNING: {REPEAT_RATES_PATH} not found.")

    return model, feature_cols, window_size, norm_stats, repeat_rates


def normalize_row(row: dict, feature_cols: list, norm_stats: dict) -> list:
    """
    Normalizes using the FILTERED sensor values (from the C++ signal_filter
    stage) for inference, using norm_stats_filtered.json -- computed from
    EMA-filtered training data, matching this exact transformation. This
    is a real fix, not a disclosed workaround: an earlier version used
    raw-data norm_stats.json here, which was a genuine train/serve
    mismatch, since the model that consumed it had been trained on raw
    (unfiltered) data.
    """
    vec = []
    for col in feature_cols:
        mean, std = norm_stats[col]
        filtered = row[f"{col}_filtered"]
        vec.append((filtered - mean) / std if std > 1e-8 else filtered - mean)
    return vec


def main():
    model, feature_cols, window_size, norm_stats, repeat_rates = load_model_and_stats()
    print(f"Loaded FILTERED-trained model checkpoint. window_size={window_size}, "
          f"num_features={len(feature_cols)}")

    influx_client = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)
    write_api = influx_client.write_api(write_options=SYNCHRONOUS)

    consumer = KafkaConsumer(
        TOPIC,
        bootstrap_servers=BOOTSTRAP_SERVERS,
        auto_offset_reset="earliest",
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
    )

    windows = defaultdict(lambda: deque(maxlen=window_size))
    seen_units = set()
    quality_monitor = DataQualityMonitor(norm_stats, repeat_rates=repeat_rates)

    print(f"Consumer connected. Listening on topic '{TOPIC}'...\n")

    for message in consumer:
        row = message.value
        unit = row["unit"]
        cycle = row["cycle"]

        quality_flags = quality_monitor.check(unit, cycle, row)
        if quality_flags:
            flags_str = "; ".join(quality_flags)
            print(f"[unit {unit}] cycle {cycle:4d}: DATA QUALITY FLAG -- {flags_str}")

            quality_point = (
                Point("data_quality")
                .tag("unit", str(unit))
                .field("cycle", cycle)
                .field("flag_count", len(quality_flags))
                .field("flags", flags_str)
            )
            write_api.write(bucket=INFLUX_BUCKET, record=quality_point)

        normalized_vec = normalize_row(row, feature_cols, norm_stats)

        if unit not in seen_units:
            seen_units.add(unit)
            print(f"[unit {unit}] New engine detected -- pre-filling window.")
            for _ in range(window_size - 1):
                windows[unit].append(normalized_vec)

        windows[unit].append(normalized_vec)

        if len(windows[unit]) == window_size:
            x = torch.tensor([list(windows[unit])], dtype=torch.float32)
            with torch.no_grad():
                pred_rul = model(x).item()

            point = (
                Point("engine_health")
                .tag("unit", str(unit))
                .field("cycle", cycle)
                .field("predicted_rul", float(pred_rul))
            )
            write_api.write(bucket=INFLUX_BUCKET, record=point)

            print(f"[unit {unit}] cycle {cycle:4d}: predicted RUL = {pred_rul:6.1f}")


if __name__ == "__main__":
    main()