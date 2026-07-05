"""
Consumes live sensor telemetry, maintains a rolling 30-cycle window PER
ENGINE, runs the trained LSTM baseline for RUL inference the moment a
window fills, and writes predictions to InfluxDB.

Key design decisions:

1. State is keyed by engine unit (a dict of deques), so concurrent/
   interleaved engines are handled correctly.

2. Early-life padding matches the OFFLINE evaluation convention exactly:
   make_test_windows_last_only() pads short trajectories by repeating the
   first row. Without matching that here, a live engine's first ~29
   cycles would produce no predictions, and its 30th-cycle prediction
   would differ from the offline pipeline's -- a silent train/serve
   inconsistency.

3. Normalization uses the exact norm_stats.json saved during training.

Run from pipeline/kafka/ (with producer.py running in another terminal):
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
from lstm_baseline import LSTMRegressor

BOOTSTRAP_SERVERS = "localhost:9092"
TOPIC = "sensor-raw"

INFLUX_URL = "http://localhost:8086"
INFLUX_TOKEN = "aerosentry-dev-token"
INFLUX_ORG = "aerosentry"
INFLUX_BUCKET = "engine-health"

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "model")
CHECKPOINT_PATH = os.path.join(MODEL_DIR, "lstm_baseline.pt")
NORM_STATS_PATH = os.path.join(MODEL_DIR, "norm_stats.json")


def load_model_and_stats():
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
    feature_cols = checkpoint["feature_cols"]
    window_size = checkpoint["window_size"]

    model = LSTMRegressor(input_size=len(feature_cols))
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    with open(NORM_STATS_PATH) as f:
        norm_stats = json.load(f)

    return model, feature_cols, window_size, norm_stats


def normalize_row(row: dict, feature_cols: list, norm_stats: dict) -> list:
    vec = []
    for col in feature_cols:
        mean, std = norm_stats[col]
        raw = row[col]
        vec.append((raw - mean) / std if std > 1e-8 else raw - mean)
    return vec


def main():
    model, feature_cols, window_size, norm_stats = load_model_and_stats()
    print(f"Loaded model checkpoint. window_size={window_size}, "
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

    print(f"Consumer connected. Listening on topic '{TOPIC}'...\n")

    for message in consumer:
        row = message.value
        unit = row["unit"]
        cycle = row["cycle"]

        normalized_vec = normalize_row(row, feature_cols, norm_stats)

        if unit not in seen_units:
            seen_units.add(unit)
            print(f"[unit {unit}] New engine detected -- pre-filling window "
                  f"(matches offline padding convention).")
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
