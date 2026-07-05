"""
Streams real C-MAPSS FD001 TEST data into Kafka, simulating live sensor
telemetry from a fleet of engines, with optional fault injection.

Sends RAW (un-normalized) sensor values. The _injected_fault ground-truth
tag is stripped before sending -- a real sensor wouldn't announce its own
malfunction, so the consumer must detect faults blind from the data alone.

Run from pipeline/kafka/:
    python producer.py
"""

import os
import sys
import json
import time

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "data"))
from loader import load_subset

from kafka import KafkaProducer
from fault_injector import FaultInjector

BOOTSTRAP_SERVERS = "localhost:9092"
TOPIC = "sensor-raw"
DELAY_SECONDS = 0.05

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "model")
NORM_STATS_PATH = os.path.join(MODEL_DIR, "norm_stats.json")


def main():
    _, test_df, _ = load_subset("FD001")
    test_df = test_df.sort_values(["unit", "cycle"])

    sensor_cols = [c for c in test_df.columns if c.startswith("sensor_")]

    with open(NORM_STATS_PATH) as f:
        norm_stats = json.load(f)
    injector = FaultInjector(norm_stats, drop_prob=0.03, stuck_prob=0.005, noise_prob=0.03, seed=42)

    producer = KafkaProducer(
        bootstrap_servers=BOOTSTRAP_SERVERS,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )

    num_engines = test_df["unit"].nunique()
    print(f"Producer connected. Streaming {num_engines} engines "
          f"({len(test_df)} total cycles) to topic '{TOPIC}'...")
    print("Fault injection ENABLED: ~3% drop, ~3% stuck-sensor, ~3% noise-spike "
          "(each independently rolled per message).\n")

    sent, dropped = 0, 0
    for _, row_series in test_df.iterrows():
        unit = int(row_series["unit"])
        cycle = int(row_series["cycle"])
        message = {
            "unit": unit,
            "cycle": cycle,
            **{c: float(row_series[c]) for c in sensor_cols},
        }

        if injector.maybe_drop():
            dropped += 1
            print(f"  [FAULT-INJECTED] Dropped cycle {cycle} for unit {unit} "
                  f"(consumer will never see this message)")
            time.sleep(DELAY_SECONDS)
            continue

        message = injector.maybe_corrupt(unit, message)
        injected = message.pop("_injected_fault", None)
        if injected:
            print(f"  [FAULT-INJECTED] {injected} at unit {unit} cycle {cycle} "
                  f"(ground truth -- consumer must detect this blind)")

        producer.send(TOPIC, value=message)
        sent += 1
        if sent % 500 == 0:
            print(f"  ...sent {sent}/{len(test_df)} cycles ({dropped} dropped so far)")
        time.sleep(DELAY_SECONDS)

    producer.flush()
    print(f"\nProducer finished. Streamed {sent} cycles, dropped {dropped}, "
          f"across {num_engines} engines.")


if __name__ == "__main__":
    main()
