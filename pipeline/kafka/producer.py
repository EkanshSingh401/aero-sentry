"""
Streams real C-MAPSS FD001 TEST data into Kafka, simulating live sensor
telemetry from a fleet of engines.

Sends RAW (un-normalized) sensor values -- normalization happens on the
consumer/inference side, matching how a real system would work.

Streams engines sequentially rather than interleaved. A real fleet would
have engines reporting concurrently -- the consumer is keyed by unit so it
would handle either case correctly, but this simplification makes the demo
easier to follow cycle-by-cycle.

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

BOOTSTRAP_SERVERS = "localhost:9092"
TOPIC = "sensor-raw"
DELAY_SECONDS = 0.05


def main():
    _, test_df, _ = load_subset("FD001")
    test_df = test_df.sort_values(["unit", "cycle"])

    sensor_cols = [c for c in test_df.columns if c.startswith("sensor_")]

    producer = KafkaProducer(
        bootstrap_servers=BOOTSTRAP_SERVERS,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )

    num_engines = test_df["unit"].nunique()
    print(f"Producer connected. Streaming {num_engines} engines "
          f"({len(test_df)} total cycles) to topic '{TOPIC}'...")

    sent = 0
    for _, row in test_df.iterrows():
        message = {
            "unit": int(row["unit"]),
            "cycle": int(row["cycle"]),
            **{c: float(row[c]) for c in sensor_cols},
        }
        producer.send(TOPIC, value=message)
        sent += 1
        if sent % 500 == 0:
            print(f"  ...sent {sent}/{len(test_df)} cycles")
        time.sleep(DELAY_SECONDS)

    producer.flush()
    print(f"Producer finished. Streamed {sent} total cycles across {num_engines} engines.")


if __name__ == "__main__":
    main()
