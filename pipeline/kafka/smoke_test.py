"""
Minimal Kafka smoke test. Proves the producer -> broker -> consumer path
works end to end BEFORE wiring in real sensor data or model inference.
Isolating this step means if something breaks later, we know it's in the
real logic, not the plumbing.

Run in two separate terminals from pipeline/kafka/:
    Terminal 1: python smoke_test.py producer
    Terminal 2: python smoke_test.py consumer
"""

import sys
import time
import json

from kafka import KafkaProducer, KafkaConsumer

BOOTSTRAP_SERVERS = "localhost:9092"
TOPIC = "aerosentry-smoke-test"


def run_producer():
    producer = KafkaProducer(
        bootstrap_servers=BOOTSTRAP_SERVERS,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )
    print(f"Producer connected. Publishing to topic '{TOPIC}'...")

    for i in range(20):
        message = {"seq": i, "timestamp": time.time(), "payload": f"test message {i}"}
        producer.send(TOPIC, value=message)
        print(f"  Sent: {message}")
        time.sleep(1)

    producer.flush()
    print("Producer done.")


def run_consumer():
    consumer = KafkaConsumer(
        TOPIC,
        bootstrap_servers=BOOTSTRAP_SERVERS,
        auto_offset_reset="earliest",
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        consumer_timeout_ms=30000,
    )
    print(f"Consumer connected. Listening on topic '{TOPIC}'...")

    count = 0
    for message in consumer:
        print(f"  Received: {message.value}")
        count += 1

    print(f"Consumer done. Received {count} messages total.")


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("producer", "consumer"):
        print("Usage: python smoke_test.py [producer|consumer]")
        sys.exit(1)

    if sys.argv[1] == "producer":
        run_producer()
    else:
        run_consumer()
