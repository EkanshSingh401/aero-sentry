# AeroSentry

Turbofan engine health monitoring and Remaining Useful Life (RUL) prognostics platform, built on NASA's C-MAPSS dataset. AeroSentry is designed to demonstrate operational systems engineering — real data pipelines, fault tolerance, and honest evaluation — not just a notebook model.

## Status

**Phase 0 (data), Phase 1 (modeling), and Phase 2 (streaming pipeline) are complete.** Phase 3 (edge deployment), Phase 4 (hardware demo), and Phase 5 (dashboard) are planned next.

## What this project actually demonstrates

- A real-time streaming inference pipeline (Kafka + a C++ signal-processing stage + Python inference + InfluxDB), not a static notebook
- A fault-injection and data-quality monitoring system that detects corrupted sensor data independently of the ML model, validated blind against ground truth
- Rigorous model evaluation: a multi-seed study that caught and corrected an earlier, overconfident single-run conclusion
- A real train/serve distribution bug (feeding a raw-trained model filtered data) found, mathematically diagnosed, and properly fixed by retraining — not patched around
- Every non-trivial claim in this repo is backed by a test, a cross-check against ground truth, or a documented limitation — see docs/

## Architecture

producer.py (Python, streams real C-MAPSS data + injects faults)
  -> Kafka "sensor-raw"
  -> signal_filter (C++, EMA filtering per engine/sensor via librdkafka + nlohmann::json)
  -> Kafka "sensor-filtered" (raw AND filtered values in one message)
  -> consumer.py (Python)
       - RAW values -> independent data-quality monitor (stuck sensors, dropped packets, statistical outliers)
       - FILTERED values -> LSTM RUL inference (trained directly on filtered data)
  -> InfluxDB (predictions + quality flags)
  -> Grafana (dashboard -- not yet built)

## Repo structure

- data/ — C-MAPSS loader, RUL labeling, windowing, EMA filtering
- model/ — PyTorch training scripts, scoring function, multi-seed study
- pipeline/cpp_signal/ — C++ EMA filter (Kafka consumer/producer)
- pipeline/kafka/ — Python producer, consumer, fault injector, data-quality monitor
- docs/ — write-ups of every major finding (see below)
- docker-compose.yml — Kafka, Zookeeper, InfluxDB, Grafana

## Key documents

- docs/model_comparison.md — LSTM vs. Transformer vs. quantile vs. asymmetric-loss models, including the multi-seed study that corrected an earlier overconfident conclusion
- docs/fault_injection.md — the fault-injection/data-quality system, including two real bugs found via live testing and how they were diagnosed and fixed
- docs/signal_filtering.md — the C++ signal-processing stage, the train/serve distribution mismatch it exposed, and the fix

## Setup

### 1. Python environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

### 2. Get the C-MAPSS dataset
Download from the NASA Prognostics Data Repository (https://data.nasa.gov/docs/legacy/CMAPSSData.zip) or the Kaggle mirror ("NASA Turbofan Jet Engine Data Set"). Place train_FD001.txt, test_FD001.txt, RUL_FD001.txt in data/raw/.

### 3. Sanity check the data
cd data && python sanity_check.py

### 4. Start the infrastructure
docker compose up -d
Kafka: localhost:9092 / InfluxDB: localhost:8086 / Grafana: localhost:3000

### 5. Train the models
cd model
python lstm_baseline.py
python transformer_model.py
python lstm_quantile.py
python lstm_asymmetric.py
python multi_seed_study.py
python compute_sensor_repeat_rates.py
python lstm_filtered.py

### 6. Build the C++ signal filter
sudo apt install -y librdkafka-dev nlohmann-json3-dev g++
cd pipeline/cpp_signal
g++ -std=c++17 -O2 signal_filter.cpp -o signal_filter -lrdkafka++

### 7. Run the live pipeline
Four terminals, in this order:

Terminal 1: cd pipeline/cpp_signal && ./signal_filter localhost:9092 sensor-raw sensor-filtered 0.3
Terminal 2: cd pipeline/kafka && python consumer.py
Terminal 3: cd pipeline/kafka && python producer.py

## Build phases

1. Data + model (done) — LSTM/Transformer/quantile/asymmetric-loss RUL models, multi-seed validated
2. Streaming pipeline (done) — Kafka producer/consumer, C++ signal filtering, independent fault detection, all validated against live traffic
3. Edge deployment (planned) — ONNX export, RKNN conversion, Orange Pi 5 NPU benchmarking
4. Hardware demo (planned) — BLDC motor + MPU-6050 accelerometer feeding the live pipeline
5. Dashboard + ConOps (planned) — Grafana dashboard and concept-of-operations documentation
