# AeroSentry

Turbofan engine health monitoring and RUL (Remaining Useful Life) prognostics platform, built on NASA's C-MAPSS dataset, with a full streaming pipeline, edge deployment on Orange Pi 5, and a physical hardware demo. Built to demonstrate operational systems engineering (defense/dual-use ML target companies), not just a notebook model.

## Setup
1. `python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt`
2. Download NASA C-MAPSS (search "NASA C-MAPSS Turbofan Engine Degradation" or Kaggle mirror "NASA Turbofan Jet Engine Data Set"). Place `train_FD001.txt`, `test_FD001.txt`, `RUL_FD001.txt` in `data/raw/`.
3. `cd data && python sanity_check.py`
4. `docker compose up -d` (Kafka: localhost:9092, InfluxDB: localhost:8086, Grafana: localhost:3000)

## Repo structure
- `data/` — C-MAPSS loader, RUL labeling, windowing
- `model/` — PyTorch training, ONNX export, RKNN conversion
- `pipeline/cpp_signal/` — C++ filtering/feature extraction
- `pipeline/kafka/` — Kafka producer/consumer
- `edge/` — RKNN inference + benchmarking (Orange Pi 5)
- `hardware/` — MPU-6050 + BLDC control (I2C)
- `dashboard/` — Grafana dashboards
- `docs/` — ConOps, architecture diagrams, benchmark reports

## Build phases
1. Data + model (LSTM to Transformer, quantile uncertainty)
2. Streaming pipeline (C++ to Kafka to InfluxDB, fault injection)
3. Edge deployment (ONNX to RKNN on Orange Pi 5, benchmarked)
4. Hardware demo (BLDC + MPU-6050 feeding the live pipeline)
5. Dashboard + ConOps documentation
