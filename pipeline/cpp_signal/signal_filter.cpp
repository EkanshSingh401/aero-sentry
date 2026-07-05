/*
 * signal_filter.cpp
 *
 * Consumes raw sensor telemetry from Kafka topic "sensor-raw", applies an
 * exponential moving average (EMA) filter independently per (engine unit,
 * sensor), and republishes to "sensor-filtered" with BOTH the original raw
 * values and the new filtered values in the same message.
 *
 * Why both in one message: the data-quality monitor needs RAW values --
 * a filter would smooth out exactly the anomalies it's trying to detect.
 * Inference may benefit from smoothed input. Keeping both lets downstream
 * consumers choose per-purpose.
 *
 * EMA formula: filtered[t] = alpha * raw[t] + (1 - alpha) * filtered[t-1]
 * State resets (to the first observed value, not zero) when a new engine
 * unit is seen -- same boundary-isolation principle as the Python
 * consumer's window padding.
 *
 * Build:
 *   g++ -std=c++17 -O2 signal_filter.cpp -o signal_filter -lrdkafka++
 *
 * Run:
 *   ./signal_filter [broker] [input_topic] [output_topic] [alpha]
 */

#include <iostream>
#include <string>
#include <unordered_map>
#include <memory>
#include <csignal>
#include <cstring>

#include <librdkafka/rdkafkacpp.h>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

static volatile bool running = true;

void signal_handler(int) {
    running = false;
}

class EmaFilterBank {
public:
    explicit EmaFilterBank(double alpha) : alpha_(alpha) {}

    double update(int unit, const std::string& sensor, double raw_value) {
        auto& unit_state = state_[unit];
        auto it = unit_state.find(sensor);
        if (it == unit_state.end()) {
            unit_state[sensor] = raw_value;
            return raw_value;
        }
        double prev_filtered = it->second;
        double filtered = alpha_ * raw_value + (1.0 - alpha_) * prev_filtered;
        it->second = filtered;
        return filtered;
    }

    void reset_unit(int unit) {
        state_.erase(unit);
    }

    bool has_unit(int unit) const {
        return state_.find(unit) != state_.end();
    }

private:
    double alpha_;
    std::unordered_map<int, std::unordered_map<std::string, double>> state_;
};

int main(int argc, char** argv) {
    std::string brokers = (argc > 1) ? argv[1] : "localhost:9092";
    std::string input_topic = (argc > 2) ? argv[2] : "sensor-raw";
    std::string output_topic = (argc > 3) ? argv[3] : "sensor-filtered";
    double alpha = (argc > 4) ? std::stod(argv[4]) : 0.3;

    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    std::cout << "signal_filter starting\n"
              << "  brokers:      " << brokers << "\n"
              << "  input_topic:  " << input_topic << "\n"
              << "  output_topic: " << output_topic << "\n"
              << "  alpha:        " << alpha << "\n";

    std::string errstr;

    std::unique_ptr<RdKafka::Conf> conf(RdKafka::Conf::create(RdKafka::Conf::CONF_GLOBAL));
    conf->set("bootstrap.servers", brokers, errstr);
    conf->set("group.id", "aerosentry-signal-filter", errstr);
    conf->set("auto.offset.reset", "earliest", errstr);

    std::unique_ptr<RdKafka::KafkaConsumer> consumer(
        RdKafka::KafkaConsumer::create(conf.get(), errstr));
    if (!consumer) {
        std::cerr << "Failed to create consumer: " << errstr << "\n";
        return 1;
    }

    RdKafka::ErrorCode subscribe_err = consumer->subscribe({input_topic});
    if (subscribe_err != RdKafka::ERR_NO_ERROR) {
        std::cerr << "Failed to subscribe: " << RdKafka::err2str(subscribe_err) << "\n";
        return 1;
    }

    std::unique_ptr<RdKafka::Conf> producer_conf(RdKafka::Conf::create(RdKafka::Conf::CONF_GLOBAL));
    producer_conf->set("bootstrap.servers", brokers, errstr);

    std::unique_ptr<RdKafka::Producer> producer(
        RdKafka::Producer::create(producer_conf.get(), errstr));
    if (!producer) {
        std::cerr << "Failed to create producer: " << errstr << "\n";
        return 1;
    }

    EmaFilterBank filter_bank(alpha);
    long messages_processed = 0;

    std::cout << "Ready. Consuming from '" << input_topic
              << "', publishing filtered output to '" << output_topic << "'.\n";

    while (running) {
        std::unique_ptr<RdKafka::Message> msg(consumer->consume(1000));

        if (msg->err() == RdKafka::ERR__TIMED_OUT) {
            continue;
        }
        if (msg->err() != RdKafka::ERR_NO_ERROR) {
            std::cerr << "Consume error: " << msg->errstr() << "\n";
            continue;
        }

        std::string payload(static_cast<const char*>(msg->payload()), msg->len());

        json parsed;
        try {
            parsed = json::parse(payload);
        } catch (const json::parse_error& e) {
            std::cerr << "JSON parse error, skipping message: " << e.what() << "\n";
            continue;
        }

        if (!parsed.contains("unit") || !parsed.contains("cycle")) {
            std::cerr << "Message missing unit/cycle, skipping.\n";
            continue;
        }

        int unit = parsed["unit"].get<int>();
        int cycle = parsed["cycle"].get<int>();

        if (!filter_bank.has_unit(unit)) {
            std::cout << "[unit " << unit << "] New engine detected, "
                      << "initializing EMA filter state.\n";
        }

        json out = parsed;

        for (auto& [key, value] : parsed.items()) {
            if (key.rfind("sensor_", 0) == 0 && value.is_number()) {
                double raw = value.get<double>();
                double filtered = filter_bank.update(unit, key, raw);
                out[key + "_filtered"] = filtered;
            }
        }

        std::string out_str = out.dump();

        RdKafka::ErrorCode produce_err = producer->produce(
            output_topic,
            RdKafka::Topic::PARTITION_UA,
            RdKafka::Producer::RK_MSG_COPY,
            const_cast<char*>(out_str.c_str()), out_str.size(),
            nullptr, 0,
            0, nullptr);

        if (produce_err != RdKafka::ERR_NO_ERROR) {
            std::cerr << "Produce failed: " << RdKafka::err2str(produce_err) << "\n";
        }

        producer->poll(0);

        messages_processed++;
        if (messages_processed % 500 == 0) {
            std::cout << "  ...processed " << messages_processed << " messages\n";
        }
    }

    std::cout << "Shutting down. Flushing producer...\n";
    producer->flush(5000);
    consumer->close();

    std::cout << "Processed " << messages_processed << " total messages.\n";
    return 0;
}
