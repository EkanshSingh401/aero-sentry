"""
Quick sanity check: queries InfluxDB for the most recent predictions written
by consumer.py, to confirm the full pipeline actually persisted data.

Run from pipeline/kafka/:
    python check_influx.py
"""

from influxdb_client import InfluxDBClient

INFLUX_URL = "http://localhost:8086"
INFLUX_TOKEN = "aerosentry-dev-token"
INFLUX_ORG = "aerosentry"
INFLUX_BUCKET = "engine-health"


def main():
    client = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)
    query_api = client.query_api()

    query = f'''
    from(bucket: "{INFLUX_BUCKET}")
      |> range(start: -1h)
      |> filter(fn: (r) => r._measurement == "engine_health")
      |> sort(columns: ["_time"], desc: true)
      |> limit(n: 20)
    '''

    tables = query_api.query(query)

    rows = []
    for table in tables:
        for record in table.records:
            rows.append({
                "time": record.get_time(),
                "unit": record.values.get("unit"),
                "field": record.get_field(),
                "value": record.get_value(),
            })

    if not rows:
        print("No data found in InfluxDB. Check that consumer.py actually ran "
              "and wrote successfully.")
        return

    print(f"Found {len(rows)} recent points in InfluxDB:\n")
    for row in rows:
        print(f"  {row['time']} | unit={row['unit']} | {row['field']}={row['value']}")


if __name__ == "__main__":
    main()
