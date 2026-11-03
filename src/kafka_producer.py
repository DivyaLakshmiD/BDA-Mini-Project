import csv
import json
import time
from collections import defaultdict
from pathlib import Path

from kafka import KafkaProducer

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_DIR / "data" / "PdM_telemetry.csv"
KAFKA_SERVER = "localhost:9092"
TOPIC_NAME = "machine-sensor-stream"
DELAY_SECONDS = 0.05
MACHINE_COUNT = 10
RECORDS_PER_MACHINE = 10


def main():
    required_columns = {
        "datetime", "machineID", "volt",
        "rotate", "pressure", "vibration"
    }

    selected = defaultdict(list)

    with DATA_FILE.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)

        if not reader.fieldnames or not required_columns.issubset(reader.fieldnames):
            raise ValueError("Unexpected telemetry dataset columns.")

        for row in reader:
            machine_id = int(row["machineID"])

            if machine_id not in range(1, MACHINE_COUNT + 1):
                continue

            if len(selected[machine_id]) < RECORDS_PER_MACHINE:
                selected[machine_id].append({
                    "datetime": row["datetime"],
                    "machineID": machine_id,
                    "volt": float(row["volt"]),
                    "rotate": float(row["rotate"]),
                    "pressure": float(row["pressure"]),
                    "vibration": float(row["vibration"])
                })

            if all(
                len(selected[machine]) == RECORDS_PER_MACHINE
                for machine in range(1, MACHINE_COUNT + 1)
            ):
                break

    missing = [
        machine for machine in range(1, MACHINE_COUNT + 1)
        if len(selected[machine]) < RECORDS_PER_MACHINE
    ]

    if missing:
        raise RuntimeError(f"Insufficient records for machines: {missing}")

    events = [
        selected[machine][index]
        for index in range(RECORDS_PER_MACHINE)
        for machine in range(1, MACHINE_COUNT + 1)
    ]

    producer = KafkaProducer(
        bootstrap_servers=KAFKA_SERVER,
        value_serializer=lambda value: json.dumps(value).encode("utf-8"),
        acks="all",
        retries=5
    )

    sent = 0

    try:
        for event in events:
            producer.send(TOPIC_NAME, value=event).get(timeout=30)
            sent += 1
            print(
                f"Sent {sent}/{len(events)}: Machine "
                f"{event['machineID']} at {event['datetime']}",
                flush=True
            )
            time.sleep(DELAY_SECONDS)

        print(f"\nBatch completed. Total events sent: {sent}")
        print("Expected: 10 events per machine across Machines 1–10.")

    except KeyboardInterrupt:
        print("\nSensor replay stopped by user.")

    finally:
        producer.flush()
        producer.close()


if __name__ == "__main__":
    main()
