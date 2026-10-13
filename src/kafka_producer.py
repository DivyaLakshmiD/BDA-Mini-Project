
import csv
import json
import time
from pathlib import Path
from collections import defaultdict
from kafka import KafkaProducer

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_DIR / "data" / "PdM_telemetry.csv"
PREDICTIONS_FILE = PROJECT_DIR / "results" / "test_predictions.csv"
KAFKA_SERVER = "localhost:9092"
TOPIC_NAME = "machine-sensor-stream"

RISK_TARGETS = {
    "LOW": 240,
    "MEDIUM": 40,
    "HIGH": 20
}

DELAY_SECONDS = 0.02


def main():
    required_columns = {
        "datetime", "machineID", "volt",
        "rotate", "pressure", "vibration"
    }

    risk_records = defaultdict(list)

    with PREDICTIONS_FILE.open(
        "r", newline="", encoding="utf-8-sig"
    ) as file:
        for row in csv.DictReader(file):
            probability = json.loads(row["probability"])[1]

            if probability >= 0.70:
                risk = "HIGH"
            elif probability >= 0.40:
                risk = "MEDIUM"
            else:
                risk = "LOW"

            risk_records[risk].append({
                "machineID": int(row["machineID"]),
                "datetime": row["datetime"],
                "probability": probability
            })

    selected_keys = {}

    for risk, target in RISK_TARGETS.items():
        records = risk_records[risk]

        if risk == "LOW":
            records.sort(key=lambda row: row["probability"])
        elif risk == "MEDIUM":
            records.sort(
                key=lambda row: abs(row["probability"] - 0.55)
            )
        else:
            records.sort(
                key=lambda row: row["probability"],
                reverse=True
            )

        chosen = []
        chosen_keys = set()
        used_machines = set()

        for row in records:
            key = (row["machineID"], row["datetime"])

            if row["machineID"] not in used_machines:
                chosen.append(row)
                chosen_keys.add(key)
                used_machines.add(row["machineID"])

            if len(chosen) >= target:
                break

        for row in records:
            key = (row["machineID"], row["datetime"])

            if len(chosen) >= target:
                break

            if key not in chosen_keys:
                chosen.append(row)
                chosen_keys.add(key)

        if len(chosen) < target:
            raise RuntimeError(
                f"Not enough {risk} records: "
                f"needed {target}, found {len(chosen)}"
            )

        for row in chosen:
            key = (row["machineID"], row["datetime"])
            selected_keys[key] = risk

    events = []

    with DATA_FILE.open(
        "r", newline="", encoding="utf-8-sig"
    ) as file:
        reader = csv.DictReader(file)

        if not reader.fieldnames or not required_columns.issubset(
            reader.fieldnames
        ):
            raise ValueError("Unexpected telemetry dataset columns.")

        for row in reader:
            key = (int(row["machineID"]), row["datetime"])

            if key in selected_keys:
                events.append({
                    "datetime": row["datetime"],
                    "machineID": int(row["machineID"]),
                    "volt": float(row["volt"]),
                    "rotate": float(row["rotate"]),
                    "pressure": float(row["pressure"]),
                    "vibration": float(row["vibration"])
                })

    if len(events) != sum(RISK_TARGETS.values()):
        raise RuntimeError(
            f"Expected {sum(RISK_TARGETS.values())} events, "
            f"but matched {len(events)}."
        )

    producer = KafkaProducer(
        bootstrap_servers=KAFKA_SERVER,
        value_serializer=lambda value: json.dumps(value).encode("utf-8"),
        acks="all",
        retries=5
    )

    try:
        for index, event in enumerate(events, start=1):
            producer.send(TOPIC_NAME, value=event).get(timeout=30)
            print(
                f"Sent {index}/{len(events)}: "
                f"Machine {event['machineID']} "
                f"at {event['datetime']}"
            )
            time.sleep(DELAY_SECONDS)
    finally:
        producer.flush()
        producer.close()

    print(f"\nTotal messages sent: {len(events)}")
    print("Selected risk targets:", RISK_TARGETS)


if __name__ == "__main__":
    main()
