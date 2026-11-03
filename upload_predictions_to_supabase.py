
import csv
import glob
import hashlib
import os
from datetime import datetime

import psycopg
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

required = [
    "SUPABASE_DB_HOST",
    "SUPABASE_DB_PORT",
    "SUPABASE_DB_NAME",
    "SUPABASE_DB_USER",
    "SUPABASE_DB_PASSWORD",
]

missing = [key for key in required if not os.getenv(key)]
if missing:
    raise RuntimeError("Missing .env settings: " + ", ".join(missing))

files = glob.glob(
    os.path.join(BASE_DIR, "outputs", "live_predictions", "part-*.csv")
)

if not files:
    raise FileNotFoundError("No prediction CSV found.")

csv_file = max(files, key=os.path.getmtime)
print("Reading:", os.path.basename(csv_file))

records = []

with open(csv_file, newline="", encoding="utf-8") as file:
    reader = csv.DictReader(file)

    required_columns = {
        "machineID",
        "datetime",
        "volt",
        "rotate",
        "pressure",
        "vibration",
        "predicted_failure",
        "failure_probability",
        "risk_level",
    }

    if not required_columns.issubset(reader.fieldnames or []):
        raise ValueError("Unexpected prediction CSV columns.")

    for row in reader:
        machine_id = int(row["machineID"])
        event_time = row["datetime"].strip()

        timestamp = datetime.fromisoformat(event_time)

        event_key = hashlib.sha256(
            f"{machine_id}|{event_time}".encode("utf-8")
        ).hexdigest()

        failure_value = row["predicted_failure"].strip().upper()
        predicted_failure = (
            1 if failure_value in {"YES", "TRUE", "1"} else 0
        )

        records.append((
            event_key,
            timestamp,
            machine_id,
            float(row["volt"]),
            float(row["rotate"]),
            float(row["pressure"]),
            float(row["vibration"]),
            row["risk_level"].strip().upper(),
            predicted_failure,
            float(row["failure_probability"]),
        ))

if not records:
    raise RuntimeError("Prediction CSV has no records.")

insert_sql = """
INSERT INTO public.machine_predictions (
    event_key,
    recorded_at,
    machine_id,
    voltage,
    rotation,
    pressure,
    vibration,
    risk_level,
    failure_prediction,
    risk_probability
)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (event_key) WHERE event_key IS NOT NULL
DO NOTHING
"""

connection_settings = {
    "host": os.environ["SUPABASE_DB_HOST"],
    "port": int(os.environ["SUPABASE_DB_PORT"]),
    "dbname": os.environ["SUPABASE_DB_NAME"],
    "user": os.environ["SUPABASE_DB_USER"],
    "password": os.environ["SUPABASE_DB_PASSWORD"],
    "sslmode": "require",
    "connect_timeout": 15,
}

with psycopg.connect(**connection_settings) as conn:
    with conn.cursor() as cur:
        cur.executemany(insert_sql, records)
        inserted = cur.rowcount

        cur.execute("""
            SELECT
                COUNT(*),
                COUNT(DISTINCT event_key)
            FROM public.machine_predictions
        """)
        total, distinct_keys = cur.fetchone()

print("Upload completed.")
print("Rows in selected file:", len(records))
print("Rows newly inserted:", inserted)
print("Rows skipped as duplicates:", len(records) - inserted)
print("Total database rows:", total)
print("Distinct event keys:", distinct_keys)
